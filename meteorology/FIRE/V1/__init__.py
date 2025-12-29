import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import logging
from typing import Tuple, Dict
import warnings

from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
from paths import aq_directory

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the analysis"""

    # Spatial matching parameters (in kilometers)
    BASE_RADIUS_KM = 15  # Base radius for all zones
    FOREST_PROXIMITY_BONUS_KM = 50  # Extra radius for forest-adjacent zones
    MAX_RADIUS_KM = 150  # Maximum search radius

    # Fire filtering thresholds
    MIN_CONFIDENCE = 50  # Minimum confidence level (0-100)
    # MIN_BRIGHTNESS = 300  # Minimum brightness (Kelvin) - filters out weak fires
    # MIN_FRP = 5.0  # Minimum Fire Radiative Power (MW) - significant fires only

    MIN_BRIGHTNESS = 0
    MIN_FRP = 0.0

    # Lag analysis parameters
    MAX_LAG_DAYS = 5  # Test correlations up to 5 days lag

    # Forest-adjacent zones (based on Bangladesh geography)
    FOREST_ZONES = {
        'Chittagong': ['Chittagong_Chittagong', 'Rangamati_Chittagong',
                       'Cox\'s Bazar_Chittagong', 'Bandarban_Chittagong',
                       'Satkania_Chittagong', 'Comilla_Chittagong'],
        'Sylhet': ['Sylhet_Sylhet', 'Maulvi Bazar_Sylhet'],
        'Khulna': ['Khulna_Khulna', 'Satkhira_Khulna', 'Bagerhat_Khulna']
    }


config = Config()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km)

    Args:
        lat1, lon1: Coordinates of point 1
        lat2, lon2: Coordinates of point 2

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def get_zone_search_radius(zone_name: str) -> float:
    """
    Determine appropriate search radius for a zone based on forest proximity

    Args:
        zone_name: Name of the PM2.5 monitoring zone

    Returns:
        Search radius in kilometers
    """
    # Check if zone is forest-adjacent
    is_forest_adjacent = any(
        zone_name in zones
        for zones in config.FOREST_ZONES.values()
    )

    if is_forest_adjacent:
        radius = min(config.BASE_RADIUS_KM + config.FOREST_PROXIMITY_BONUS_KM,
                     config.MAX_RADIUS_KM)
        logger.debug(f"{zone_name}: Forest-adjacent zone, radius = {radius} km")
    else:
        radius = config.BASE_RADIUS_KM
        logger.debug(f"{zone_name}: Urban zone, radius = {radius} km")

    return radius


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_fire_data(fire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and clean fire data with quality filtering

    Args:
        fire_df: Raw fire dataframe

    Returns:
        Cleaned fire dataframe
    """
    logger.info("=" * 70)
    logger.info("LOADING AND CLEANING FIRE DATA")
    logger.info("=" * 70)

    initial_count = len(fire_df)
    logger.info(f"Initial fire records: {initial_count:,}")

    # Create datetime column
    fire_df['datetime'] = pd.to_datetime(fire_df['acq_date']) + pd.to_timedelta(
        fire_df['acq_time'].astype(str).str.zfill(4).str[:2] + ':' + fire_df['acq_time'].astype(str).str.zfill(4).str[
                                                                     2:] + ':00')
    fire_df['date'] = fire_df['datetime'].dt.date

    # print(fire_df['confidence'])
    # Filter by confidence
    # fire_df = fire_df[fire_df['confidence'] >= config.MIN_CONFIDENCE]
    logger.info(
        f"After confidence filter (>={config.MIN_CONFIDENCE}): {len(fire_df):,} records ({len(fire_df) / initial_count * 100:.1f}%)")

    # Filter by brightness
    fire_df = fire_df[fire_df['brightness'] >= config.MIN_BRIGHTNESS]
    logger.info(
        f"After brightness filter (>={config.MIN_BRIGHTNESS}K): {len(fire_df):,} records ({len(fire_df) / initial_count * 100:.1f}%)")

    # Filter by FRP
    fire_df = fire_df[fire_df['frp'] >= config.MIN_FRP]
    logger.info(
        f"After FRP filter (>={config.MIN_FRP} MW): {len(fire_df):,} records ({len(fire_df) / initial_count * 100:.1f}%)")

    logger.info(
        f"Final retained fire records: {len(fire_df):,} ({len(fire_df) / initial_count * 100:.1f}% of original)")
    logger.info(f"Date range: {fire_df['date'].min()} to {fire_df['date'].max()}")
    logger.info("")

    return fire_df


def prepare_pm25_data(pm25_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare PM2.5 data with daily aggregation

    Args:
        pm25_df: PM2.5 time series dataframe
        metadata_df: Metadata with zone coordinates

    Returns:
        Daily PM2.5 dataframe
    """
    logger.info("=" * 70)
    logger.info("PREPARING PM2.5 DATA")
    logger.info("=" * 70)

    logger.info(f"PM2.5 zones: {len(pm25_df.columns)}")
    logger.info(f"PM2.5 date range: {pm25_df.index.min()} to {pm25_df.index.max()}")
    logger.info(f"Total hourly records: {len(pm25_df):,}")

    # Daily aggregation
    pm25_daily = pm25_df.resample('D').mean()
    logger.info(f"Daily aggregated records: {len(pm25_daily):,}")
    logger.info("")

    return pm25_daily


# ============================================================================
# FIRE-PM2.5 MATCHING
# ============================================================================

def aggregate_fires_by_zone(fire_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fires by date and zone using distance-based matching

    Args:
        fire_df: Cleaned fire dataframe
        metadata_df: Zone metadata with coordinates

    Returns:
        Dataframe with daily FRP by zone
    """
    logger.info("=" * 70)
    logger.info("SPATIAL MATCHING: FIRES TO PM2.5 ZONES")
    logger.info("=" * 70)

    # Initialize result dictionary
    fire_aggregated = {}

    for zone_idx, zone_row in metadata_df.iterrows():
        zone_name = zone_idx
        zone_lat = zone_row['Latitude']
        zone_lon = zone_row['Longitude']
        # search_radius = get_zone_search_radius(zone_name)
        search_radius = config.BASE_RADIUS_KM  # Uniform radius for simplicity

        logger.info(f"Processing: {zone_name}")
        logger.info(f"  Location: ({zone_lat:.4f}, {zone_lon:.4f})")
        logger.info(f"  Search radius: {search_radius} km")

        # Calculate distances to all fires
        distances = fire_df.apply(
            lambda row: haversine_distance(zone_lat, zone_lon, row['latitude'], row['longitude']),
            axis=1
        )

        # Filter fires within radius
        nearby_fires = fire_df[distances <= search_radius].copy()

        logger.info(f"  Fires within radius: {len(nearby_fires):,}")

        if len(nearby_fires) > 0:
            # Aggregate FRP by date
            daily_frp = nearby_fires.groupby('date')['frp'].sum()
            fire_aggregated[zone_name] = daily_frp
            logger.info(f"  Days with fire activity: {len(daily_frp)}")
            logger.info(f"  Mean daily FRP: {daily_frp.mean():.2f} MW")
            logger.info(f"  Max daily FRP: {daily_frp.max():.2f} MW")
        else:
            logger.info(f"  No fires detected in this zone")

        logger.info("")

    # Convert to DataFrame
    fire_by_zone = pd.DataFrame(fire_aggregated).fillna(0)

    logger.info("=" * 70)
    logger.info(f"FIRE AGGREGATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total zones: {len(fire_by_zone.columns)}")
    logger.info(f"Zones with fires: {(fire_by_zone.sum() > 0).sum()}")
    logger.info(f"Date range: {fire_by_zone.index.min()} to {fire_by_zone.index.max()}")
    logger.info("")

    return fire_by_zone


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_lagged_correlation(fire_series: pd.Series, pm25_series: pd.Series,
                                 max_lag: int = 5) -> Dict[int, Tuple[float, float]]:
    """
    Calculate correlation between fire and PM2.5 with various time lags

    Args:
        fire_series: Daily FRP time series
        pm25_series: Daily PM2.5 time series
        max_lag: Maximum lag in days to test

    Returns:
        Dictionary of {lag_days: (correlation, p_value)}
    """
    correlations = {}

    for lag in range(max_lag + 1):
        # Shift fire data backward (fires happen 'lag' days before PM2.5)
        fire_lagged = fire_series.shift(lag)

        # Align series and remove NaN
        aligned = pd.DataFrame({
            'fire': fire_lagged,
            'pm25': pm25_series
        }).dropna()

        if len(aligned) > 10:  # Need sufficient data
            corr, pval = pearsonr(aligned['fire'], aligned['pm25'])
            correlations[lag] = (corr, pval)

    return correlations


def analyze_correlations(fire_by_zone: pd.DataFrame, pm25_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive correlation analysis with lag effects

    Args:
        fire_by_zone: Daily FRP by zone
        pm25_daily: Daily PM2.5 by zone

    Returns:
        Dataframe with correlation results
    """
    logger.info("=" * 70)
    logger.info("CORRELATION ANALYSIS WITH LAG EFFECTS")
    logger.info("=" * 70)

    results = []

    # Get common zones
    common_zones = list(set(fire_by_zone.columns) & set(pm25_daily.columns))
    logger.info(f"Analyzing {len(common_zones)} common zones")
    logger.info("")

    for zone in common_zones:
        logger.info(f"Analyzing: {zone}")

        # Get aligned date ranges
        fire_series = fire_by_zone[zone]
        pm25_series = pm25_daily[zone]

        # Find common date range
        common_dates = fire_series.index.intersection(pm25_series.index)
        fire_aligned = fire_series.loc[common_dates]
        pm25_aligned = pm25_series.loc[common_dates]

        logger.info(f"  Common dates: {len(common_dates)}")
        logger.info(f"  Fire activity days: {(fire_aligned > 0).sum()}")
        logger.info(f"  Mean daily FRP: {fire_aligned.mean():.2f} MW")
        logger.info(f"  Mean PM2.5: {pm25_aligned.mean():.2f} µg/m³")

        # Calculate lagged correlations
        lag_correlations = calculate_lagged_correlation(
            fire_aligned, pm25_aligned, config.MAX_LAG_DAYS
        )

        # Log results for each lag
        logger.info(f"  Lag correlations:")
        for lag, (corr, pval) in lag_correlations.items():
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            logger.info(f"    Lag {lag}d: r={corr:+.3f}, p={pval:.4f} {significance}")

            results.append({
                'Zone': zone,
                'Lag_Days': lag,
                'Correlation': corr,
                'P_Value': pval,
                'Significant': pval < 0.05,
                'Fire_Days': (fire_aligned > 0).sum(),
                'Mean_FRP': fire_aligned.mean(),
                'Mean_PM25': pm25_aligned.mean(),
                'N_Observations': len(common_dates)
            })

        logger.info("")

    results_df = pd.DataFrame(results)

    # Summary statistics
    logger.info("=" * 70)
    logger.info("CORRELATION ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total correlations calculated: {len(results_df)}")
    logger.info(f"Significant correlations (p<0.05): {results_df['Significant'].sum()}")
    logger.info(f"Significant correlation rate: {results_df['Significant'].mean() * 100:.1f}%")
    logger.info("")

    # Best correlations by lag
    logger.info("STRONGEST CORRELATIONS BY LAG:")
    for lag in range(config.MAX_LAG_DAYS + 1):
        lag_data = results_df[results_df['Lag_Days'] == lag]
        if len(lag_data) > 0:
            best = lag_data.loc[lag_data['Correlation'].abs().idxmax()]
            logger.info(f"  Lag {lag}d: {best['Zone']}, r={best['Correlation']:+.3f}, p={best['P_Value']:.4f}")

    logger.info("")

    return results_df


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_fire_pm25_correlation_analysis(fire_df: pd.DataFrame,
                                       pm25_df: pd.DataFrame,
                                       metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline for fire-PM2.5 correlation analysis

    Args:
        fire_df: Raw fire data
        pm25_df: PM2.5 time series
        metadata_df: Zone metadata

    Returns:
        Dataframe with correlation results
    """
    logger.info("\n" + "=" * 70)
    logger.info("FIRE-PM2.5 CORRELATION ANALYSIS PIPELINE")
    logger.info("=" * 70 + "\n")

    # Step 1: Clean fire data
    fire_clean = load_and_clean_fire_data(fire_df)

    # Step 2: Prepare PM2.5 data
    pm25_daily = prepare_pm25_data(pm25_df, metadata_df)

    # Step 3: Spatial matching and aggregation
    fire_by_zone = aggregate_fires_by_zone(fire_clean, metadata_df)

    # Step 4: Correlation analysis
    results = analyze_correlations(fire_by_zone, pm25_daily)

    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 70)

    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    # fire_data = pd.read_csv('fire_data.csv')
    # pm25_data = pd.read_csv('pm25_data.csv', index_col='time', parse_dates=True)
    # metadata = pd.read_csv('metadata.csv', index_col='index')

    fire_data = pd.read_csv(aq_directory+'/NASA FIRMS DATASET/DL_FIRE_J1V-C2_668887/fire_archive_J1V-C2_668887.csv',parse_dates=['acq_date'])

    # print(fire_data.info())
    # exit()

    metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data_4_years()

    # Run analysis
    results = run_fire_pm25_correlation_analysis(fire_data, series, metadata)

    # Save results
    results.to_csv('fire_pm25_correlation_results.csv', index=False)

    # Get significant correlations
    significant = results[results['Significant'] == True].sort_values('Correlation', ascending=False)
    print("\nTop 10 Significant Correlations:")
    print(significant.head(10))

    # pass