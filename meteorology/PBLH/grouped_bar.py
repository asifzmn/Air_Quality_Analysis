import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import pearsonr
import logging
from typing import Tuple, Dict, List
import warnings
import plotly.graph_objects as go

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
    # RADIUS_VALUES = [10, 25, 50, 75, 100, 150]  # Test multiple radii
    RADIUS_VALUES = [20, 30, 40]  # Test multiple radii

    # Lag analysis parameters
    MAX_LAG_DAYS = 5  # Test correlations up to 5 days lag


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


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_pblh_data(pblh_file: str) -> pd.DataFrame:
    """
    Load and clean PBLH data

    Args:
        pblh_file: Path to PBLH CSV file

    Returns:
        Cleaned PBLH dataframe
    """
    logger.info("=" * 70)
    logger.info("LOADING AND CLEANING PBLH DATA")
    logger.info("=" * 70)

    # Load PBLH data
    pblh_df = pd.read_csv(pblh_file, parse_dates=['valid_time']).head(100000)

    initial_count = len(pblh_df)
    logger.info(f"Initial PBLH records: {initial_count:,}")

    # Extract date for daily aggregation
    pblh_df['date'] = pblh_df['valid_time'].dt.date

    logger.info(f"Date range: {pblh_df['date'].min()} to {pblh_df['date'].max()}")
    logger.info(f"Unique locations: {pblh_df.groupby(['latitude', 'longitude']).ngroups:,}")
    logger.info(f"Mean PBLH: {pblh_df['blh'].mean():.2f} m")
    logger.info(f"Std PBLH: {pblh_df['blh'].std():.2f} m")
    logger.info("")

    return pblh_df


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
# PBLH-PM2.5 MATCHING
# ============================================================================

def aggregate_pblh_by_zone(pblh_df: pd.DataFrame, metadata_df: pd.DataFrame,
                           radius_km: float) -> pd.DataFrame:
    """
    Aggregate PBLH by date and zone using distance-based matching

    Args:
        pblh_df: Cleaned PBLH dataframe
        metadata_df: Zone metadata with coordinates
        radius_km: Search radius in kilometers

    Returns:
        Dataframe with daily PBLH by zone
    """
    logger.info(f"Processing radius: {radius_km} km")

    # Initialize result dictionary
    pblh_aggregated = {}

    for zone_idx, zone_row in metadata_df.iterrows():
        zone_name = zone_idx
        zone_lat = zone_row['Latitude']
        zone_lon = zone_row['Longitude']

        # Calculate distances to all PBLH readings
        distances = pblh_df.apply(
            lambda row: haversine_distance(zone_lat, zone_lon, row['latitude'], row['longitude']),
            axis=1
        )

        # Filter PBLH readings within radius
        nearby_pblh = pblh_df[distances <= radius_km].copy()

        if len(nearby_pblh) > 0:
            # Aggregate PBLH by date (mean of all readings within radius)
            daily_pblh = nearby_pblh.groupby('date')['blh'].mean()
            pblh_aggregated[zone_name] = daily_pblh

    # Convert to DataFrame
    pblh_by_zone = pd.DataFrame(pblh_aggregated).fillna(np.nan)

    logger.info(f"  Zones with PBLH data: {pblh_by_zone.notna().any(axis=0).sum()}/{len(pblh_by_zone.columns)}")

    return pblh_by_zone


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_lagged_correlation(pblh_series: pd.Series, pm25_series: pd.Series,
                                 max_lag: int = 5) -> Dict[int, Tuple[float, float]]:
    """
    Calculate correlation between PBLH and PM2.5 with various time lags

    Args:
        pblh_series: Daily PBLH time series
        pm25_series: Daily PM2.5 time series
        max_lag: Maximum lag in days to test

    Returns:
        Dictionary of {lag_days: (correlation, p_value)}
    """
    correlations = {}

    for lag in range(max_lag + 1):
        # Shift PBLH data backward (PBLH happens 'lag' days before PM2.5)
        pblh_lagged = pblh_series.shift(lag)

        # Align series and remove NaN
        aligned = pd.DataFrame({
            'pblh': pblh_lagged,
            'pm25': pm25_series
        }).dropna()

        if len(aligned) > 10:  # Need sufficient data
            corr, pval = pearsonr(aligned['pblh'], aligned['pm25'])
            correlations[lag] = (corr, pval)

    return correlations


def analyze_correlations_for_radius(pblh_by_zone: pd.DataFrame, pm25_daily: pd.DataFrame,
                                    radius_km: float) -> List[Dict]:
    """
    Perform correlation analysis for a specific radius

    Args:
        pblh_by_zone: Daily PBLH by zone
        pm25_daily: Daily PM2.5 by zone
        radius_km: Search radius used

    Returns:
        List of result dictionaries
    """
    results = []

    # Get common zones
    common_zones = list(set(pblh_by_zone.columns) & set(pm25_daily.columns))

    for zone in common_zones:
        # Get aligned date ranges
        pblh_series = pblh_by_zone[zone]
        pm25_series = pm25_daily[zone]

        # Find common date range
        common_dates = pblh_series.index.intersection(pm25_series.index)
        pblh_aligned = pblh_series.loc[common_dates]
        pm25_aligned = pm25_series.loc[common_dates]

        # Remove NaN values
        valid_mask = pblh_aligned.notna() & pm25_aligned.notna()
        pblh_aligned = pblh_aligned[valid_mask]
        pm25_aligned = pm25_aligned[valid_mask]

        if len(pblh_aligned) < 10:
            continue

        # Calculate lagged correlations
        lag_correlations = calculate_lagged_correlation(
            pblh_aligned, pm25_aligned, config.MAX_LAG_DAYS
        )

        # Store results for each lag
        for lag, (corr, pval) in lag_correlations.items():
            results.append({
                'Radius_km': radius_km,
                'Zone': zone,
                'Lag_Days': lag,
                'Correlation': corr,
                'P_Value': pval,
                'Significant': pval < 0.05,
                'Mean_PBLH': pblh_aligned.mean(),
                'Mean_PM25': pm25_aligned.mean(),
                'N_Observations': len(pblh_aligned)
            })

    return results


# ============================================================================
# MULTI-RADIUS ANALYSIS PIPELINE
# ============================================================================

def run_multi_radius_analysis(pblh_file: str,
                              pm25_df: pd.DataFrame,
                              metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline for multi-radius PBLH-PM2.5 correlation analysis

    Args:
        pblh_file: Path to PBLH CSV file
        pm25_df: PM2.5 time series
        metadata_df: Zone metadata

    Returns:
        Dataframe with correlation results for all radii and lags
    """
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-RADIUS PBLH-PM2.5 CORRELATION ANALYSIS")
    logger.info("=" * 70 + "\n")

    # Step 1: Load PBLH data
    pblh_data = load_and_clean_pblh_data(pblh_file)

    # Step 2: Prepare PM2.5 data
    pm25_daily = prepare_pm25_data(pm25_df, metadata_df)

    # Step 3: Run analysis for each radius
    all_results = []

    logger.info("=" * 70)
    logger.info("RUNNING ANALYSIS FOR MULTIPLE RADII")
    logger.info("=" * 70)

    for radius in config.RADIUS_VALUES:
        logger.info(f"\n--- RADIUS: {radius} km ---")

        # Spatial matching and aggregation
        pblh_by_zone = aggregate_pblh_by_zone(pblh_data, metadata_df, radius)

        # Correlation analysis
        radius_results = analyze_correlations_for_radius(pblh_by_zone, pm25_daily, radius)
        all_results.extend(radius_results)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total correlations calculated: {len(results_df)}")
    logger.info(f"Significant correlations (p<0.05): {results_df['Significant'].sum()}")
    logger.info(f"Mean correlation: {results_df['Correlation'].mean():.4f}")
    logger.info(f"Mean absolute correlation: {results_df['Correlation'].abs().mean():.4f}")

    return results_df


# ============================================================================
# AGGREGATION AND VISUALIZATION
# ============================================================================

def create_summary_for_visualization(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated summary for grouped bar chart

    Args:
        results_df: Full results dataframe

    Returns:
        Summary dataframe with avg correlation per radius/lag
    """
    summary = results_df.groupby(['Radius_km', 'Lag_Days']).agg({
        'Correlation': 'mean',
        'Significant': 'sum',
        'Zone': 'count'
    }).reset_index()

    summary.columns = ['Radius_km', 'Lag_Days', 'Avg_Correlation', 'Num_Significant', 'Num_Zones']

    return summary


def create_grouped_bar_chart(summary_df: pd.DataFrame, output_file: str = 'pblh_pm25_grouped_bar.html'):
    """
    Create grouped bar chart with Plotly

    Args:
        summary_df: Summary dataframe
        output_file: Output HTML file path
    """
    logger.info("\n" + "=" * 70)
    logger.info("CREATING GROUPED BAR CHART")
    logger.info("=" * 70)

    # Create figure
    fig = go.Figure()

    # Define blue color palette (darker to lighter blues)
    blue_colors = [
        '#08306b',  # Lag 0 - darkest blue
        '#08519c',  # Lag 1
        '#2171b5',  # Lag 2
        '#4292c6',  # Lag 3
        '#6baed6',  # Lag 4
        '#9ecae1',  # Lag 5 - lightest blue
    ]

    # Add bars for each lag day
    for lag in range(config.MAX_LAG_DAYS + 1):
        lag_data = summary_df[summary_df['Lag_Days'] == lag]

        fig.add_trace(go.Bar(
            name=f'Lag {lag}d',
            x=lag_data['Radius_km'],
            y=lag_data['Avg_Correlation'],
            marker_color=blue_colors[lag],
            text=lag_data['Avg_Correlation'].round(3),
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate=(
                    f'<b>Lag {lag} days</b><br>' +
                    'Radius: %{x} km<br>' +
                    'Avg Correlation: %{y:.4f}<br>' +
                    '<extra></extra>'
            )
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'PBLH-PM2.5 Correlation Analysis: Impact of Radius and Lag Time<br><sub>Expected: Negative Correlation (Higher PBLH → Lower PM2.5)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis=dict(
            title='Search Radius (km)',
            tickmode='linear',
            tick0=0,
            dtick=25,
            gridcolor='#ecf0f1',
            title_font={'size': 14, 'color': '#34495e'}
        ),
        yaxis=dict(
            title='Average Correlation Coefficient',
            gridcolor='#ecf0f1',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#95a5a6',
            title_font={'size': 14, 'color': '#34495e'}
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title='Time Lag',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='right',
            x=1.15,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        bargap=0.15,
        bargroupgap=0.1,
        height=600,
        width=1200,
        font={'family': 'Arial, sans-serif', 'color': '#2c3e50'}
    )

    # Save to HTML
    fig.write_html(output_file)
    logger.info(f"Chart saved to: {output_file}")

    # Display in notebook if available
    fig.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    logger.info("Loading data...")

    pblh_file = aq_directory + '/PBLH/bangladesh_pblh_2018_2021.csv'

    metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data_4_years()

    # Run multi-radius analysis
    # results = run_multi_radius_analysis(pblh_file, series, metadata)
    results = run_multi_radius_analysis(pblh_file, series, metadata_country)

    # Save full results
    results.to_csv('pblh_pm25_multiradius_results.csv', index=False)
    logger.info("\nFull results saved to: pblh_pm25_multiradius_results.csv")

    # Create summary for visualization
    summary = create_summary_for_visualization(results)
    summary.to_csv('pblh_pm25_summary.csv', index=False)
    logger.info("Summary saved to: pblh_pm25_summary.csv")

    # Create grouped bar chart
    create_grouped_bar_chart(summary, 'pblh_pm25_grouped_bar.html')

    # Display summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)
    print("\nAverage Correlation by Radius and Lag:")
    print(summary.pivot(index='Radius_km', columns='Lag_Days', values='Avg_Correlation').round(4))

    print("\nNumber of Significant Correlations by Radius and Lag:")
    print(summary.pivot(index='Radius_km', columns='Lag_Days', values='Num_Significant'))

    # Interpretation note
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION GUIDE")
    logger.info("=" * 70)
    logger.info("Expected: NEGATIVE correlations (higher PBLH → better dispersion → lower PM2.5)")
    logger.info("Positive correlations may indicate other factors dominating the relationship")