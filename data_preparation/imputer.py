import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# import xgboost as xgb

from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer


def fill_lag_with_forward_lookup(df, col, lag_hours=24):
    """
    Fill missing lag values by looking forward in time day-by-day.
    Searches next day, then next day, then next day... until valid value or out of index.

    Args:
        df: DataFrame with time index
        col: Column name to create lag for
        lag_hours: Hours to shift backward (default 24 for lag24)

    Returns:
        Series with lag values, filled forward when possible
    """
    # Create initial lag
    lag_col = df[col].shift(lag_hours)

    # Find missing indices
    missing_mask = lag_col.isna()
    missing_indices = lag_col[missing_mask].index

    # For each missing value, look forward day by day
    for idx in missing_indices:
        day_offset = 1
        while True:
            # Calculate forward lookup time (add days)
            forward_idx = idx + pd.Timedelta(hours=day_offset * lag_hours)

            # Check if that index exists
            if forward_idx not in df.index:
                break  # Out of index, stop searching

            # Check if value is valid
            value = df.loc[forward_idx, col]
            if pd.notna(value):
                lag_col.loc[idx] = value
                break  # Found valid value, stop searching

            day_offset += 1

    return lag_col


def prepare_forward_lookup_features(df):
    df_with_features = df.copy()

    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['day'] = df_with_features.index.day
    df_with_features['month'] = df_with_features.index.month
    df_with_features['dayofweek'] = df_with_features.index.dayofweek
    df_with_features['dayofyear'] = df_with_features.index.dayofyear

    # Cyclical encoding
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['dayofweek_sin'] = np.sin(2 * np.pi * df_with_features['dayofweek'] / 7)
    df_with_features['dayofweek_cos'] = np.cos(2 * np.pi * df_with_features['dayofweek'] / 7)

    for col in df.columns:
        df_with_features[f'{col}_lag1'] = df[col].shift(1)
        df_with_features[f'{col}_lag-1'] = df[col].shift(-1)
        df_with_features[f'{col}_lag24--'] = fill_lag_with_forward_lookup(df, col, lag_hours=24)
        df_with_features[f'{col}_lag-24--'] = fill_lag_with_forward_lookup(df, col, lag_hours=-24)

    return df_with_features


def ImputeSimpleMICE(df, max_iter=10):
    """
    MICE (Multiple Imputation by Chained Equations) with default estimator
    Simple but effective iterative imputation

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    max_iter : int
        Number of imputation rounds

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_filled = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    return df_filled


def imputeWithLinearRegression(df, max_iter=10):
    """
    MICE (Multiple Imputation by Chained Equations) with Linear Regression estimator
    Simple but effective iterative imputation

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    max_iter : int
        Number of imputation rounds

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    df_with_features = prepare_forward_lookup_features(df)

    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=max_iter,
        random_state=42
    )
    df_filled_full = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )
    return df_filled_full[df.columns]


def ImputeWithBayesianRidge(df, max_iter=10):
    """
    Bayesian Ridge Regression imputation
    Learns patterns from other zones

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    max_iter : int
        Number of imputation rounds

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    from sklearn.linear_model import BayesianRidge

    df_with_features = prepare_forward_lookup_features(df)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=42
    )
    df_filled_full = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )
    return df_filled_full[df.columns]


def ImputeWithKNN(df, n_neighbors=5):
    """
    K-Nearest Neighbors imputation - finds similar time periods

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    n_neighbors : int
        Number of nearest neighbors to use

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    df_filled = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    return df_filled


def ImputeWithFeaturesKNN(df, zone_coords=None, n_neighbors=5):
    """
    KNN imputation with engineered features (time features + spatial features)

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    zone_coords : dict, optional
        Coordinates for each zone for distance features
    n_neighbors : int
        Number of nearest neighbors

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    # Add time-based features
    df_with_features = df.copy()
    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['day'] = df_with_features.index.day
    df_with_features['month'] = df_with_features.index.month
    df_with_features['dayofweek'] = df_with_features.index.dayofweek
    df_with_features['dayofyear'] = df_with_features.index.dayofyear

    # Add cyclical encoding for time features
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)

    # Impute
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    # Return only the original PM2.5 columns
    return df_imputed[df.columns]


def ImputeWithFeaturesKNNForwardLookup(df, n_neighbors):
    df_with_features = prepare_forward_lookup_features(df)

    # Impute
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    # Return only the original PM2.5 columns
    return df_imputed[df.columns]


def ImputeWithRandomForest(df, n_estimators=100, max_iter=10):
    """
    Iterative imputation using Random Forest
    Learns patterns from other zones and features

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    n_estimators : int
        Number of trees in the forest
    max_iter : int
        Number of imputation rounds

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=42),
        max_iter=max_iter,
        random_state=42
    )
    df_filled = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    return df_filled


def ImputeWithFeaturesRF(df, zone_coords=None, n_estimators=100, max_iter=10):
    """
    Random Forest imputation with engineered features
    Most powerful approach - learns complex patterns

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with zones as columns, datetime as index
    zone_coords : dict, optional
        Coordinates for each zone
    n_estimators : int
        Number of trees
    max_iter : int
        Number of imputation rounds

    Returns:
    --------
    pd.DataFrame : Imputed data
    """
    # Add time-based features
    df_with_features = df.copy()
    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['day'] = df_with_features.index.day
    df_with_features['month'] = df_with_features.index.month
    df_with_features['dayofweek'] = df_with_features.index.dayofweek
    df_with_features['dayofyear'] = df_with_features.index.dayofyear
    # df_with_features['week'] = df_with_features.index.isocalendar().week

    # Cyclical encoding
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['dayofweek_sin'] = np.sin(2 * np.pi * df_with_features['dayofweek'] / 7)
    df_with_features['dayofweek_cos'] = np.cos(2 * np.pi * df_with_features['dayofweek'] / 7)

    # Add lagged features (values from previous hours)
    for col in df.columns:
        df_with_features[f'{col}_lag1'] = df[col].shift(1)
        df_with_features[f'{col}_lag24'] = df[col].shift(24)
        df_with_features[f'{col}_lag168'] = df[col].shift(168)  # 1 week

    # Impute
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        max_iter=max_iter,
        random_state=42
    )

    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    # Return only the original PM2.5 columns
    return df_imputed[df.columns]


def ImputeWithRandomForest_FeaturesForwardLookup(df, n_estimators=100, max_iter=10):
    df_with_features = prepare_forward_lookup_features(df)

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        max_iter=max_iter,
        random_state=42
    )

    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    # Return only the original PM2.5 columns
    return df_imputed[df.columns]


def ImputeWithRandomForest_FeaturesForwardLookupWrapper(df_test, n_estimators, max_iter):
    return pd.concat(
        [ImputeWithRandomForest_FeaturesForwardLookup(df_test[[zone]], n_estimators=n_estimators, max_iter=max_iter) for
         zone in df_test.columns], axis=1)


def ImputeWithLightGBM(df, max_iter=15):
    """
    LightGBM - Faster than XGBoost, often similar accuracy
    Very efficient for large datasets
    """
    imputer = IterativeImputer(
        estimator=lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        max_iter=max_iter,
        random_state=42
    )

    df_filled = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    return df_filled


def ImputeWithFeaturesLightGBM(df, max_iter=15):
    """
    LightGBM with time features - FASTEST + VERY ACCURATE
    Best balance of speed and accuracy
    """
    df_with_features = df.copy()

    # Time features
    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['dayofweek'] = df_with_features.index.dayofweek
    df_with_features['month'] = df_with_features.index.month
    df_with_features['dayofyear'] = df_with_features.index.dayofyear
    df_with_features['is_weekend'] = (df_with_features.index.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)

    # Lagged features
    for col in df.columns[:5]:
        df_with_features[f'{col}_lag1'] = df[col].shift(1)
        df_with_features[f'{col}_lag24'] = df[col].shift(24)

    imputer = IterativeImputer(
        estimator=lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        max_iter=max_iter,
        random_state=42
    )

    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    return df_imputed[df.columns]


def ImputeWithFeaturesLightGBMForwardLookup(df, max_iter=15):
    """
    LightGBM with time features - FASTEST + VERY ACCURATE
    Best balance of speed and accuracy
    """
    df_with_features = prepare_forward_lookup_features(df)

    imputer = IterativeImputer(
        estimator=lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        max_iter=max_iter,
        random_state=42
    )

    # print(df_with_features.iloc[:,-2:])
    print(df_with_features.columns)
    # return

    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_with_features),
        index=df_with_features.index,
        columns=df_with_features.columns
    )

    return df_imputed[df.columns]


def ImputeWithFeaturesLightGBMForwardLookupWrapper(df_test, max_iter):
    return pd.concat([ImputeWithFeaturesLightGBMForwardLookup(df_test[[zone]]) for zone in df_test.columns], axis=1)


# def ImputeWithXGBoost(df, max_iter=10):
#     """
#     Iterative imputation using XGBoost
#     Very powerful for capturing complex patterns
#
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Time series data with zones as columns, datetime as index
#     max_iter : int
#         Number of imputation rounds
#
#     Returns:
#     --------
#     pd.DataFrame : Imputed data
#     """
#     imputer = IterativeImputer(
#         estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
#         max_iter=max_iter,
#         random_state=42
#     )
#     df_filled = pd.DataFrame(
#         imputer.fit_transform(df),
#         index=df.index,
#         columns=df.columns
#     )
#     return df_filled

# Example usage:


"""
import pandas as pd

# Load your data
df = pd.read_csv('pm25_data.csv', index_col=0, parse_dates=True)

# Method 1: KNN (Simple and Fast) - RECOMMENDED FOR START
df_filled = ImputeWithKNN(df, n_neighbors=5)

# Method 2: KNN with Time Features (Better)
df_filled = ImputeWithFeaturesKNN(df, n_neighbors=5)

# Method 3: Random Forest (Most Powerful) - BEST ACCURACY
df_filled = ImputeWithRandomForest(df, n_estimators=100, max_iter=10)

# Method 4: Random Forest with Features (MOST POWERFUL)
df_filled = ImputeWithFeaturesRF(df, n_estimators=100, max_iter=10)

# Method 5: XGBoost (Very Good)
df_filled = ImputeWithXGBoost(df, max_iter=10)

# Method 6: MICE (Simple Iterative)
df_filled = ImputeSimpleMICE(df, max_iter=10)

# Check results
print(f"Missing before: {df.isna().sum().sum()}")
print(f"Missing after: {df_filled.isna().sum().sum()}")

# Save
df_filled.to_csv('pm25_filled.csv')
"""


# Quick comparison function
def CompareMLMethods(df, sample_frac=0.1):
    """
    Compare different ML imputation methods by hiding some known values

    Parameters:
    -----------
    df : pd.DataFrame
        Complete data (or mostly complete)
    sample_frac : float
        Fraction of data to hide for testing

    Returns:
    --------
    dict : Performance metrics for each method
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Create test set by hiding some values
    df_test = df.copy()
    mask = np.random.random(df.shape) < sample_frac
    mask[1000:1100] = True  # Ensure some known missing for testing
    mask[5000:5100] = True  # Ensure some known missing for testing
    mask[8000:8100] = True  # Ensure some known missing for testing
    actual_values = df_test.values[mask]
    df_test.values[mask] = np.nan

    methods = {
        'LinearRegression': lambda: imputeWithLinearRegression(df_test, max_iter=5),
        'BayesianRidge': lambda: ImputeWithBayesianRidge(df_test, max_iter=5),
        # 'MICE': lambda: ImputeSimpleMICE(df_test, max_iter=5),

        # 'KNN': lambda: ImputeWithKNN(df_test, n_neighbors=5),
        # 'KNN_Features': lambda: ImputeWithFeaturesKNN(df_test, n_neighbors=5),
        # 'KNN_FeaturesForwardLookup': lambda: ImputeWithFeaturesKNNForwardLookup(df_test, n_neighbors=5),

        # 'RandomForest': lambda: ImputeWithRandomForest(df_test, n_estimators=50, max_iter=5),
        # 'RandomForest_Features': lambda: ImputeWithFeaturesRF(df_test, n_estimators=50, max_iter=5),
        # 'RandomForest_FeaturesForwardLookup': lambda: ImputeWithRandomForest_FeaturesForwardLookup(df_test,n_estimators=50,max_iter=5),

        # 'lightGBM': lambda: ImputeWithLightGBM(df_test, max_iter=5),
        # 'lightGBM_Features': lambda: ImputeWithFeaturesLightGBM(df_test, max_iter=5),
        # 'ImputeWithFeaturesLightGBMForwardLookup': lambda: ImputeWithFeaturesLightGBMForwardLookup(df_test, max_iter=5),
    }

    results = {}
    for name, method in methods.items():
        print(f"Testing {name}...")
        df_imputed = method()
        predicted_values = df_imputed.values[mask]

        # Remove any remaining NaN pairs
        valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
        actual = actual_values[valid_mask]
        predicted = predicted_values[valid_mask]

        results[name] = {
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'R2': r2_score(actual, predicted)
        }

    return pd.DataFrame(results).T


if __name__ == '__main__':
    metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data_4_years()

    results = CompareMLMethods(region_series.iloc[:])
    # results = CompareMLMethods(region_series.iloc[:].head(24 * 365))

    print(results)

    # /home/asif/PycharmProjects/Air_Quality_Analysis/venv/bin/python /home/asif/PycharmProjects/Air_Quality_Analysis/data_preparation/imputer.py
    # Testing KNN...
    # Testing KNN_Features...
    # Testing RandomForest...
    # /home/asif/PycharmProjects/Air_Quality_Analysis/venv/lib/python3.6/site-packages/sklearn/impute/_iterative.py:686: ConvergenceWarning:
    #
    # [IterativeImputer] Early stopping criterion not reached.
    #
    #                    MAE      RMSE        R2
    # KNN           2.746919  5.185307  0.985563
    # KNN_Features  2.610771  5.829190  0.981755
    # RandomForest  1.772237  3.329518  0.994048
    #
    #                    MAE      RMSE        R2
    # KNN               2.558487  5.104098  0.988498
    # MICE              1.684560  3.681843  0.994015
    # RandomForest      1.731347  3.198550  0.995483
    # RF_Features       1.763326  3.142829  0.995614
    # lightGBM_Features 1.486084  2.90078   0.996404
    # LightGBMFL        1.412865  2.840891  0.996411

    #                                                MAE       RMSE        R2
    # MICE                                     40.386940  50.100410 -0.000038
    # KNN_Features                              9.064953  0.909815  14.697555
    # KNN_FeaturesForwardLookup                 8.656672  15.789790  0.900669
    # RandomForest_FeaturesForwardLookup        6.782514  13.124770  0.931369
    # ImputeWithFeaturesLightGBMForwardLookup   6.661351  12.873956  0.933967

    #                                  MAE       RMSE        R2
    # KNN                           35.316553  43.840976  -0.000388
    # KNN_Features                  11.180523  17.255057   0.845032
    # KNN_FeaturesFL                 7.837690  13.209590   0.909179
    # RandomForest_FeaturesFL        6.208170  10.505631   0.942208
    # ImputeWithFeaturesLightGBMFL   6.103854  10.263067   0.944846
    # From the results, Random Forest imputation performs best in terms of MAE, RMSE, and R2. because it can capture complex patterns in the data. on the other hand, KNN is faster but less accurate.

    #                                  MAE      RMSE        R2
    # LinearRegression              1.150766  2.534456  0.996510
    # BayesianRidge                 1.149585  2.532179  0.996516
    # KNN                           3.435856  6.312992  0.978306
    # KNN_Features                  2.932274  5.633413  0.982725
    # KNN_FeaturesFL                4.465343  7.312523  0.970893
    # RandomForest_FeaturesFL       2.176895  3.913272  0.991879
    # ImputeWithFeaturesLightGBMFL  1.800310  3.438361  0.993731

    # LinearRegression                         14.913600  23.302608  0.815281
    # BayesianRidge                            14.846514  23.246506  0.816170
    # KNN_FeaturesForwardLookup                17.281972  26.213015  0.766259
    # RandomForest_FeaturesForwardLookup       16.164294  25.510714  0.778616
    # ImputeWithFeaturesLightGBMForwardLookup  14.620064  22.839799  0.822546
