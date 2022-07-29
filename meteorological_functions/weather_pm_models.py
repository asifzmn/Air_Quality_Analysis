from sklearn.preprocessing import StandardScaler
import pandas as pd

from meteorological_functions import get_district_data
import statsmodels.api as sm


def pm_relation_model_ols(meteo_data, time_series):
    zonal_meteo_data = get_district_data(meteo_data, 'Dhaka').assign(Reading=time_series['Dhaka'])
    zonal_meteo_data = zonal_meteo_data.reset_index().drop('time', axis=1).dropna()

    model = sm.OLS(zonal_meteo_data['Reading'], zonal_meteo_data.drop('Reading', axis=1))
    results = model.fit()
    print(results.summary())


def ModelPreparation(timeSeries, reading):
    factors = ['Temperature', 'Humidity', 'Wind Speed']
    timeSeries = timeSeries[factors]
    timeSeries = (timeSeries.resample('H').mean())

    # print(timeSeries.isnull().any(axis=1))
    # print(timeSeries[timeSeries.isnull().any(axis=1)].index)
    # print(reading.loc[timeSeries[timeSeries.isnull().any(axis=1)].index].isnull().sum())
    # missing_data_heatmap(timeSeries)

    # allData = reading.join(timeSeries)
    allData = pd.concat((reading, timeSeries), axis=1)
    # PairDistributionSummary(allData)
    # print(allData.info())
    allData = allData.dropna()

    scaler = StandardScaler()
    scaler.fit(allData)
    timeSeries_model = pd.DataFrame(data=scaler.transform(allData), index=allData.index, columns=allData.columns)

    timeSeries_model = pd.concat([timeSeries_model.reset_index().drop('index', axis=1),
                                  pd.get_dummies(timeSeries_model.index.month.astype(str), prefix='month'),
                                  pd.get_dummies(timeSeries_model.index.hour.astype(str), prefix='hour')], axis=1)
    print(timeSeries_model.info())

    model = sm.OLS(timeSeries_model['Reading'], timeSeries_model.drop('Reading', axis=1))
    results = model.fit()
    print(results.summary())
