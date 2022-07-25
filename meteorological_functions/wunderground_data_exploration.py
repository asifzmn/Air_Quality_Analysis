from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from data_preparation import *
from visualization import missing_data_heatmap
import statsmodels.api as sm

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))


# Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category

def ConditionStats(all_data):
    condition_group = (all_data.groupby('Condition').median()).loc[
        ['Haze', 'Fog', 'Rain', 'T-Storm', 'Drizzle', 'Thunder', 'Light Rain', 'Light Rain with Thunder',
         'Haze / Windy']]
    fig = go.Figure(data=[
        go.Bar(y=condition_group.Reading, x=condition_group.index, marker_color="grey")
    ])
    fig.show()

    direction_group = (all_data.groupby('Wind').median())

    fig = go.Figure(data=[
        go.Bar(y=direction_group.Reading, x=direction_group.index, marker_color="grey")
    ])
    fig.show()


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


def FactorAnalysis(timeSeries, reading):
    # allData = reading.join(timeSeries)
    # allData = allData.resample('D').mean()
    allData = pd.concat((reading, timeSeries), axis=1)

    # print(allData.corr())

    meteo_factors = ['Temperature', 'Humidity', 'Wind Speed']

    for f in meteo_factors:
        fig = px.scatter(allData, y="Reading", x=f)
        fig.update_layout(height=750, width=750, font=dict(size=21))
        fig.show()

    # BoxPlotSeason(timeSeries)
    # BoxPlotHour(timeSeries)


def VectorAnalysis():
    timeSeries = prepare_data()
    print(timeSeries.sample(15).to_string())

    timeSeries['wind_angle'] = timeSeries.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    print(timeSeries['wind_angle'].resample('M').agg(pd.Series.mode))
    # print(timeSeries['wind_angle'].resample('M').apply(lambda x: x.value_counts().iloc[:3]))
    # print(timeSeries.Wind.value_counts())
    # print(timeSeries.Wind.value_counts().sort_index())


def prepare_data(region="Dhaka"):
    region = region + '/'
    # region = "NCT/"
    time_range = pd.date_range('2019-01-01', '2021-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    return time_series.set_index("Time")


if __name__ == '__main__':
    from data_preparation import get_metadata, get_series, clip_missing_prone_values, \
        prepare_division_and_country_series

    series_with_heavy_missing, metadata_with_heavy_missing = get_series(), get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_division_and_country_series(series,
                                                                                                           metadata)

    reading_data = region_series["2019":"2021"].Dhaka
    reading_data.name = "Reading"

    raw_data = prepare_data()

    # print(raw_data.Condition.value_counts())
    # ConditionStats(raw_data)
    ModelPreparation(raw_data, reading_data)
    # FactorAnalysis(raw_data, reading_data)
    # VectorAnalysis()
