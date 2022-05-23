from sklearn.preprocessing import StandardScaler

from data_preparation import *
from aq_analysis import *
import statsmodels.api as sm

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))


# Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category

def ConditionStats(allData):
    condition_group = (allData.groupby('Condition').median()).loc[
        ['Haze', 'Fog', 'Rain', 'T-Storm', 'Drizzle', 'Thunder', 'Light Rain', 'Light Rain with Thunder',
         'Haze / Windy']]
    fig = go.Figure(data=[
        go.Bar(y=condition_group.Reading, x=condition_group.index, marker_color="grey")
    ])
    fig.show()

    direction_group = (allData.groupby('Wind').median())

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
    # MissingDataHeatmap(timeSeries)

    allData = reading.join(timeSeries)
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


def PrepareSeasonData():
    timeRange = pd.date_range('2017-01-01', '2019-12-31')
    timeSeries = pd.concat([pd.read_csv(
        aq_directory + 'Past Weather/' + str(singleDate.date())).dropna(how='all') for
                            singleDate in timeRange])
    timeSeries.Time = pd.to_datetime(timeSeries.Time)
    return timeSeries.set_index("Time")


def FactorAnalysis():
    reading = get_series()[['Tongi']]['2017-01-01': '2020-12-31']
    reading.columns = ['Reading']

    timeSeries = PrepareSeasonData()
    # ModelPreparation(timeSeries,reading)
    # return

    # print(timeSeries)
    # print(timeSeries.columns)

    # reading = reading.fillna(reading.median())
    # timeSeries = timeSeries.fillna(timeSeries.median())

    allData = reading.join(timeSeries)
    allData = allData.resample('D').mean()

    print(allData)
    # print(timeSeries.Condition.value_counts())
    # print(allData.corr())

    meteo_factors = ['Temperature', 'Humidity', 'Wind Speed']

    for f in meteo_factors:
        fig = px.scatter(allData, y="Reading", x=f)
        fig.update_layout(height=750, width=750, font=dict(size=21))
        fig.show()


def VectorAnalysis():
    timeSeries = PrepareSeasonData()
    print(timeSeries.sample(15).to_string())
    # factors = ['Temperature', 'Humidity', 'Wind Speed']
    # timeSeries = (timeSeries[factors].resample('H').mean())

    # BoxPlotSeason(timeSeries)
    # BoxPlotHour(timeSeries)
    timeSeries['wind_angle'] = timeSeries.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    print(timeSeries['wind_angle'].resample('M').agg(pd.Series.mode))
    # print(timeSeries['wind_angle'].resample('M').apply(lambda x: x.value_counts().iloc[:3]))
    # print(timeSeries.Wind.value_counts())
    # print(timeSeries.Wind.value_counts().sort_index())
    # print(timeSeries.Condition.value_counts())


if __name__ == '__main__':
    meta_data = get_metadata()
    # time_series = ReadPandasCSV()
