import os
import time
from os import listdir
from collections import Counter
from os.path import isfile, join
from datetime import datetime, timedelta
from timeit import default_timer as timer
from pandas_profiling import ProfileReport
from visualization_modules import SimpleTimeseries
from data_preparation import *
import xarray as xr
import statsmodels.api as sm

# factors = ['Temperature [2 m]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure', 'Precipitation',
#            'Cloud Cover High', 'Cloud Cover Medium', 'Cloud Cover Low', 'Sunshine Duration', 'Shortwave Radiation',
#            'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
#            'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]',
#            'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
#            'Wind Direction [700 mb]', 'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]',
#            'Temperature [850 mb]', 'Temperature [700 mb]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]',
#            'Soil Moisture [0-10 cm down]']
#
# factors =['Temperature [2 m elevation corrected]', 'Relative Humidity [2 m]',
#        'Mean Sea Level Pressure [MSL]', 'Precipitation Total',
#        'Cloud Cover High [high cld lay]', 'Cloud Cover Medium [mid cld lay]',
#        'Cloud Cover Low [low cld lay]', 'Sunshine Duration',
#        'Shortwave Radiation', 'Direct Shortwave Radiation',
#        'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
#        'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]',
#        'Wind Speed [900 mb]', 'Wind Direction [900 mb]', 'Wind Speed [850 mb]',
#        'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
#        'Wind Direction [700 mb]', 'Wind Speed [500 mb]',
#        'Wind Direction [500 mb]', 'Temperature [1000 mb]',
#        'Temperature [850 mb]', 'Temperature [700 mb]', 'Temperature',
#        'Soil Temperature [0-10 cm down]', 'Soil Moisture [0-10 cm down]']

factors = ['Temperature [2 m elevation corrected]', 'Growing Degree Days [2 m elevation corrected]',
           'Temperature [900 mb]', 'Temperature [850 mb]', 'Temperature [800 mb]', 'Temperature [700 mb]',
           'Temperature [500 mb]', 'Precipitation Total', 'Precipitation Runoff', 'Relative Humidity [2 m]',
           'Wind Gust', 'Wind Speed [10 m]', 'Wind Direction [10 m]', 'Wind Speed [100 m]', 'Wind Direction [100 m]',
           'Wind Speed [900 mb]', 'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]',
           'Wind Speed [800 mb]', 'Wind Direction [800 mb]', 'Wind Speed [700 mb]', 'Wind Direction [700 mb]',
           'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Wind Speed [250 mb]', 'Wind Direction [250 mb]',
           'Cloud Cover Total', 'Cloud Cover High [high cld lay]', 'Cloud Cover Medium [mid cld lay]',
           'Cloud Cover Low [low cld lay]', 'Sunshine Duration', 'Shortwave Radiation', 'Longwave Radiation',
           'UV Radiation', 'Mean Sea Level Pressure [MSL]', 'Evapotranspiration']

selFactors = ['Temperature [2 m]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]', 'Temperature [700 mb]',
              'Temperature [850 mb]', 'Temperature [1000 mb]']
factorUnit = 'Temperature', 'Celsius', ['#260637', '#843B58', '#B73239', '#FFA500', '#F9C53D', '#EADAA2', ]

# selFactors = ['Relative Humidity [2 m]','Soil Moisture [0-10 cm down]']
# factorUnit = 'Humidity','Fraction',['mediumblue','lightblue']

radiation_factors = ['Shortwave Radiation', 'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation']
wind_speed = ['Wind Speed [10 m]', 'Wind Speed [80 m]', 'Wind Speed [500 mb]', 'Wind Speed [700 mb]',
              'Wind Speed [850 mb]',
              'Wind Speed [900 mb]']

sampleFactors = ['Wind Gust', 'Wind Speed [10 m]', 'Wind Direction [10 m]', 'Temperature [2 m elevation corrected]',
                 'Relative Humidity [2 m]']


def PlotlyRosePlot(info, colorPal, alldistricts):
    fig = go.Figure()
    # for [r,name] in info:fig.add_trace(go.Barpolar(r=r,name=name,marker_color=colorPal))
    for infod in info:
        for [r, name], colorp in zip(infod, colorPal):
            fig.add_trace(go.Barpolar(r=r, name=name, marker_color=colorp))

    districtCount, windStaes = info.shape[0], info.shape[1]
    states = np.full((districtCount, districtCount * windStaes), False, dtype=bool)
    for i in range(districtCount): states[i][windStaes * i:windStaes * (i + 1)] = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label=district, method="update",
                         args=[{"visible": state}, {"title": "Wind direction in " + district}])
                    for district, state in zip(alldistricts, states)]),
                active=0
            )
        ])

    fig.update_traces(
        text=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
    fig.update_layout(
        title='Wind', font_size=16, legend_font_size=16, polar_radialaxis_ticksuffix='', polar_angularaxis_rotation=90,
        template="plotly_dark", polar_angularaxis_direction="clockwise"
    )
    fig.show()


def WindGraphTeam(meteoData):
    windDirNames, colorPal = WindDirFactors()
    alldis = meteoData['districts'].values
    directions = np.array(
        # [[[getSides(meteoData.loc[dis, '2020-03-29':'2020-03-31', factor].values), factor] for factor in windDirNames]
        [[[get_cardinal_direction(meteoData.loc[dis, :, factor].values), factor] for factor in windDirNames]
         for dis in alldis])
    PlotlyRosePlot(directions, colorPal, alldis)


def WindDirFactors():
    windDirNames = np.array(
        ['Wind Direction [10 m]', 'Wind Direction [80 m]', 'Wind Direction [500 mb]', 'Wind Direction [700 mb]',
         'Wind Direction [850 mb]', 'Wind Direction [900 mb]'])
    # colorPal = np.array(['#ffffb1', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB', '#1E90FF'])
    colorPal = np.array(['#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0'])
    return windDirNames, colorPal


def get_cardinal_direction(x, seg=16):
    direction = np.floor((x - (360 / seg) / 2) / (360 / seg)).astype('int')
    direction[direction == -1] = seg - 1
    return np.roll(np.hstack((np.bincount(direction), np.zeros(seg - max(direction) - 1))), 1)


# def WindGraph(x): PlotlyRosePlot([[getSides(x), 'Wind', '#3f65b1']])

def PrepareMeteoData2(file):
    def dates(file):
        dates = file.split('/')[-2]
        time = pd.to_datetime(dates.split(' to '))
        return pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.columns, meteoInfo.index = factors, dates(file)
    meteoInfo.columns.name, meteoInfo.index.name = 'Factors', 'Time'
    meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    return meteoInfo.stack()


def PrepareMeteoDatatime(file): return pd.to_datetime(pd.read_csv(file, sep=',', skiprows=9)['timestamp'])


def MeteoTimeSeries(meteoData, factorname, unit, colors):
    fig = go.Figure()
    windDirNames = WindDirFactors()[0]
    for i, factor in enumerate(np.setdiff1d(meteoData['factor'].values, windDirNames)): fig.add_trace(
        go.Scatter(x=pd.Series(meteoData['time'].values), y=meteoData.loc[:, factor], name=factor,
                   marker_color=colors[i]))  # line_color='deepskyblue','dimgray'
    fig.update_traces(mode='lines+markers', marker=dict(line_width=0, symbol='circle', size=3))
    fig.update_layout(title=factorname, font_size=16, legend_font_size=16, template="ggplot2",
                      xaxis_title="Time", yaxis_title=unit)
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


def PrepareMeteoData(file, district):
    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.columns = meteoInfo.columns.str.split(" ", len(district.split(' '))).str[-1]

    print(meteoInfo.sample(5).to_string())

    meteoInfo = meteoInfo[factors]
    # meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    meteoInfo['Relative Humidity [2 m]'] = meteoInfo['Relative Humidity [2 m]'] / 100

    return meteoInfo.values


def oneFolder(locationMain, date_folder):
    time = pd.to_datetime(date_folder.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [PrepareMeteoData(f'{locationMain}/{date_folder}/{district}.csv', district) for district in meta_data.index])
    return xr.DataArray(data=meteoData,
                        coords={"district": meta_data.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def GetAllMeteoData():
    locationMain = meteoblue_data_path + 'Meteoblue Scrapped Data'
    return xr.merge([oneFolder(locationMain, date_folder) for date_folder in listdir(locationMain)[:]])
    # for date_folder in listdir(locationMain)[:3]:
    #     print(oneFolder(locationMain, date_folder))


def meteo_box_plot(meteoData):
    factor = [['Surface Temperature', 'indianred'], ['Relative Humidity [2 m]', 'skyblue']][1]
    fig = go.Figure()
    for dis in meta_data.index.values: fig.add_trace(
        go.Box(y=meteoData.loc[dis, :, factor[0]], name=dis, marker_color=factor[1]))
    fig.show()


def date_continuity(df):
    all = pd.Series(data=pd.date_range(start=df[0].min(), end=df[0].max(), freq='M'))
    mask = all.isin(df[0].values)
    print(all[~mask])


def oneFolderAnother(locationMain, folders):
    time = pd.to_datetime(folders.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [PrepareMeteoData(locationMain + '/' + folders + '/' + district + '.csv') for district in meta_data.index])

    return xr.DataArray(data=meteoData,
                        coords={"district": meta_data.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def get_factor_data(meteo_data, factor):
    return meteo_data.sel(factor=factor).to_dataframe().drop('factor', axis=1).unstack().T.droplevel(level=0)


def get_district_data(meteoData, district):
    return meteoData.sel(district=district).to_dataframe().drop('district', axis=1).unstack().T.droplevel(level=0).T


def pm_vs_factor_scatter():
    compare_factors = ['Temperature [2 m elevation corrected]', 'Relative Humidity [2 m]', 'Wind Speed [10 m]',
                       'Mean Sea Level Pressure [MSL]']
    compare_factors_rename = ['Temperature [Celcius]', 'Relative Humidity [Percentage]',
                              'Wind Speed [kilometer/second]', 'Mean Sea Level Pressure [hPa]']

    for factor, factor_rename in zip(compare_factors[:], compare_factors_rename):
        factor_meteo_data = get_factor_data(meteo_data, factor)
        if factor == 'Relative Humidity [2 m]': factor_meteo_data = factor_meteo_data * 100
        all_data = pd.concat((factor_meteo_data.stack(), time_series.stack(dropna=False)), axis=1).droplevel(1)
        all_data.columns = [factor_rename, "PM2.5 Reading"]
        all_data = all_data[all_data.index.month.isin([1, 2, 6, 7, 8, 12])]
        all_data['month'] = all_data.index.month
        all_data['season'] = all_data.apply(lambda x: 'summer' if x.month in (range(6, 9)) else 'winter', axis=1)

        # print(factor_rename)
        # # print((all_data[factor_rename].shift(30*1)))
        # lag_range = list(range(0,24))
        # lag_series = pd.Series(data=[abs(all_data['PM2.5 Reading'].shift(30*i).corr(all_data[factor_rename])) for i in lag_range],index=lag_range)
        # fig = px.scatter(lag_series)
        # fig.show()
        # print(lag_series)
        # print(lag_series.idxmax())
        # print()

        all_data = all_data.join(get_diurnal_period()).sample(1310 * 3)
        # all_data = all_data.join(get_diurnal_period()['2019']).sample(1310 * 3)
        all_data["seasonal_diurnal"] = all_data['season'] + ' ' + all_data['diurnal_name']
        # color_map = {'summer day': 'light red','summer night': 'dark red'}
        color_map = {'winter day': '#82CAFF', 'winter night': '#151B54', 'summer day': '#E67451',
                     'summer night': '#551606'}

        # print(factor_meteo_data)
        # print(factor_meteo_data.stack().corr(time_series.stack(dropna=False)))
        fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename, color='seasonal_diurnal', opacity=0.45,
                         trendline='ols', color_discrete_map=color_map, template='seaborn')
        # fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename,opacity=0.33,trendline='ols',color_discrete_map=color_map)
        fig.update_layout(height=750, width=750, font=dict(size=18), legend_orientation='h', yaxis_range=[0, 175])
        fig.show()


def prepare_meteo_data():
    meteoData = oneFolder(meteoblue_data_path_2019, '2019-01-01 to 2019-12-31')
    meteoData.to_netcdf('meteoData_2019.nc')


if __name__ == '__main__':
    meta_data, time_series = get_metadata(), get_series()['2019']

    # meteoData = GetAllMeteoData()
    # meteoData.to_netcdf('meteoData.nc')

    meteo_data = xr.open_dataset('Files/meteoData_2019.nc')['meteo']

    print(meteo_data)
    print(meteo_data.shape)
    print(meteo_data.dims)
    print(meteo_data.coords)
    for dim in meteo_data.dims:
        print(meteo_data.coords[dim])

    factor = 'Temperature [2 m elevation corrected]'
    # print(meteoData.loc[:, '2019-07-02':'2019-07-05', ['Temperature [2 m]', 'Relative Humidity [2 m]']])
    print(meteo_data.sel(factor=factor))

    # for sampleFactor in sampleFactors:
    #     getFactorData(meteoData, sampleFactor)['2020'].to_csv(sampleFactor+'.csv')

    factor = factors[-3]
    df = get_factor_data(meteo_data, factor)
    exit()

    # for factor in factors: print(get_factor_data(meteoData, factor))

    # print(time_series.columns)
    # zonal_meteo_data = get_district_data(meteoData, 'Dhaka').assign(Reading=time_series['Dhaka'])
    # zonal_meteo_data = zonal_meteo_data.reset_index().drop('time', axis=1).dropna()
    #
    # model = sm.OLS(zonal_meteo_data['Reading'], zonal_meteo_data.drop('Reading', axis=1))
    # results = model.fit()
    # print(results.summary())

    # print(time_series.corrwith(factor_meteo_data).median())
    # print(meteoData.to_dataframe())

    # prof = ProfileReport(df, minimal=False,title='Meteo Data')
    # prof.to_file(output_file='Meteo Data.html')

    # MeteoTimeSeries(meteoData.loc["Dhaka",:,selFactors],factorUnit[0],factorUnit[1],factorUnit[2])
    # MeteoBoxPlot(meteoData)
    # WindGraphTeam(meteoData)

    # for meteoInfo in meteoData[:1]:
    #     # dayGroup = meteoInfo.iloc[:72].groupby(meteoInfo.iloc[:72].index.day)
    #     dayGroup = meteoInfo.groupby(meteoInfo.index.day)
    #     for dayData in list(dayGroup)[::]:WindGraphTeam(dayData[1],metaFrame['Zone'].values)

    exit()
