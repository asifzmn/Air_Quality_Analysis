import os
import time
from os import listdir
from data_preparation import *
from aq_analysis import *
from collections import Counter
from os.path import isfile, join
from datetime import datetime, timedelta
from timeit import default_timer as timer
from pandas_profiling import ProfileReport
from visualization_modules import SimpleTimeseries
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


class MeteorologicalVariableType:
    def __init__(self, name, unit, factor_list, color_list):
        assert len(factor_list) == len(color_list)
        self.name = name
        self.unit = unit
        self.factor_list = factor_list
        self.color_list = color_list


# factor_list = ['Temperature [2 m]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]', 'Temperature [700 mb]',
#                'Temperature [850 mb]', 'Temperature [1000 mb]']
# name, unit, color_list = 'Temperature', 'Celsius', ['#260637', '#843B58', '#B73239', '#FFA500', '#F9C53D', '#EADAA2']

# factor_list = ['Relative Humidity [2 m]','Soil Moisture [0-10 cm down]']
# name,unit,color_list = 'Humidity','Fraction',['mediumblue','lightblue']

# factor_list = ['Shortwave Radiation', 'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation']
# name, unit, color_list = 'Radiation', 'Flux', ['#260637', '#843B58', '#B73239']


factor_list = ['Wind Speed [10 m]', 'Wind Speed [100 m]', 'Wind Speed [500 mb]', 'Wind Speed [700 mb]',
               'Wind Speed [850 mb]', 'Wind Speed [900 mb]']
name, unit, color_list = 'Wind', 'KM/S', ['#260637', '#843B58', '#B73239', '#FFA500', '#F9C53D', '#EADAA2']

wind_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

# sampleFactors = ['Wind Gust', 'Wind Speed [10 m]', 'Wind Direction [10 m]', 'Temperature [2 m elevation corrected]',
#                  'Relative Humidity [2 m]']


def plotly_rose_plot(info, color_pal, all_districts):
    fig = go.Figure()
    # for [r,name] in info:fig.add_trace(go.Barpolar(r=r,name=name,marker_color=colorPal))
    for infod in info:
        for [r, name], colorp in zip(infod, color_pal):
            fig.add_trace(go.Barpolar(r=r, name=name, marker_color=colorp))

    district_count, wind_states = info.shape[0], info.shape[1]
    states = np.full((district_count, district_count * wind_states), False, dtype=bool)
    for i in range(district_count): states[i][wind_states * i:wind_states * (i + 1)] = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label=district, method="update",
                         args=[{"visible": state}, {"title": "Wind direction in " + district}])
                    for district, state in zip(all_districts, states)]),
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


def wind_direction_factors():
    wind_dir_names = np.array(wind_factor.factor_list)
    color_pal = np.array(['#ffffb1', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB', '#1E90FF'])
    # color_pal = np.array(['#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0'])
    return wind_dir_names, color_pal


def get_cardinal_direction(x, seg=16):
    direction = np.floor((x - (360 / seg) / 2) / (360 / seg)).astype('int')
    direction[direction == -1] = seg - 1
    return np.roll(np.hstack((np.bincount(direction), np.zeros(seg - max(direction) - 1))), 1)


def wind_graph_team(meteo_data):
    wind_dir_names, color_pal = wind_direction_factors()
    alldis = meteo_data['district'].values
    directions = np.array(
        # [[[getSides(meteoData.loc[dis, '2020-03-29':'2020-03-31', factor].values), factor]
        # for factor in wind_dir_names]
        [[[get_cardinal_direction(meteo_data.loc[dis, :, factor].values), factor] for factor in wind_dir_names]
         for dis in alldis])
    plotly_rose_plot(directions, color_pal, alldis)


# def WindGraph(x): PlotlyRosePlot([[getSides(x), 'Wind', '#3f65b1']])


def meteo_time_series(meteoData, factorname, unit, colors):
    fig = go.Figure()
    windDirNames = wind_direction_factors()[0]
    for i, factor in enumerate(np.setdiff1d(meteoData['factor'].values, windDirNames)): fig.add_trace(
        go.Scatter(x=pd.Series(meteoData['time'].values), y=meteoData.loc[:, factor], name=factor,
                   marker_color=colors[i]))  # line_color='deepskyblue','dimgray'
    fig.update_traces(mode='lines+markers', marker=dict(line_width=0, symbol='circle', size=3))
    fig.update_layout(title=factorname, font_size=16, legend_font_size=16, template="ggplot2",
                      xaxis_title="Time", yaxis_title=unit)
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


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


def get_factor_data(meteo_data, factor):
    return meteo_data.sel(factor=factor).to_dataframe().drop('factor', axis=1).unstack().T.droplevel(level=0)


def get_district_data(meteo_data, district):
    return meteo_data.sel(district=district).to_dataframe().drop('district', axis=1).unstack().T.droplevel(level=0).T


def pm_vs_factor_scatter():
    compare_factors = ['Temperature [2 m elevation corrected]', 'Relative Humidity [2 m]', 'Wind Speed [10 m]',
                       'Mean Sea Level Pressure [MSL]']
    compare_factors_rename = ['Temperature [Celcius]', 'Relative Humidity [Percentage]',
                              'Wind Speed [kilometer/second]', 'Mean Sea Level Pressure [hPa]']

    for factor, factor_rename in zip(compare_factors[:], compare_factors_rename):
        factor_meteo_data = get_factor_data(meteo_data, factor)
        if factor == 'Relative Humidity [2 m]':
            factor_meteo_data = factor_meteo_data * 100
        all_data = pd.concat((factor_meteo_data.stack(), time_series.stack(dropna=False)), axis=1).droplevel(1)
        all_data.columns = [factor_rename, "PM2.5 Reading"]
        all_data = all_data[all_data.index.month.isin([1, 2, 6, 7, 8, 12])]
        all_data['month'] = all_data.index.month
        all_data['season'] = all_data.apply(lambda x: 'summer' if x.month in (range(6, 9)) else 'winter', axis=1)

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

        # print(factor_meteo_data.stack().corr(time_series.stack(dropna=False)))
        fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename, color='seasonal_diurnal', opacity=0.45,
                         trendline='ols', color_discrete_map=color_map, template='seaborn')
        # fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename,opacity=0.33,trendline='ols',
        # color_discrete_map=color_map)
        fig.update_layout(height=750, width=750, font=dict(size=18), legend_orientation='h', yaxis_range=[0, 175])
        fig.show()


def one_folder(locationMain, date_folder):
    time = pd.to_datetime(date_folder.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [prepare_meteo_data(f'{locationMain}/{date_folder}/{district}.csv', district) for district in meta_data.index])
    return xr.DataArray(data=meteoData,
                        coords={"district": meta_data.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def one_folder_another(locationMain, folders):
    time = pd.to_datetime(folders.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [prepare_meteo_data(locationMain + '/' + folders + '/' + district + '.csv') for district in meta_data.index])

    return xr.DataArray(data=meteoData,
                        coords={"district": meta_data.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def prepare_meteo_data(file, district):
    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.columns = meteoInfo.columns.str.split(" ", len(district.split(' '))).str[-1]

    print(meteoInfo.sample(5).to_string())

    meteoInfo = meteoInfo[factors]
    # meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    meteoInfo['Relative Humidity [2 m]'] = meteoInfo['Relative Humidity [2 m]'] / 100

    return meteoInfo.values


def prepare_meteo_data2(file):
    def date_splitter(file_name):
        dates = file_name.split('/')[-2]
        time = pd.to_datetime(dates.split(' to '))
        return pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.columns, meteoInfo.index = factors, date_splitter(file)
    meteoInfo.columns.name, meteoInfo.index.name = 'Factors', 'Time'
    meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    return meteoInfo.stack()


def create_meteo_data_file():
    meteo_data = one_folder(meteoblue_data_path_2019, '2019-01-01 to 2019-12-31')
    meteo_data.to_netcdf('meteoData_2019.nc')


def prepare_meteo_data_time(file): return pd.to_datetime(pd.read_csv(file, sep=',', skiprows=9)['timestamp'])


def get_all_meteo_data_():
    locationMain = meteoblue_data_path + 'Meteoblue Scrapped Data'
    return xr.merge([one_folder(locationMain, date_folder) for date_folder in listdir(locationMain)[:]])
    # for date_folder in listdir(locationMain)[:3]:
    #     print(oneFolder(locationMain, date_folder))


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
    #     get_factor_data(meteo_data, sampleFactor)['2020'].to_csv(sampleFactor+'.csv')

    factor = factors[-3]
    df = get_factor_data(meteo_data, factor)
    # exit()

    # for factor in factors: print(get_factor_data(meteo_data, factor))
    # print(time_series.columns)

    zonal_meteo_data = get_district_data(meteo_data, 'Dhaka').assign(Reading=time_series['Dhaka'])
    zonal_meteo_data = zonal_meteo_data.reset_index().drop('time', axis=1).dropna()

    model = sm.OLS(zonal_meteo_data['Reading'], zonal_meteo_data.drop('Reading', axis=1))
    results = model.fit()
    print(results.summary())

    # print(time_series.corrwith(factor_meteo_data).median())
    # print(meteoData.to_dataframe())

    # prof = ProfileReport(df, minimal=False,title='Meteo Data')
    # prof.to_file(output_file='Meteo Data.html')

    # meteo_time_series(meteo_data.loc["Dhaka",:,selFactors],factorUnit[0],factorUnit[1],factorUnit[2])
    # meteo_box_plot(meteo_data)
    # wind_graph_team(meteo_data)

    # for meteoInfo in meteoData[:1]:
    #     # dayGroup = meteoInfo.iloc[:72].groupby(meteoInfo.iloc[:72].index.day)
    #     dayGroup = meteoInfo.groupby(meteoInfo.index.day)
    #     for dayData in list(dayGroup)[::]:WindGraphTeam(dayData[1],metaFrame['Zone'].values)

    exit()
