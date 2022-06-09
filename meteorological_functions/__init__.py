# import os
# import time
import xarray as xr
import statsmodels.api as sm
import plotly.express as px
# from os import listdir
# from collections import Counter
# from os.path import isfile, join
# from datetime import datetime, timedelta
# from timeit import default_timer as timer
# from pandas_profiling import ProfileReport
import plotly.graph_objects as go
from data_preparation import *

from meteorological_data_preparation import factors, factor
from meteorological_data_preparation import prepare_and_save_meteo_data, meteorological_variable_type_list_linear
from meteorological_data_preparation import temperature_factor, humidity_factor, precipitation_factor
from meteorological_data_preparation import cloud_cover_factor, radiation_factor, pressure_factor
from meteorological_data_preparation import wind_speed_factor, wind_direction_factor

from meteorological_functions.wind_data_exploration import wind_direction_factors
from meteorological_functions.wind_data_exploration import *


def meteo_box_plot(meteo_data):
    factor = [['Surface Temperature', 'indianred'], ['Relative Humidity [2 m]', 'skyblue']][1]
    fig = go.Figure()
    for dis in meteo_data['district'].values: fig.add_trace(
        go.Box(y=meteo_data.loc[dis, :, factor[0]], name=dis, marker_color=factor[1]))
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

    for factor, factor_rename in zip(compare_factors, compare_factors_rename):
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
        # lag_series = pd.Series(data=[abs(all_data['PM2.5 Reading'].shift(30*i).corr(all_data[factor_rename]))
        # for i in lag_range],index=lag_range)
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


def profile_report():
    df = get_factor_data(meteo_data, factor)
    # prof = ProfileReport(df, minimal=False, title='Meteo Data')
    # prof.to_file(output_file='Meteo Data.html')


def pm_relation_model_ols():
    zonal_meteo_data = get_district_data(meteo_data, 'Dhaka').assign(Reading=time_series['Dhaka'])
    zonal_meteo_data = zonal_meteo_data.reset_index().drop('time', axis=1).dropna()

    model = sm.OLS(zonal_meteo_data['Reading'], zonal_meteo_data.drop('Reading', axis=1))
    results = model.fit()
    print(results.summary())


def read_meteo_data():
    return xr.open_dataset('../Files/meteoData_2019.nc')['meteo']


def meteo_data_basic_info():
    print(meteo_data)
    print(meteo_data.shape)
    print(meteo_data.dims)
    print(meteo_data.coords)
    print(meteo_data.to_dataframe())

    for dim in meteo_data.dims:
        print(meteo_data.coords[dim])


def chunk_data_write():
    for sample_factor in factors:
        get_factor_data(meteo_data, sample_factor)['2020'].to_csv(sample_factor + '.csv')


def meteo_time_series(meteo_data, meteorological_variable):
    fig = go.Figure()
    print(meteorological_variable.factor_list)
    meteo_data = meteo_data.loc[:, meteorological_variable.factor_list]
    # wind_dir_names = wind_direction_factors(wind_direction_factor)[0]
    wind_dir_names = []
    for i, factor in enumerate(np.setdiff1d(meteo_data['factor'].values, wind_dir_names)):
        fig.add_trace(
            go.Scatter(x=pd.Series(meteo_data['time'].values), y=meteo_data.loc[:, factor], name=factor,
                       marker_color=meteorological_variable.color_list[i]))  # line_color='deepskyblue','dimgray'

    fig.update_traces(mode='lines+markers', marker=dict(line_width=0, symbol='circle', size=3))
    fig.update_layout(title=meteorological_variable.name, font_size=16, legend_font_size=16, template="ggplot2",
                      xaxis_title="Time", yaxis_title=meteorological_variable.unit, legend_orientation='h')
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


if __name__ == '__main__':
    meta_data, time_series = get_metadata(), get_series()['2019']

    # prepare_and_save_meteo_data()
    meteo_data = read_meteo_data()
    # pm_vs_factor_scatter()
    # wind_graph_multi_zone(meteo_data)
    # exit()

    # print(time_series.corrwith(meteo_data).median())

    # factor = 'Temperature [2 m elevation corrected]'
    # print(meteoData.loc[:, '2019-07-02':'2019-07-05', ['Temperature [2 m]', 'Relative Humidity [2 m]']])
    # print(meteo_data.sel(factor=factor))

    # for factor in factors: print(get_factor_data(meteo_data, factor))

    # meteo_time_series(meteo_data.loc["Dhaka", :,:], wind_speed_factor)
    # meteo_time_series(meteo_data.loc["Dhaka", :,:], radiation_factor)

    for meteorological_variable in meteorological_variable_type_list_linear[:]:
        meteo_time_series(meteo_data.mean(axis=0).resample(time="M").mean(), meteorological_variable)

    # meteo_box_plot(meteo_data)
    # wind_graph_multi_zone(meteo_data,wind_direction_factor)

    # for meteoInfo in meteo_data[:1]:
    #     # dayGroup = meteoInfo.iloc[:72].groupby(meteoInfo.iloc[:72].index.day)
    #     dayGroup = meteoInfo.groupby(meteoInfo.index.day)
    #     for dayData in list(dayGroup)[::]: wind_graph_multi_zone(dayData[1],wind_direction_factor)

    exit()
