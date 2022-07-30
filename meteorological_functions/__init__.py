# import os
# import time
import xarray as xr
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from pandas_profiling import ProfileReport

# from os import listdir
# from collections import Counter
# from os.path import isfile, join
# from datetime import datetime, timedelta
# from timeit import default_timer as timer

from data_preparation import *

from meteoblue_data_preparation import factors, factor, get_factor_data
from meteoblue_data_preparation import prepare_multi_file_and_save_meteo_data, \
    meteorological_variable_type_list_linear
from meteoblue_data_preparation import temperature_factor, humidity_factor, precipitation_factor
from meteoblue_data_preparation import cloud_cover_factor, radiation_factor, pressure_factor
from meteoblue_data_preparation import wind_speed_factor, wind_direction_factor
from meteorological_functions.linear_data_visualization import meteo_time_series

from meteorological_functions.wind_data_exploration import wind_direction_factors
from meteorological_functions.wind_data_exploration import *


def date_continuity(df):
    all = pd.Series(data=pd.date_range(start=df[0].min(), end=df[0].max(), freq='M'))
    mask = all.isin(df[0].values)
    print(all[~mask])

def profile_report():
    df = get_factor_data(meteo_data, factor)
    prof = ProfileReport(df, minimal=False, title='Meteo Data')
    prof.to_file(output_file='Meteo Data.html')


def read_meteo_data():
    return xr.open_dataset('../Files/meteo data/meteoblue/meteoData_2019_BD_WB_NCT.nc')['meteo']


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
