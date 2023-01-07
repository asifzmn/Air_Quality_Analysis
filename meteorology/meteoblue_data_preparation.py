import pandas as pd
import numpy as np
import xarray as xr
from os import listdir
from datetime import timedelta
from data_preparation import get_metadata
from meteorology import MeteorologicalVariableType
from paths import meteoblue_data_path, meteoblue_data_path_2019
from scrapers.meteoblue import selcet_meteoblue_zones

meta_data = get_metadata()

# factors = ['Temperature [2 m]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure', 'Precipitation',
#            'Cloud Cover High', 'Cloud Cover Medium', 'Cloud Cover Low', 'Sunshine Duration', 'Shortwave Radiation',
#            'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
#            'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]',
#            'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
#            'Wind Direction [700 mb]', 'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]',
#            'Temperature [850 mb]', 'Temperature [700 mb]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]',
#            'Soil Moisture [0-10 cm down]']
#
# factors = ['Temperature [2 m elevation corrected]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure [MSL]',
#            'Precipitation Total', 'Cloud Cover High [high cld lay]', 'Cloud Cover Medium [mid cld lay]',
#            'Cloud Cover Low [low cld lay]', 'Sunshine Duration', 'Shortwave Radiation', 'Direct Shortwave Radiation',
#            'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]', 'Wind Direction [10 m]',
#            'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]', 'Wind Direction [900 mb]',
#            'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]', 'Wind Direction [700 mb]',
#            'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]', 'Temperature [850 mb]',
#            'Temperature [700 mb]', 'Temperature', 'Soil Temperature [0-10 cm down]', 'Soil Moisture [0-10 cm down]']

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

factor = factors[-3]

factor_list = ['Temperature [2 m elevation corrected]', 'Temperature [900 mb]', 'Temperature [850 mb]',
               'Temperature [800 mb]', 'Temperature [700 mb]', 'Temperature [500 mb]']
name, unit, color_list = 'Temperature', 'Celsius', ['#260637', '#843B58', '#B73239', '#FFA500', '#F9C53D', '#EADAA2']
temperature_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Relative Humidity [2 m]']
name, unit, color_list = 'Humidity', 'Fraction', ['#87CEEB']
humidity_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Precipitation Total', 'Precipitation Runoff']
name, unit, color_list = 'Precipitation', 'mm', ['mediumblue', 'lightblue']
precipitation_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Cloud Cover Total', 'Cloud Cover High [high cld lay]', 'Cloud Cover Medium [mid cld lay]',
               'Cloud Cover Low [low cld lay]', ]
name, unit, color_list = 'Cloud Cover', 'Percentage', ['#6D6968', '#D3D3D3', '#B6B6B4', '#736F6E']
cloud_cover_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Shortwave Radiation', 'Longwave Radiation', 'UV Radiation']
name, unit, color_list = 'Radiation', 'Flux', ['#FFE87C', '#B73239', '#8D38C9']
radiation_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Mean Sea Level Pressure [MSL]']
name, unit, color_list = 'MSL Pressure', 'hPa', ['#2B65EC']
pressure_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Speed [10 m]', 'Wind Speed [100 m]', 'Wind Speed [250 mb]', 'Wind Speed [500 mb]',
               'Wind Speed [700 mb]', 'Wind Speed [850 mb]', 'Wind Speed [900 mb]']
name, unit, color_list = 'Wind Speed', 'KM/H', ['#347C2C', '#254117', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB',
                                                '#1E90FF']
wind_speed_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Direction [10 m]', 'Wind Direction [100 m]', 'Wind Direction [250 mb]', 'Wind Direction [500 mb]',
               'Wind Direction [700 mb]', 'Wind Direction [850 mb]', 'Wind Direction [900 mb]']
name, unit, color_list = 'Wind Direction', 'Degree', ['#347C2C', '#254117', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB',
                                                      '#1E90FF']
wind_direction_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

meteorological_variable_type_list_linear = [temperature_factor, humidity_factor, precipitation_factor,
                                            cloud_cover_factor, radiation_factor, pressure_factor, wind_speed_factor]

meteorological_variable_type_list_circular = [wind_direction_factor, wind_speed_factor]


# sampleFactors = ['Wind Gust', 'Wind Speed [10 m]', 'Wind Direction [10 m]', 'Temperature [2 m elevation corrected]',
#                  'Relative Humidity [2 m]']


def prepare_multi_file_and_save_meteo_data():
    location_main = meteoblue_data_path + 'Meteoblue Scrapped Data'
    meteo_data = xr.merge([process_one_folder(location_main, date_folder) for date_folder in listdir(location_main)[:]])
    # for date_folder in listdir(location_main)[:3]:
    #     print(oneFolder(location_main, date_folder))
    meteo_data.to_netcdf('meteo_data.nc')


def prepare_meteo_data_time(file): return pd.to_datetime(pd.read_csv(file, sep=',', skiprows=9)['timestamp'])


def prepare_meteo_data_another(file):
    def date_splitter(file_name):
        dates = file_name.split('/')[-2]
        time = pd.to_datetime(dates.split(' to '))
        return pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteo_info = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteo_info.drop(['timestamp'], axis=1, inplace=True)
    meteo_info = meteo_info.apply(pd.to_numeric)
    meteo_info.columns, meteo_info.index = factors, date_splitter(file)
    meteo_info.columns.name, meteo_info.index.name = 'Factors', 'Time'
    meteo_info.iloc[:, 1] = meteo_info.iloc[:, 1] / 100
    return meteo_info.stack()


def prepare_meteo_data(file, district):
    meteo_info = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteo_info.drop(['timestamp'], axis=1, inplace=True)
    meteo_info = meteo_info.apply(pd.to_numeric)
    meteo_info.columns = meteo_info.columns.str.split(" ", len(district.split(' '))).str[-1]

    print(district)
    if district == 'Bhola':
        meteo_info.columns = meteo_info.columns.str.replace('District ', '')
    if district == 'Kalyani':
        meteo_info.columns = meteo_info.columns.str.replace('22.98°N ', '')
        meteo_info.columns = meteo_info.columns.str.replace('88.48°E ', '')
    if district == 'Rangpur':
        meteo_info.columns = meteo_info.columns.str.replace('City ', '')

    # print(meteo_info.sample(5).to_string())
    # print(meteo_info.columns)

    meteo_info = meteo_info[factors]
    # meteo_info.iloc[:, 1] = meteo_info.iloc[:, 1] / 100
    meteo_info['Relative Humidity [2 m]'] = meteo_info['Relative Humidity [2 m]'] / 100

    return meteo_info.values


def process_one_folder(location_main, date_folder):
    from data_preparation.spatio_temporal_filtering import read_bd_meta_data

    time = pd.to_datetime(date_folder.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    # meta_data = get_metadata()
    # meta_data = selcet_meteoblue_zones(meta_data)
    meta_data = read_bd_meta_data()

    meteo_data = np.array(
        [prepare_meteo_data(f'{location_main}/{date_folder}/{district}.csv', district) for district in
         meta_data.Zone[:]])
    return xr.DataArray(data=meteo_data,
                        coords={"district": meta_data.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def create_meteo_data_file():
    meteo_data = process_one_folder(meteoblue_data_path_2019, '2019-01-01 to 2019-12-31')
    meteo_data.to_netcdf('meteoData_2019.nc')


def read_meteo_data():
    return xr.open_dataset('../Files/meteo data/meteoblue/meteoData_2019.nc')['meteo']


def read_meteo_data_file_bd_and_neighbours():
    return xr.open_dataset('meteoblue_data_2019_bd_and_neighbours.nc')['meteo']


def create_meteo_data_file_bd_and_neighbours():
    from paths import aq_directory
    meteoblue_data_path_2019_bd_and_neighbours = aq_directory + 'Meteoblue Data/MeteoBlue Data 2019 BD-WB_NCT/BD'
    meteo_data = process_one_folder(meteoblue_data_path_2019_bd_and_neighbours, '2019-01-01 to 2019-12-31')
    meteo_data.to_netcdf('meteoblue_data_2019_bd_and_neighbours.nc')


if __name__ == '__main__':
    # print(len(factors))
    # create_meteo_data_file_bd_and_neighbours()
    print(read_meteo_data_file_bd_and_neighbours())
