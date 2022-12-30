import math
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta, datetime

from meteorological_functions import MeteorologicalVariableType, vector_calculation
from paths import wunderground_data_path, wunderground_data_path_compressed

nc_file_path = '../Files/meteo data/wunderground/meteoData_BD_WB_NCT.nc'
regions = ['Dhaka', 'West Bengal', 'NCT']

linear_var = ['Temperature', 'Dew Point', 'Humidity', 'Wind Gust', 'Pressure', 'Precip.']
circular_var = ['Wind Direction', 'Wind Speed']
nominal_var = ['Condition']

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))

factor_list = ['Temperature']
name, unit, color_list = 'Temperature', 'Celsius', ['#FFA500']
temperature_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Humidity']
name, unit, color_list = 'Humidity', 'Fraction', ['#87CEEB']
humidity_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Pressure']
name, unit, color_list = 'Pressure', 'hPa', ['#2B65EC']
pressure_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Speed']
name, unit, color_list = 'Wind Speed', 'KM/H', ['#347C2C']
wind_speed_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Direction']
name, unit, color_list = 'Wind Direction', 'KM/H', ['#347C2C']
wind_direction_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)


def read_all_single_day_data_and_merge_usa(region="Dhaka"):
    # Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category
    region = region + '/'
    time_range = pd.date_range('2012-01-01', '2016-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    # time_series.to_csv(wunderground_data_path_compressed + region + '.csv')
    return time_series.set_index("Time")


def read_all_single_day_data_and_merge(region="Dhaka"):
    # Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category
    region = region + '/'
    time_range = pd.date_range('2019-01-01', '2021-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    # time_series.to_csv(wunderground_data_path_compressed + region + '.csv')
    return time_series.set_index("Time")


def fix_units_and_columns_compressed_data(time_series):
    time_series.Time = pd.to_datetime(time_series.Time)
    time_series['Wind Direction'] = time_series.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    time_series['Humidity'] = time_series['Humidity'] * 0.01
    time_series['Wind Speed'] = time_series['Wind Speed'] * 1.6
    time_series['Wind Gust'] = time_series['Wind Gust'] * 1.6
    time_series['Pressure'] = time_series['Pressure'] * 33.86
    time_series['Precip.'] = time_series['Precip.'] * 25.4
    time_series = time_series.drop('Wind', axis=1)
    return time_series.set_index("Time")


def read_compressed_data(region="Dhaka"):
    time_series = pd.read_csv(wunderground_data_path_compressed + region + '.csv').dropna(how='all')
    return fix_units_and_columns_compressed_data(time_series)


def eliminate_invalid_values(raw_data_copy):
    raw_data_copy.Temperature = raw_data_copy.Temperature.mask(raw_data_copy.Temperature < 0, np.nan).astype(float)
    raw_data_copy['Dew Point'] = raw_data_copy['Dew Point'].mask(raw_data_copy['Dew Point'] < 0, np.nan).astype(float)
    raw_data_copy.Pressure = raw_data_copy.Pressure.mask(raw_data_copy.Pressure == 0, np.nan).astype(float)
    raw_data_copy.Humidity = raw_data_copy.Humidity.mask(raw_data_copy.Humidity == 0, np.nan).astype(float)
    return raw_data_copy


def impute_by_previous_next_average_value(raw_data_copy):
    missing_data_linear_hourly = raw_data_copy[linear_var].resample("H").mean()
    clean_data_linear_var_hourly = pd.concat(
        [missing_data_linear_hourly[linear_var].ffill(), missing_data_linear_hourly[linear_var].bfill()]).groupby(
        level=0).mean()
    return clean_data_linear_var_hourly


def prepare_wind_vector_data(raw_data):
    circular_data_daily_samples = raw_data[circular_var].resample('H')
    wind_vector_list = [vector_calculation(x) for idx, x in circular_data_daily_samples]
    wind_vector_hourly = pd.DataFrame(wind_vector_list, columns=circular_var[::-1],
                                      index=circular_data_daily_samples.groups)
    return wind_vector_hourly


def clean_and_process_all_variable_data(raw_data):
    clean_data_with_missing = eliminate_invalid_values(raw_data)
    clean_data_linear_var_hourly = impute_by_previous_next_average_value(clean_data_with_missing)
    clean_data_circular_var_hourly = prepare_wind_vector_data(clean_data_with_missing)
    hourly_data_prepared = pd.concat((clean_data_linear_var_hourly, clean_data_circular_var_hourly), axis=1)
    return hourly_data_prepared


def prepare_xarray_dataset():
    time = pd.date_range(start='2019-01-01', end='2022-01-01', freq='H')[:-1]

    meteo_data_array = np.array(
        [clean_and_process_all_variable_data(read_compressed_data(region)) for region in regions])

    factor = ['Temperature', 'Dew Point', 'Humidity', 'Wind Gust', 'Pressure', 'Precip.', 'Wind Speed',
              'Wind Direction']

    meteo_data = xr.DataArray(data=meteo_data_array,
                              coords={"district": regions, "time": time, "factor": factor},
                              dims=["district", "time", "factor"], name='meteo')
    return meteo_data


def create_meteo_data_file(raw_data):
    meteo_data = clean_and_process_all_variable_data(raw_data)
    meteo_data.to_netcdf(nc_file_path)


def read_meteo_data_file(nc_file_path):
    return xr.open_dataset(nc_file_path)['meteo']
