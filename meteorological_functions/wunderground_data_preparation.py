import math
import pandas as pd
import numpy as np

from meteorological_functions import MeteorologicalVariableType
from paths import wunderground_data_path

regions = ['Dhaka', 'West Bengal', 'NCT']

linear_var = ['Temperature', 'Humidity', 'Wind Gust', 'Pressure', 'Precip.', 'Wind Speed']
circular_var = ['Wind Direction', 'Wind Speed']
nominal_var = ['Condition']

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))

factor_list = ['Temperature']
name, unit, color_list = 'Temperature', 'Celsius', ['#260637']
temperature_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Humidity']
name, unit, color_list = 'Humidity', 'Fraction', ['#87CEEB']
humidity_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Pressure']
name, unit, color_list = 'Pressure', 'hPa', ['#2B65EC']
Pressure_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Speed']
name, unit, color_list = 'Wind Speed', 'KM/H', ['#347C2C']
wind_speed_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)

factor_list = ['Wind Direction']
name, unit, color_list = 'Wind Direction', 'KM/H', ['#347C2C']
wind_direction_factor = MeteorologicalVariableType(name, unit, factor_list, color_list)


def prepare_data(region="Dhaka"):
    # Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category
    region = region + '/'
    # region = "NCT/"
    time_range = pd.date_range('2019-01-01', '2021-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    time_series['Wind Direction'] = time_series.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    time_series['Wind Speed'] = time_series['Wind Speed'] * 1.6
    time_series['Wind Gust'] = time_series['Wind Gust'] * 1.6
    time_series['Pressure'] = time_series['Pressure'] * 33.86
    time_series['Precip.'] = time_series['Precip.'] * 25.4
    time_series = time_series.drop('Wind', axis=1)
    return time_series.set_index("Time")
