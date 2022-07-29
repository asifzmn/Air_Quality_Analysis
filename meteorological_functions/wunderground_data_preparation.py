import pandas as pd
import numpy as np

from paths import wunderground_data_path

regions = ['Dhaka', 'West Bengal', 'NCT']

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))


# Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category


def prepare_data(region="Dhaka"):
    region = region + '/'
    # region = "NCT/"
    time_range = pd.date_range('2019-01-01', '2021-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    return time_series.set_index("Time")
