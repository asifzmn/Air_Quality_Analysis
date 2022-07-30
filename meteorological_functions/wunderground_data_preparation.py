import math

import pandas as pd
import numpy as np

from paths import wunderground_data_path

regions = ['Dhaka', 'West Bengal', 'NCT']

linear_var = ['Temperature', 'Dew Point', 'Humidity', 'Wind Gust', 'Pressure', 'Precip.','Wind Speed']
circular_var = ['Wind Direction', 'Wind Speed']
nominal_var = ['Condition']

text = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angle = np.linspace(0, 360, 17)[:-1]
text_to_angle = dict(zip(text, angle))


# Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category

def vector_calculation(x):
    EW_Vector = (np.sin(np.radians(x['Wind Direction'])) * x['Wind Speed']).sum()
    NS_Vector = (np.cos(np.radians(x['Wind Direction'])) * x['Wind Speed']).sum()

    EW_Average = (EW_Vector / x.shape[0]) * 1
    NS_Average = (NS_Vector / x.shape[0]) * 1

    averageWindSpeed = np.sqrt(EW_Average * EW_Average + NS_Average * NS_Average)

    Atan2Direction = math.atan2(EW_Average, NS_Average)

    averageWindDirection = np.degrees(Atan2Direction)

    # //Correction As specified in webmet.com webpage http://www.webmet.com/met_monitoring/622.html
    # if(AvgDirectionInDeg > 180):
    #     AvgDirectionInDeg = AvgDirectionInDeg - 180
    # elif (AvgDirectionInDeg < 180):
    #     AvgDirectionInDeg = AvgDirectionInDeg + 180

    if averageWindDirection < 0:
        averageWindDirection += 360

    return averageWindSpeed, averageWindDirection


def prepare_data(region="Dhaka"):
    region = region + '/'
    # region = "NCT/"
    time_range = pd.date_range('2019-01-01', '2021-12-31')
    time_series = pd.concat([pd.read_csv(
        wunderground_data_path + region + str(singleDate.date())).dropna(how='all') for singleDate in time_range])
    time_series.Time = pd.to_datetime(time_series.Time)
    time_series['Wind Direction'] = time_series.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    time_series = time_series.drop('Wind',axis=1)
    return time_series.set_index("Time")
