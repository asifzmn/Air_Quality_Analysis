import math
import numpy as np

class MeteorologicalVariableType:
    def __init__(self, name, unit, factor_list, color_list):
        assert len(factor_list) == len(color_list)
        self.name = name
        self.unit = unit
        self.factor_list = factor_list
        self.color_list = color_list


def vector_calculation(x):
    if (x['Wind Speed'] == 0).all() or x.empty:
        return 0, 0
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
