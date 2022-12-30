import pandas as pd
import numpy as np


def FillMissingDataFromHours(x, hours=1):
    ss = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromYears(y):
    ss = [y.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    return y.fillna((pd.concat(ss, axis=1).mean(axis=1)))

# df = df.fillna(df.rolling(6, min_periods=1, ).mean()).round(3)
# data.resample('S', fill_method='pad')  # forming a series of seconds
