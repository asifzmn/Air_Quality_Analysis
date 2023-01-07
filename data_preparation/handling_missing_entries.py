import itertools

import pandas as pd
import numpy as np


def FillMissingDataFromHours(x, hours=1):
    ss = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromYears(y, years=2):
    ss = [y.shift(shft, freq='YS') for shft in np.delete(np.arange(-years, years + 1), years)]
    # ss = [y.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    return y.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingFromCombined(x, hours=1, days=7, years=2):
    s = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    sss = [x.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    s_all = list(itertools.chain(s, ss, sss))
    return x.fillna((pd.concat(s_all, axis=1)).mean(axis=1))

def FillMissingFromNeighbors(series):
    return series.T.fillna(series.mean(axis=1)).T

# df = df.fillna(df.rolling(6, min_periods=1, ).mean()).round(3)
# data.resample('S', fill_method='pad')  # forming a series of seconds
