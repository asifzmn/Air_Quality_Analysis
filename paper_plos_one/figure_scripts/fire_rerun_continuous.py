# Usage: python fire_rerun_continuous.py fire_rerun_continuous.csv  (final review, 2026-10-28; replaces the fire-date-only index of meteorology/FIRE/grouped_bar.py)
# Corrected fire-PM2.5 rerun: continuous daily index, FRP=0 on fire-free days, all zones, true day lags.
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/asif/PycharmProjects/Air_Quality_Analysis')
from scipy.stats import pearsonr
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
from paths import aq_directory
fire = pd.read_csv(aq_directory + '/NASA FIRMS DATASET/DL_FIRE_J1V-C2_668887/fire_archive_J1V-C2_668887.csv', parse_dates=['acq_date'])
meta, series, *_ = read_bd_data_4_years()
print('series index', series.index.min(), series.index.max(), 'tz', series.index.tz, 'zones', series.shape[1])
pm = series.resample('D').mean()
days = pm.index
fire = fire[(fire.acq_date >= days.min()) & (fire.acq_date <= days.max())]
R = 6371.0
la1, lo1 = np.radians(meta['Latitude'].values)[:, None], np.radians(meta['Longitude'].values)[:, None]
la2, lo2 = np.radians(fire.latitude.values)[None, :], np.radians(fire.longitude.values)[None, :]
d = 2*R*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
fd = fire.acq_date.dt.normalize().values
anom = lambda s: s - s.rolling(31, center=True, min_periods=15).mean()
rows = []
for rad in [10, 25, 50, 75, 100, 150, 200]:
    for zi, z in enumerate(meta.index):
        m = d[zi] <= rad
        frp = pd.Series(fire.frp.values[m], index=fd[m]).groupby(level=0).sum().reindex(days, fill_value=0.0)
        p = pm[z]
        for lag in range(6):
            f = frp.shift(lag)
            for kind, a, b in [('raw', f, p), ('deseas', anom(f), anom(p))]:
                x = pd.concat([a, b], axis=1).dropna()
                if len(x) > 30 and x.iloc[:, 0].std() > 0:
                    r, pv = pearsonr(x.iloc[:, 0], x.iloc[:, 1])
                    rows.append((rad, z, lag, kind, r, pv, len(x), (frp > 0).sum()))
df = pd.DataFrame(rows, columns=['rad', 'zone', 'lag', 'kind', 'r', 'p', 'n', 'firedays'])
df.to_csv(sys.argv[1], index=False)
for kind in ['raw', 'deseas']:
    s = df[df.kind == kind]
    print('\n==', kind, 'mean r (rows radius, cols lag)')
    print(s.pivot_table(index='rad', columns='lag', values='r', aggfunc='mean').round(3))
    print('zones, max zone r, n zones r>0 & p<.05 at lag1:')
    print(s[s.lag == 1].groupby('rad').agg(zones=('zone', 'nunique'), maxr=('r', 'max'), sigpos=('r', lambda v: ((v > 0) & (s.loc[v.index, 'p'] < .05)).sum()), firedays=('firedays', 'median')))
