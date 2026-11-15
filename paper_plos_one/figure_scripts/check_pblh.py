"""Reproduce PBLH-PM2.5 lag analysis (cdsapi_pblh.py logic): per-zone hourly Pearson r at lags 0-24 h."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
pm = read_bd_data_4_years()[1]
blh = pd.read_csv("/home/asif/Desktop/Work/Dataset/AQ Dataset/PBLH/bd_series_PBLH.csv", index_col=0, parse_dates=True)
blh.index = pm.index if len(blh) == len(pm) else blh.index
blh = blh.reindex(pm.index)[pm.columns]
print("PBLH coverage:", blh.notna().mean().mean().round(3), "| zones:", blh.shape[1])
r = pd.DataFrame({lag: [pm[z].corr(blh[z].shift(lag)) for z in pm.columns] for lag in range(25)}, index=pm.columns).T
print("median r across zones by lag:\n", r.median(axis=1).round(3).to_string())
best = r.abs().idxmax()
print("\nbest lag per zone: value counts\n", best.value_counts().sort_index().to_string())
for lag in (0, 3):
    v = r.loc[lag]; h, e = np.histogram(v, bins=np.arange(-0.6, 0.01, 0.05))
    print(f"lag {lag}: mean {v.mean():.3f}, median {v.median():.3f}, min {v.min():.3f}, max {v.max():.3f}, modal bin [{e[h.argmax()]:.2f},{e[h.argmax()+1]:.2f})")
r.to_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/pblh_lag_corr.csv")
