"""Fig 9: daily mobility vs. deseasonalized daily PM2.5 at lags of 0-27 days (same pipeline as mobility/mobility_analysis.ipynb, cells 17, 47, 49)."""
import os, sys, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from data_preparation.additional_data_preparation import prepare_covid_dataset, get_mobility_data
from data_preparation.spatio_temporal_filtering import read_bd_data
from exploration import yearly_seasonal_decomposition_bd
from data_preparation.handling_missing_entries import FillMissingFromCombined

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
csv = os.path.join(HERE, "mobility_lag_corr_daily.csv")
if not os.path.exists(csv):
    country_series = read_bd_data()[5]
    _, start, end = prepare_covid_dataset()
    mobility = get_mobility_data()[start:end]
    ts = country_series.apply(FillMissingFromCombined).iloc[:, 0]["2017-03-01":"2022-02-28"].resample("D").mean()
    ts = yearly_seasonal_decomposition_bd(ts); ts = (ts - ts.min()).iloc[:, 0]
    corr = pd.DataFrame({i: mobility.apply(lambda m: m.corr(ts.shift(-i))) for i in range(28)}).T
    corr.index.name = "lag_days"; corr.to_csv(csv)
corr = pd.read_csv(csv, index_col=0)
names = {"retail_and_recreation": "Retail & Recreation", "grocery_and_pharmacy": "Grocery & Pharmacy", "parks": "Parks",
         "transit_stations": "Transit Stations", "workplaces": "Workplaces", "residential": "Residential"}
z = corr.T.rename(index=names); lim = float(np.ceil(np.abs(z.values).max() * 10) / 10)
cmap = LinearSegmentedColormap.from_list("steel_rose", ["#246cb4", "#ffffff", "#b5506d"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(13, 3.8), dpi=300)
im = ax.imshow(z.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for r in range(z.shape[0]):
    for c in range(z.shape[1]):
        ax.text(c, r, f"{z.values[r, c]:.2f}", ha="center", va="center", fontsize=6.5, color="#222")
ax.set_xticks(range(z.shape[1])); ax.set_xticklabels(z.columns); ax.set_yticks(range(z.shape[0])); ax.set_yticklabels(z.index)
ax.set_xlabel("Lag (days): mobility on day t vs. deseasonalized PM$_{2.5}$ on day t + lag"); ax.set_ylabel("Mobility category")
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Pearson r")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mobility_corr.png"), dpi=300); print("fig9 saved")
