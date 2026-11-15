"""Fig 9 (lagged correlation, mobility vs deseasonalized PM2.5) and Fig 22 (deseasonalized series, low-mobility days).
Same pipeline as mobility/mobility_analysis.ipynb (cells 17, 29-34, 47, 49) with Berkeley Earth UTC timestamps
converted to local time (UTC+6) before daily averaging, so PM2.5 days match Google Mobility (local) days."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis"); sys.path.insert(0, HERE)
import plos_style as S
from data_preparation.additional_data_preparation import prepare_covid_dataset, get_mobility_data
from data_preparation.spatio_temporal_filtering import read_bd_data
from exploration import yearly_seasonal_decomposition_bd
from data_preparation.handling_missing_entries import FillMissingFromCombined

cs = read_bd_data()[5]
cs.index = cs.index + pd.Timedelta(hours=6)                          # UTC -> local
ts = cs.apply(FillMissingFromCombined).iloc[:, 0]["2017-03-01":"2022-02-28"].resample("D").mean()
ts = yearly_seasonal_decomposition_bd(ts); ts = (ts - ts.min()).iloc[:, 0]
_, start, end = prepare_covid_dataset()
mob_all = get_mobility_data(); mob = mob_all[start:end]

# ---- Fig 9: lagged correlations ----
corr = pd.DataFrame({i: mob.apply(lambda m: m.corr(ts.shift(-i))) for i in range(28)}).T
corr.index.name = "lag_days"; corr.to_csv(HERE + "/mobility_lag_corr_daily.csv")
print("best |r| lag per category:\n", corr.abs().idxmax().to_string(), "\nrange of r:", corr.values.min().round(3), corr.values.max().round(3))
# ---- Fig 22 data ----
low = ts.index.isin(mob_all.index[(mob_all.workplaces < 0) & (mob_all.transit_stations < 0)])
m_low, m_oth = ts[low].median(), ts[~low].median()
inmob = ts.index >= mob_all.index.min()
print("medians: low %.2f | all other %.2f (diff %.2f) | other within mobility period %.2f (diff %.2f)" %
      (m_low, m_oth, m_oth - m_low, ts[(~low) & inmob].median(), ts[(~low) & inmob].median() - m_low))
pd.DataFrame({"pm": ts, "low": low}).to_csv(HERE + "/desea_series.csv")

S.setup()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
# Fig 9
names = {"retail_and_recreation": "Retail & recreation", "grocery_and_pharmacy": "Grocery & pharmacy", "parks": "Parks",
         "transit_stations": "Transit stations", "workplaces": "Workplaces", "residential": "Residential"}
z = corr.T.rename(index=names); lim = 0.2
cmap = LinearSegmentedColormap.from_list("steel_rose", [S.DEEP, "#ffffff", S.SLATE])
fig, ax = plt.subplots(figsize=(S.FULL_W, 2.5))
im = ax.imshow(z.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
best = corr.abs().idxmax()
for r, c in enumerate(corr.columns):
    ax.add_patch(plt.Rectangle((best[c] - 0.5, r - 0.5), 1, 1, fill=False, ec=S.INK, lw=0.9))
ax.set_xticks(range(0, 28, 2)); ax.set_xticklabels(range(0, 28, 2))
ax.set_yticks(range(z.shape[0])); ax.set_yticklabels(z.index)
ax.set_xlabel("Lag (days): mobility on day t vs. deseasonalized PM$_{2.5}$ on day t + lag")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015); cb.set_label("Pearson r"); cb.outline.set_visible(False)
S.save(fig, "mobility_corr")
# Fig 22
fig, ax = plt.subplots(figsize=(S.FULL_W, 2.9))
ax.axhline(m_oth, color=S.GREYLINE, lw=1.0, zorder=2); ax.axhline(m_low, color=S.DEEP, lw=1.0, zorder=2)
ax.scatter(ts.index[~low], ts[~low], s=7, color=S.GREYBLUE, edgecolor="#8197ab", linewidth=0.25, zorder=3)
ax.scatter(ts.index[low], ts[low], s=9, color=S.DEEP, edgecolor="#174a7c", linewidth=0.3, zorder=4)
ax.set_xlim(ts.index.min() - pd.Timedelta(days=15), ts.index.max() + pd.Timedelta(days=15)); ax.set_ylim(-5, 190); ax.set_yticks(range(0, 161, 40))
ax.set_ylabel("Deseasonalized PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.5, color=S.DEEP, label="Low-mobility day"),
                   Line2D([], [], ls="", marker="o", ms=4.5, color=S.GREYBLUE, mec="#8197ab", label="Other days"),
                   Line2D([], [], color=S.DEEP, lw=1.2, label=f"Median, low-mobility days ({m_low:.1f})"),
                   Line2D([], [], color=S.GREYLINE, lw=1.2, label=f"Median, other days ({m_oth:.1f})")],
          frameon=False, loc="upper left", ncol=2)
S.ygrid(ax)
S.save(fig, "desea_time_series")
