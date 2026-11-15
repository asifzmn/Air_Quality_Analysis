import sys, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BLUE, LIGHT, PALE, INK = "#246cb4", "#9cc0d8", "#d6e4ef", "#222222"
OUT = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/out/"
sys.path.insert(0, "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"); import plos_style as PS; PS.setup()

def tidy(ax, grid="x"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e9e9e9"); ax.set_axisbelow(True)

# ---------- Fig 17: weekday distribution by season (as in exploration/EDA_time_series.ipynb, cell 46) ----------
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
country_series = read_bd_data_4_years()[5]
s = country_series["Bangladesh"].copy(); s.index = s.index + pd.Timedelta(hours=6)   # UTC -> local (UTC+6)
from scipy import stats
week_day = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
weekend = {"Friday", "Saturday"}
fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.4), sharex=True)
for ax, (season, months) in zip(axes, [("Summer (Jun–Aug)", [6, 7, 8]), ("Winter (Dec–Feb)", [12, 1, 2])]):
    ss = s[s.index.month.isin(months)]
    g = ss.groupby(ss.index.day_name())
    info = pd.concat([g.mean(), g.std()], axis=1).reindex(week_day)
    print(season, info.round(2).to_dict())
    dm = ss.resample("D").mean().dropna(); wk = dm[dm.index.day_name().isin(weekend)]; wd = dm[~dm.index.day_name().isin(weekend)]
    print(season, "hourly mean weekday %.2f weekend %.2f diff %.1f%%" % (ss[~ss.index.day_name().isin(weekend)].mean(), ss[ss.index.day_name().isin(weekend)].mean(),
          100 * (ss[ss.index.day_name().isin(weekend)].mean() / ss[~ss.index.day_name().isin(weekend)].mean() - 1)),
          "| daily-mean Welch t p = %.3f, Mann-Whitney p = %.3f" % (stats.ttest_ind(wd, wk, equal_var=False).pvalue, stats.mannwhitneyu(wd, wk, alternative="two-sided").pvalue))
    cols = [BLUE if d in weekend else LIGHT for d in week_day]
    ax.barh(range(7), info.iloc[:, 0], xerr=info.iloc[:, 1], color=cols, edgecolor="#1b3f66", linewidth=0.5,
            error_kw=dict(ecolor="#555", lw=0.9, capsize=2.5))
    ax.set_yticks(range(7)); ax.set_yticklabels(week_day); ax.invert_yaxis()
    ax.set_ylabel(season); tidy(ax)
axes[-1].set_xlabel("Mean PM$_{2.5}$ ($\\mu$g m$^{-3}$), error bars = 1 SD")
axes[0].legend(handles=[Patch(color=LIGHT, label="Weekday"), Patch(color=BLUE, label="Weekend (Fri–Sat)")],
               frameon=False, loc="lower right", ncol=2)
fig.tight_layout(); PS.save(fig, "week_distribution")
print("fig17 saved")

