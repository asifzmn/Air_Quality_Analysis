import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE, LIGHT, LIGHT_LINE = "#008bfb", "#b9c8d6", "#7f97ad"
d = pd.read_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/desea_series.csv", index_col=0, parse_dates=True)
low, ts = d["low"].astype(bool), d["pm"]
m_low, m_oth = ts[low].median(), ts[~low].median()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=300)
ax.scatter(ts.index[~low], ts[~low], s=11, color=LIGHT, edgecolor="#8197ab", linewidth=0.25, zorder=3)
ax.scatter(ts.index[low], ts[low], s=13, color=BLUE, edgecolor="#005a9e", linewidth=0.3, zorder=4)
x0, x1 = ts.index.min(), ts.index.max()
ax.axhline(m_oth, color=LIGHT_LINE, lw=1.1, zorder=2)
ax.axhline(m_low, color=BLUE, lw=1.1, zorder=2)
ax.set_xlim(x0 - pd.Timedelta(days=20), x1 + pd.Timedelta(days=20))
ax.set_ylabel("Deseasonalized PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=BLUE, label="Low-mobility day"),
                   Line2D([], [], ls="", marker="o", ms=6, color=LIGHT, label="Other days"),
                   Line2D([], [], color=BLUE, lw=1.4, label=f"Median, low-mobility days ({m_low:.1f})"),
                   Line2D([], [], color=LIGHT_LINE, lw=1.4, label=f"Median, other days ({m_oth:.1f})")],
          frameon=False, loc="upper left", ncol=4, fontsize=9.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#999"); ax.spines["bottom"].set_color("#999")
ax.tick_params(colors="#444")
ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True)
ax.set_ylim(-5, 160)
fig.tight_layout(); fig.savefig("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/out/desea_time_series.png", dpi=300)
print("saved")
