"""Fig 3 (division PM2.5 distributions), Fig 8 (demographics vs PM2.5), Fig 25 (cross-correlation statistics).
Data pipelines follow exploration/demographic_analysis.ipynb and cross_correlation/leader_follower_zone.ipynb."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
PROJ = "/home/asif/PycharmProjects/Air_Quality_Analysis"; sys.path.insert(0, PROJ)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
STEEL, LIGHT, GREY, EDGE, ROSE = "#3f78ad", "#9cc0d8", "#b3c3d1", "#2c4a66", "#b5506d"
sys.path.insert(0, "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"); import plos_style as PS; PS.setup()
def tidy(ax, grid="x"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e9e9e9"); ax.set_axisbelow(True)
which = sys.argv[1:] or ["3", "8", "25"]

if "3" in which:
    from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
    region_series = read_bd_data_4_years()[3]
    order = region_series.median().sort_values().index.tolist()
    data = [region_series[d].dropna().values for d in order]
    bands = [(0, 12, "Good", "#cfe8cf"), (12, 35.4, "Moderate", "#f6f0c4"), (35.4, 55.4, "Sensitive", "#f8dfc2"),
             (55.4, 150.4, "Unhealthy", "#f3cccc"), (150.4, 250.4, "Very unhealthy", "#e3d1e8")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for lo, hi, name, col in bands:
        ax.axvspan(lo, hi, color=col, alpha=0.45, lw=0, zorder=0)
        ax.text((lo + hi) / 2, len(order) + 0.75, name, ha="center", va="bottom", fontsize=7, color="#555")
    national = float(read_bd_data_4_years()[1].stack().median())  # all zone-hours pooled, as in EDA_time_series.ipynb cell 7 (44.2)
    ax.axvline(national, color="#6b7c8c", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(national, len(order) + 0.45, f"National median {national:.1f}", fontsize=8, color="#4d5c6a", va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, widths=0.5, patch_artist=True,
                    medianprops=dict(color=EDGE, lw=2.2), whiskerprops=dict(color="#7d8fa0", lw=1.1),
                    capprops=dict(color="#7d8fa0", lw=0), boxprops=dict(facecolor=LIGHT, edgecolor=EDGE, lw=0.7))
    for i, d in enumerate(data):
        q1, q3 = np.percentile(d, [25, 75])
        ax.text(q3 + 2, i + 1, f"{np.median(d):.1f}", ha="left", va="center", fontsize=7.5, color=EDGE, fontweight="bold")
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order)
    ax.set_xlim(0, 250); ax.set_ylim(0.4, len(order) + 1.15)
    ax.set_xlabel("Hourly PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for lo, hi, name, col in bands[1:]:
        ax.axvline(lo, color="white", lw=1.2, zorder=0.5)
    fig.tight_layout(); PS.save(fig, "box_plot_avg")
    print("fig3 saved | medians:", {d: round(float(region_series[d].median()), 2) for d in order})

if "8" in which:
    os.chdir(os.path.join(PROJ, "exploration"))
    nb = json.load(open("demographic_analysis.ipynb"))
    g = {}
    for i in [0, 6, 10, 12, 14, 23, 24, 25, 27, 29, 34, 37, 38, 39, 41, 42, 44, 46, 54]:
        code = "".join(nb["cells"][i]["source"])
        code = "\n".join(l for l in code.split("\n") if "print(" not in l and "to_latex" not in l)
        exec(code, g)
    X, y = g["road_and_household_data_district"], g["average_reading_district"]
    drop = ["Dhaka_Dhaka", "Chittagong_Chittagong"]  # metropolitan outliers excluded in the reported analysis
    X, y = X.drop(drop), y.drop(drop)
    from matplotlib.ticker import FuncFormatter
    kfmt = FuncFormatter(lambda v, _: f"{v/1e6:g}M" if abs(v) >= 1e6 else (f"{v/1e3:g}k" if abs(v) >= 1e3 else f"{v:g}"))
    cols = ['Total Highways', 'Per Capita Income', 'Households Total', 'Households General', 'Households Institutional',
            'Households Others', 'Population', 'Male', 'Female']
    units = ['km', 'BDT', 'count', 'count', 'count', 'count', 'count', 'count', 'count']
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.6))
    for ax, c, u in zip(axes.flat, cols, units):
        x = X[c].astype(float); r = x.corr(y); res = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, res.intercept + res.slope * xs, color=STEEL, lw=1.6, zorder=2)
        ax.scatter(x, y, s=26, color=LIGHT, edgecolor=EDGE, linewidth=0.5, zorder=3)
        ax.set_title(c, fontsize=10.5); ax.set_xlabel(f"{c} ({u})", fontsize=9)
        ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY))
        ax.xaxis.set_major_formatter(kfmt); ax.tick_params(labelsize=8.5)
        tidy(ax, "both")
    for ax in axes[:, 0]: ax.set_ylabel("Average PM$_{2.5}$ ($\\mu$g m$^{-3}$)", fontsize=9)
    fig.tight_layout(); PS.save(fig, "demographic_scatterplot")
    print("fig8 saved | n =", len(y), "| r:", {c: round(X[c].astype(float).corr(y), 2) for c in cols})

if "25" in which:
    ev = pd.read_csv(os.path.join(PROJ, "cross_correlation/evidence_df.csv"))
    det = ev[ev.transport_detected]
    def stats_for(key):
        s = pd.concat((det.groupby(key)[["max_correlation", "max_correlation_lag"]].mean(),
                       det[key].value_counts() / ev[key].value_counts() * 100), axis=1)
        s.columns = ["corr", "lag", "pct"]; return s.sort_values("pct").round(2)
    bd, ind = stats_for("bangladesh_zone"), stats_for("india_zone")
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2), gridspec_kw={"height_ratios": [len(bd), len(ind)]})
    specs = [("corr", "Mean lagged r", (0, 0.75)), ("lag", "Mean best lag (h)", (0, 5.6)),
             ("pct", "Windows flagged (%)", (0, 8.6))]
    for row, (tab, label, col) in enumerate([(bd, "Bangladeshi divisions (followers)", STEEL), (ind, "Indian states (leaders)", LIGHT)]):
        for k, (m, xl, lim) in enumerate(specs):
            ax = axes[row, k]
            ax.barh(tab.index, tab[m], color=col, edgecolor=EDGE, linewidth=0.5, height=0.7)
            for yi, v in enumerate(tab[m]):
                ax.text(v + lim[1] * 0.01, yi, f"{v:.2f}", va="center", fontsize=6.5)
            ax.set_xlim(*lim); tidy(ax)
            if k > 0: ax.set_yticklabels([])
            if row == 1: ax.set_xlabel(xl)
            else: ax.set_title(xl, fontsize=10.5)
        axes[row, 0].set_ylabel(label)
    fig.tight_layout(); PS.save(fig, "cross_corr_zone_stat")
    print("fig25 saved"); print(bd.to_string()); print(ind.to_string())
