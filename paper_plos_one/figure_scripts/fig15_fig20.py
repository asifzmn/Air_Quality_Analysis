"""Fig 15 (imputation benchmark, values from data_preparation/missing_values.ipynb cell 34) and
Fig 20 (fire-PM2.5 correlation, meteorology/FIRE/fire_pm25_summary.csv). Steel-blue / grey palette."""
import os, sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
STEEL, STEEL_GREY, EDGE = "#3f78ad", "#b3c3d1", "#2c4a66"
sys.path.insert(0, "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"); import plos_style as PS; PS.setup()
def tidy(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e9e9e9"); ax.set_axisbelow(True)

# ---------- Fig 15 ----------
df = pd.DataFrame({"Model": ["KNN", "Random\nForest", "Linear\nRegression", "Bayesian\nRidge", "LightGBM"],
                   "MAE": [17.281972, 16.164294, 14.913600, 14.846514, 14.620064],
                   "RMSE": [26.213015, 25.510714, 23.302608, 23.246506, 22.839799],
                   "R2": [0.766259, 0.778616, 0.815281, 0.816170, 0.822546]})
cols = [STEEL_GREY] * 4 + [STEEL]
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
for ax, (m, title, fmt) in zip(axes, [("MAE", "MAE (lower is better)", "{:.2f}"),
                                       ("RMSE", "RMSE (lower is better)", "{:.2f}"),
                                       ("R2", "R$^2$ (higher is better)", "{:.3f}")]):
    ax.bar(df.Model, df[m], color=cols, edgecolor=EDGE, linewidth=0.5, width=0.7)
    for i, v in enumerate(df[m]):
        ax.text(i, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.set_title(title); ax.set_ylim(0, df[m].max() * 1.12)
    ax.tick_params(axis="x", labelsize=7); tidy(ax)
fig.tight_layout(); PS.save(fig, "imputation_evaluation")
print("fig15 saved")

# ---------- Fig 20 ----------
d = pd.read_csv("/home/asif/PycharmProjects/Air_Quality_Analysis/meteorology/FIRE/fire_pm25_summary.csv")
d = d[d.Radius_km.isin([10, 25, 50, 75, 100, 150, 200])]
p = d.pivot(index="Radius_km", columns="Lag_Days", values="Avg_Correlation")
shades = ["#2f5f8f", "#4677a8", "#6390bd", "#86aacd", "#a9c2da", "#ccdae6"]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
w = 0.13
for k, lag in enumerate(p.columns):
    x = [i + (k - 2.5) * w for i in range(len(p))]
    ax.bar(x, p[lag], width=w, color=shades[k], edgecolor=EDGE, linewidth=0.4, label=f"{lag} d")
ax.set_xticks(range(len(p))); ax.set_xticklabels([f"{r}" for r in p.index])
ax.set_xlabel("Search radius around each zone (km)"); ax.set_ylabel("Mean correlation (fire FRP vs. PM$_{2.5}$)")
ax.axhline(0, color="#666", lw=0.8)
ax.legend(title="Lag", frameon=False, ncol=6, loc="upper left", fontsize=9, title_fontsize=9)
ax.set_ylim(min(-0.01, p.values.min() - 0.005), p.values.max() * 1.25)
tidy(ax); fig.tight_layout(); PS.save(fig, "Fire_analysis")
print("fig20 saved")
