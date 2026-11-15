"""Fig 20 (corrected): fire FRP vs PM2.5 on a continuous daily index (FRP = 0 on fire-free days), all 55 zones.
(a) raw daily series, (b) both series deseasonalized (minus 31-day centred rolling mean). Data: fire_rerun_continuous.csv."""
import os, sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import plos_style as PS; PS.setup()
EDGE = "#2c4a66"; shades = ["#2f5f8f", "#4677a8", "#6390bd", "#86aacd", "#a9c2da", "#ccdae6"]
d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_rerun_continuous.csv"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), sharey=True)
for ax, kind, title in zip(axes, ["raw", "deseas"], ["(a) Daily series", "(b) Seasonal cycle removed"]):
    p = d[d.kind == kind].pivot_table(index="rad", columns="lag", values="r", aggfunc="mean")
    w = 0.13
    for k, lag in enumerate(p.columns):
        ax.bar([i + (k - 2.5) * w for i in range(len(p))], p[lag], width=w, color=shades[k], edgecolor=EDGE, linewidth=0.4, label=f"{lag} d")
    ax.set_xticks(range(len(p))); ax.set_xticklabels([str(r) for r in p.index])
    ax.set_xlabel("Search radius around each zone (km)"); ax.axhline(0, color="#666", lw=0.8); ax.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e9e9e9"); ax.set_axisbelow(True)
axes[0].set_ylabel("Mean correlation (fire FRP vs. PM$_{2.5}$)")
axes[0].legend(title="Fire lead", frameon=False, ncol=6, loc="upper left", fontsize=8, title_fontsize=8)
axes[0].set_ylim(-0.04, 0.12)
fig.tight_layout(); PS.save(fig, "Fire_analysis")
print(d[d.kind == "raw"].groupby(["rad", "lag"]).r.mean().max(), d[d.kind == "deseas"].groupby(["rad", "lag"]).r.mean().max())
