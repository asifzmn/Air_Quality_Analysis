"""SI stacked bar: share of zone-hours in each US EPA PM2.5 category by division, 2018-2021 (observed hourly values, 55 zones).
Sequential blue palette (no red). Output: out/stacked_bar.(png|tif) and aq_category_by_division.csv."""
import sys, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis"); sys.path.insert(0, HERE)
import plos_style as S
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years

s = read_bd_data_4_years()[1]
bins = [-1, 12, 35.4, 55.4, 150.4, 250.4, 1e9]
lab = ["Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]
st = s.stack().reset_index(); st.columns = ["t", "zone", "v"]
st["div"] = st.zone.str.split("_").str[-1]
st["cat"] = pd.cut(st.v, bins, labels=lab).astype(str)
t = (pd.crosstab(st["div"], st["cat"], normalize="index") * 100)[lab]
order = st.groupby("div").v.median().sort_values().index      # most polluted at top
t = t.loc[order]
t.round(2).to_csv(HERE + "/aq_category_by_division.csv")
print(t.round(2).to_string())

S.setup()
import matplotlib.pyplot as plt
cols = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]   # official US EPA AQI category colors (exception to no-red rule)
fig, ax = plt.subplots(figsize=(S.FULL_W, 3.2))
left = pd.Series(0.0, index=t.index)
for c, col in zip(lab, cols):
    ax.barh(t.index, t[c], left=left, color=col, edgecolor=S.EDGE, lw=0.4, height=0.68, label=c)
    for y, (l, w) in enumerate(zip(left, t[c])):
        if w >= 6:
            ax.text(l + w / 2, y, f"{w:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if col in cols[3:] else S.INK)
    left += t[c]
ax.set_xlim(0, 100); ax.set_xlabel("Share of hours (%)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=7)
fig.tight_layout()
S.save(fig, "stacked_bar")
