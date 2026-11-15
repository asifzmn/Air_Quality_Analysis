"""SI correlation heatmaps (zero lag), Berkeley Earth converted to local time.
(a) Wunderground: Dhaka division PM2.5 vs Dhaka airport, hourly 2019-2021.
(b) Meteoblue: mean of the 55 per-zone correlation matrices, hourly 2019 (method of meteoblue_cross_corr.ipynb cells 20-23)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis"); sys.path.insert(0, HERE)
import plos_style as S
from data_preparation import read_all_granularity_data
from data_preparation.spatio_temporal_filtering import read_bd_data
from meteorology.meteoblue_data_preparation import read_meteo_data_file_bd_and_neighbours
from meteorology.wunderground_data_preparation import read_meteo_data_file_wg
from meteorology import get_district_data
shift = pd.Timedelta(hours=6)

reg = read_all_granularity_data()[3]["Dhaka"].copy(); reg.index = reg.index + shift
wg = get_district_data(read_meteo_data_file_wg(), "Dhaka")[["Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure"]]
A = pd.concat([reg["2019":"2021"].rename("PM2.5"), wg], axis=1).dropna().corr()

series = read_bd_data()[1].copy(); series.index = series.index + shift
mb = read_meteo_data_file_bd_and_neighbours()
f = ["Temperature [2 m elevation corrected]", "Longwave Radiation", "Relative Humidity [2 m]", "Mean Sea Level Pressure [MSL]",
     "Wind Speed [10 m]", "Wind Gust", "Precipitation Total", "Cloud Cover Total"]
mats = [pd.concat([series[z]["2019"].rename("PM2.5"), get_district_data(mb, z)[f]], axis=1).corr() for z in series.columns]
B = pd.concat(mats).groupby(level=0).mean().reindex(mats[0].index)[mats[0].columns]
B = B.rename(index=lambda x: x.split(" [")[0], columns=lambda x: x.split(" [")[0])
print(A.round(2).to_string()); print(B.round(2).to_string())
A.round(3).to_csv(HERE + "/heatmap_wunderground_local.csv"); B.round(3).to_csv(HERE + "/heatmap_meteoblue_local.csv")

S.setup()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
cmap = LinearSegmentedColormap.from_list("sr", [S.DEEP, "#ffffff", S.SLATE])
short = {"PM2.5": "PM$_{2.5}$", "Temperature": "Temp.", "Dew Point": "Dew pt.", "Humidity": "RH", "Relative Humidity": "RH",
         "Wind Speed": "Wind", "Wind Gust": "Gust", "Pressure": "Pressure", "Mean Sea Level Pressure": "MSLP",
         "Longwave Radiation": "LW rad.", "Precipitation Total": "Precip.", "Cloud Cover Total": "Cloud"}
nA, nB = len(A), len(B)
c = 0.40                                    # cell size (in), identical in both panels
W, H = S.FULL_W, 0.70 + (nB - 1) * c + 0.32
fig = plt.figure(figsize=(W, H))
xa, xb, yb = 0.80, 0.80 + (nA - 1) * c + 0.95, 0.70
axa = fig.add_axes([xa / W, (yb + (nB - nA) * c) / H, (nA - 1) * c / W, (nA - 1) * c / H])
axb = fig.add_axes([xb / W, yb / H, (nB - 1) * c / W, (nB - 1) * c / H])
axes = [axa, axb]
for ax, M, t in zip(axes, [A, B], ["(a) Wunderground, Dhaka, 2019–2021", "(b) Meteoblue, 55-zone mean, 2019"]):
    n = len(M); V = M.values[1:, :-1]                      # drop diagonal: rows 2..n, cols 1..n-1
    mask = np.triu(np.ones((n - 1, n - 1), bool), 1)
    im = ax.imshow(np.ma.array(V, mask=mask), cmap=cmap, vmin=-1, vmax=1)
    for i in range(n - 1):
        for j in range(i + 1):
            v = V[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.6,
                    fontweight="bold" if j == 0 else "normal", color="white" if abs(v) > 0.6 else S.INK)
    ax.add_patch(Rectangle((-0.5, -0.5), 1, n - 1, fill=False, ec=S.EDGE, lw=1.1))   # PM2.5 column
    ax.set_xticks(range(n - 1)); ax.set_xticklabels([short.get(c, c) for c in M.columns[:-1]], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n - 1)); ax.set_yticklabels([short.get(c, c) for c in M.index[1:]], fontsize=7)
    ax.set_title(t, loc="left", fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
cax = fig.add_axes([xa / W, (yb + 0.55) / H, (nA - 1) * c / W, 0.11 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"], fontsize=7); cb.set_label("Pearson r", fontsize=7.5); cb.outline.set_visible(False)
S.save(fig, "variable_correlation_heatmaps")
