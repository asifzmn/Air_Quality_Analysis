"""Overview figure: Bangladesh PM2.5 at a glance, 2018-2021 (local time, observed data).
(a) Calendar strip: every day of the national (55-zone mean) daily PM2.5, coloured by US EPA AQI category (24-h breakpoints).
(b) Map: share of days each zone exceeded the Bangladesh 24-h standard (65 ug/m3), daily means with >= 18 valid hours.
(c) Share of days in each AQI category per year.
Output: out/overview_calendar.(png|tif), overview_calendar_stats.csv"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis"); sys.path.insert(0, HERE)
import plos_style as S
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years

r = read_bd_data_4_years(); meta, s = r[0], r[1].copy()
s.index = s.index + pd.Timedelta(hours=6)                       # UTC -> local
s = s["2018":"2021"]
nat_h = s.mean(axis=1)
cnt = nat_h.resample("D").count(); nat = nat_h.resample("D").mean().where(cnt >= 18)

bins = [-1, 12, 35.4, 55.4, 150.4, 250.4, 1e9]
lab = ["Good", "Moderate", "Unhealthy for\nsensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]
cols = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]   # official US EPA AQI colours
cat = pd.cut(nat, bins, labels=False)

# zone exceedance of the 24-h standard
zc = s.resample("D").count(); zd = s.resample("D").mean().where(zc >= 18)
pct = (zd > 65).sum() / zd.notna().sum() * 100
lat, lon = meta.loc[pct.index, "Latitude"], meta.loc[pct.index, "Longitude"]

yr = pd.DataFrame({y: pd.Series(cat[str(y)].values).value_counts(normalize=True).reindex(range(6), fill_value=0) * 100
                   for y in [2018, 2019, 2020, 2021]}).T
stats = yr.copy(); stats.columns = [l.replace("\n", " ") for l in lab]
stats["days_above_65_pct"] = [(nat[str(y)].dropna() > 65).mean() * 100 for y in yr.index]
stats.round(2).to_csv(HERE + "/overview_calendar_stats.csv"); print(stats.round(1).to_string())
print(f"zone pct days >65: min {pct.min():.1f} ({pct.idxmin()}) max {pct.max():.1f} ({pct.idxmax()}); national missing days {nat.isna().sum()}")

S.setup()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
import geopandas as gpd

W, H = S.FULL_W, 6.9
fig = plt.figure(figsize=(W, H))
A = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

# (a) calendar strip
ax = A(0.55, 4.95, 6.85, 1.65)
cmap = ListedColormap(cols); norm = BoundaryNorm(np.arange(-0.5, 6.5), 6)
years = [2018, 2019, 2020, 2021]
grid = np.full((4, 366), np.nan)
for k, y in enumerate(years):
    c = cat[str(y)]; grid[k, c.index.dayofyear - 1] = c.values
ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
ax.set_facecolor("#e6e6e6")                                       # missing days
ax.set_yticks(range(4)); ax.set_yticklabels(years)
mstart = pd.date_range("2019-01-01", periods=12, freq="MS").dayofyear - 1
ax.set_xticks(mstart); ax.set_xticklabels([d.strftime("%b") for d in pd.date_range("2019-01-01", periods=12, freq="MS")], ha="left")
for x in mstart[1:]: ax.axvline(x - 0.5, color="white", lw=0.8)
for y in np.arange(0.5, 3.5): ax.axhline(y, color="white", lw=2.5)
lo, hi = pd.Timestamp("2020-03-26").dayofyear - 1, pd.Timestamp("2020-05-30").dayofyear - 1   # Bangladesh general holiday (lockdown)
ax.add_patch(plt.Rectangle((lo - 0.5, 1.55), hi - lo + 1, 0.9, fill=False, ec="black", lw=1.1))
ax.annotate("2020 lockdown\n(26 Mar–30 May)", xy=(hi + 1, 2.0), xytext=(hi + 18, 2.0), fontsize=6.8, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.7, color="black"))
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
ax.set_title("(a) National daily mean PM$_{2.5}$, every day 2018–2021, by US EPA air quality category", loc="left", fontsize=8)
ax.legend(handles=[Patch(fc=c, ec="#555", lw=0.4, label=l.replace("\n", " ")) for c, l in zip(cols, lab)] +
          [Patch(fc="#e6e6e6", ec="#555", lw=0.4, label="Missing")],
          ncol=7, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(-0.02, -0.17), handlelength=1.2, columnspacing=0.9)

# (b), (c): two maps with the same red ramp (deeper = worse); labels for well-known cities
bd = gpd.read_file("/home/asif/Desktop/Work/Dataset/Bangladesh/bdBounds.geojson")
reds = LinearSegmentedColormap.from_list("reds", ["#fde3d6", "#f08060", "#c62a1e", "#6e0b0b"])
zmean = s.mean()
cities = [("Dhaka_Dhaka", "Dhaka", (-64, -4)), ("Chittagong_Chittagong", "Chittagong", (14, -8)),
          ("Sylhet_Sylhet", "Sylhet", (12, 6)), ("Rajshahi_Rajshahi", "Rajshahi", (-66, 8))]
def bdmap(ax, val, vmin, vmax, ticks, label, fmt, title):
    bd.plot(ax=ax, color="#f4f4f4", edgecolor="#9a9a9a", lw=0.6)
    o = np.argsort(val.values)
    sc = ax.scatter(lon[val.index].values[o], lat[val.index].values[o], c=val.values[o], cmap=reds, vmin=vmin, vmax=vmax,
                    s=26, edgecolor="white", lw=0.6, zorder=3)
    for z, name, off in cities:
        ax.annotate(f"{name} {fmt(val[z])}", (lon[z], lat[z]), xytext=off, textcoords="offset points", fontsize=6.5,
                    color=S.INK, arrowprops=dict(arrowstyle="-", lw=0.6, color="#555", shrinkA=0, shrinkB=3), zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_axis_off(); ax.set_aspect("equal")
    cax = ax.inset_axes([0.2, -0.06, 0.6, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=6.8, labelpad=1); cb.ax.tick_params(labelsize=6.3, length=2); cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=8)
axm = A(0.15, 0.6, 3.6, 3.55)
bdmap(axm, pct, 20, 45, [20, 25, 30, 35, 40, 45], "Days above 65 $\\mu$g m$^{-3}$ (%)", lambda v: f"{v:.0f}%",
      "(b) Days above the 24-h standard (65 $\\mu$g m$^{-3}$)")
axc = A(3.85, 0.6, 3.6, 3.55)
bdmap(axc, zmean / 15, 2.5, 5, [2.5, 3, 3.5, 4, 4.5, 5], "Mean PM$_{2.5}$ / annual standard (15 $\\mu$g m$^{-3}$)", lambda v: f"{v:.1f}$\\times$",
      "(c) Mean as a multiple of the annual standard")
print("multiple of standard: min %.2f max %.2f" % ((zmean / 15).min(), (zmean / 15).max()))
S.save(fig, "overview_calendar")
