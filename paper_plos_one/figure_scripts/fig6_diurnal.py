"""Fig 6 (corrected): day vs night PM2.5 by month, with Berkeley Earth UTC timestamps converted to local time (UTC+6).
Day/night from local sunrise/sunset (2017-2020 table; 2021 uses the 2020 calendar-day times).
Statistic: for each date, mean of day hours and mean of night hours of the national (55-zone mean) series; paired Wilcoxon per month."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from scipy.stats import wilcoxon
import plos_style as S
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
from paths import sun_rise_set_time_2017_2020

pm = read_bd_data_4_years()[1]
pm.index = pm.index + pd.Timedelta(hours=6)                         # UTC -> Asia/Dhaka
nat = pm.mean(axis=1)                                               # national mean of 55 zones

sun = pd.read_csv(sun_rise_set_time_2017_2020, sep="\t")
sun["date"] = pd.to_datetime(sun["Date "])
sun["rise"] = pd.to_datetime(sun["Date "] + " " + sun["Sunrise "]).dt.hour + pd.to_datetime(sun["Date "] + " " + sun["Sunrise "]).dt.minute / 60
sun["set"] = pd.to_datetime(sun["Date "] + " " + sun["Sunset "]).dt.hour + pd.to_datetime(sun["Date "] + " " + sun["Sunset "]).dt.minute / 60
sun = sun.set_index(sun.date.dt.strftime("%m-%d"))[["rise", "set"]].groupby(level=0).mean()   # calendar-day mean times
md = nat.index.strftime("%m-%d"); h = nat.index.hour + 0.5                                       # hour midpoint
is_day = (h >= sun.reindex(md)["rise"].values) & (h < sun.reindex(md)["set"].values)

df = pd.DataFrame({"pm": nat.values, "day": is_day, "date": nat.index.normalize()}, index=nat.index).dropna()
dd = df.groupby(["date", "day"])["pm"].mean().unstack().dropna()
dd.columns = ["night", "day"]
dd["month"] = dd.index.month

rows = []
for m, g in dd.groupby("month"):
    stat, p = wilcoxon(g["night"], g["day"])
    rows.append(dict(month=m, n_days=len(g), day_median=g["day"].median(), night_median=g["night"].median(),
                     ratio=g["night"].median() / g["day"].median(), pct_nights_higher=100 * (g["night"] > g["day"]).mean(), p=p))
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv(S.OUT + "/../diurnal_day_night_stats.csv", index=False)
print("winter (DJF) ratio %.2f | monsoon (JJAS) ratio %.2f" % (
    dd[dd.month.isin([12, 1, 2])].night.median() / dd[dd.month.isin([12, 1, 2])].day.median(),
    dd[dd.month.isin([6, 7, 8, 9])].night.median() / dd[dd.month.isin([6, 7, 8, 9])].day.median()))

# ---------------- figure ----------------
S.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(S.FULL_W, 3.0))
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
w = 0.34
for k, (col, face) in enumerate([("day", S.LIGHT), ("night", S.STEEL)]):
    data = [dd.loc[dd.month == m, col].values for m in range(1, 13)]
    pos = np.arange(1, 13) + (k - 0.5) * (w + 0.04)
    ax.boxplot(data, positions=pos, widths=w, whis=(5, 95), showfliers=False, patch_artist=True,
               medianprops=dict(color=S.EDGE, lw=1.3), whiskerprops=dict(color=S.GREYLINE, lw=0.8),
               capprops=dict(lw=0), boxprops=dict(facecolor=face, edgecolor=S.EDGE, lw=0.6))
top = [np.percentile(dd.loc[dd.month == m, ["day", "night"]].values, 95) for m in range(1, 13)]
for i, r in res.iterrows():
    star = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else "n.s."
    ax.text(r.month, top[i] + 6, star, ha="center", va="bottom", fontsize=7, color=S.INK)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(months)
ax.set_ylabel("Daily mean PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
ax.set_xlim(0.4, 12.6); S.aqi_hbands(ax, max(top) + 25)
ax.legend(handles=[Patch(facecolor=S.LIGHT, edgecolor=S.EDGE, lw=0.6, label="Daytime (sunrise–sunset)"),
                   Patch(facecolor=S.STEEL, edgecolor=S.EDGE, lw=0.6, label="Night-time")],
          frameon=False, loc="upper center", ncol=2)
S.save(fig, "diurnal_distribution")
