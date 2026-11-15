"""Fig 7 (corrected): lagged correlation between meteorology and PM2.5, same method as
meteorology/meteoblue_cross_corr.ipynb (cells 31-36) and wunderground_cross_corr.ipynb (cells 15-19),
but with Berkeley Earth PM2.5 converted from UTC to local time (UTC+6) so that it matches the weather data clocks.
Lag L means weather at time t-L is correlated with PM2.5 at time t (weather leads)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
sys.path.insert(0, "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts")
import plos_style as S
from data_preparation import read_all_granularity_data
from meteorology.meteoblue_data_preparation import read_meteo_data_file_bd_and_neighbours
from meteorology.wunderground_data_preparation import read_meteo_data_file_wg
from meteorology import get_district_data

_, series_all, _, region_series_all, _, _ = read_all_granularity_data()
LAGS = range(24 * 7 * 2)

def lag_curves(pm, met, factors):
    out = {}
    for f in factors:
        out[f] = [pm.corr(met[f].shift(L)) for L in LAGS]
    return pd.DataFrame(out, index=list(LAGS))

def summarise(curves, label):
    best = curves.abs().idxmax()
    return pd.DataFrame({"source": label, "r_lag0": curves.iloc[0], "best_lag_h": best,
                         "r_best": [curves.loc[best[c], c] for c in curves.columns]})

results = {}
for local in (False, True):
    shift = pd.Timedelta(hours=6) if local else pd.Timedelta(0)
    # Meteoblue, Azimpur_Dhaka, 2019
    mb = get_district_data(read_meteo_data_file_bd_and_neighbours(), "Azimpur_Dhaka")
    mb_f = ["Temperature [2 m elevation corrected]", "Longwave Radiation", "Relative Humidity [2 m]",
            "Mean Sea Level Pressure [MSL]", "Wind Speed [10 m]", "Wind Gust", "Precipitation Total", "Cloud Cover Total"]
    pm = series_all["Azimpur_Dhaka"].copy(); pm.index = pm.index + shift; pm = pm["2019"]
    c_mb = lag_curves(pm, mb, mb_f)
    # Wunderground, Dhaka division mean vs Dhaka airport, 2019-2021
    wg = get_district_data(read_meteo_data_file_wg(), "Dhaka")
    wg_f = ["Temperature", "Humidity", "Pressure", "Wind Speed"]
    pm2 = region_series_all["Dhaka"].copy(); pm2.index = pm2.index + shift; pm2 = pm2["2019":"2021"]
    c_wg = lag_curves(pm2, wg, wg_f)
    tag = "local (corrected)" if local else "UTC (as published)"
    results[tag] = (c_mb, c_wg)
    print("\n=====", tag)
    print(summarise(c_mb, "Meteoblue").round(3).to_string())
    print(summarise(c_wg, "Wunderground").round(3).to_string())

c_mb, c_wg = results["local (corrected)"]
pd.concat([summarise(c_mb, "Meteoblue"), summarise(c_wg, "Wunderground")]).round(3).to_csv(S.OUT + "/../meteo_lag_corrected.csv")
c_mb.to_csv(S.OUT + "/../meteo_lag_curves_meteoblue_local.csv"); c_wg.to_csv(S.OUT + "/../meteo_lag_curves_wunderground_local.csv")

# ---------------- figure: lag-correlation curves (first 96 h), 4 variables common to both sources ----------------
S.setup()
import matplotlib.pyplot as plt
style = {"Temperature": (S.DEEP, "-", 1.6), "Pressure": (S.EDGE, (0, (5, 2)), 1.6),
         "Humidity": (S.LIGHT, "-", 1.8), "Wind speed": ("#7f8c99", (0, (1.2, 1.4)), 1.8)}
mb_names = {"Temperature [2 m elevation corrected]": "Temperature", "Mean Sea Level Pressure [MSL]": "Pressure",
            "Relative Humidity [2 m]": "Humidity", "Wind Speed [10 m]": "Wind speed"}
wg_names = {"Temperature": "Temperature", "Pressure": "Pressure", "Humidity": "Humidity", "Wind Speed": "Wind speed"}
H = 96
fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 3.1), sharey=True)
for ax, (curves, names, title) in zip(axes, [(c_mb, mb_names, "(a) Meteoblue, Azimpur zone, 2019"),
                                            (c_wg, wg_names, "(b) Wunderground, Dhaka airport, 2019\u20132021")]):
    for x in range(24, H, 24): ax.axvline(x, color="#e3e3e3", lw=0.6, zorder=0)
    ax.axhline(0, color="#9a9a9a", lw=0.7, zorder=1)
    for col, nm in names.items():
        c, l, w = style[nm]; y = curves[col].iloc[:H + 1]
        ax.plot(y.index, y.values, color=c, ls=l, lw=w, label=nm, zorder=3)
    ax.set_xticks(range(0, H + 1, 12)); ax.set_xlim(0, H); ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("Lag of meteorological variable (hours)"); ax.set_title(title, loc="left")
axes[0].set_ylabel("Pearson r with PM$_{2.5}$")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout()
axes[0].legend(h, l, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.10), handlelength=2.6, borderaxespad=0)
S.save(fig, "bar_chart_lag_hour_meteo")
