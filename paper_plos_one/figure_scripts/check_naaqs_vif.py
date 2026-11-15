"""R1.3.9 / RQ1: zone means vs Bangladesh NAAQS (annual 15, 24-h 65 ug/m3) and WHO 2021 (5 / 15).
R2.M4b: correlation matrix + VIF of the forecasting input variables (Tungi zone + Dhaka Wunderground, daily, 2019-2021)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
s = read_bd_data_4_years()[1]                     # hourly, 55 zones, 2018-2021, not imputed
print("series:", s.shape, s.index.min(), s.index.max())

# daily means only where >= 18 of 24 hours are present (75% completeness)
cnt = s.resample("D").count(); daily = s.resample("D").mean().where(cnt >= 18)
zone_mean = s.mean()
annual = s.groupby(s.index.year).mean()
print("\n== annual standard (15) ==")
print("zone 4-yr means: min %.1f (%s), max %.1f (%s); zones > 15: %d/55; > 5 (WHO): %d/55"
      % (zone_mean.min(), zone_mean.idxmin(), zone_mean.max(), zone_mean.idxmax(), (zone_mean > 15).sum(), (zone_mean > 5).sum()))
print("zone-year means > 15: %d of %d; min zone-year mean %.1f" % ((annual > 15).values.sum(), annual.size, annual.values.min()))
print("\n== 24-h standard (65) ==")
valid = daily.notna().sum()
ex65 = (daily > 65).sum() / valid * 100; ex15 = (daily > 15).sum() / valid * 100
print("valid zone-days: %d" % valid.sum())
print("pooled share of zone-days > 65: %.1f%%; > WHO 15: %.1f%%" % ((daily > 65).values.sum() / valid.sum() * 100, (daily > 15).values.sum() / valid.sum() * 100))
print("per-zone share of days > 65: min %.1f%% (%s), median %.1f%%, max %.1f%% (%s)" % (ex65.min(), ex65.idxmin(), ex65.median(), ex65.max(), ex65.idxmax()))
nat = s.mean(axis=1).resample("D").mean()
print("national-mean series: days > 65: %.1f%%" % ((nat > 65).mean() * 100))
ex65.rename("pct_days_above_65").to_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/naaqs_zone_exceedance.csv")

# ---- VIF / correlation of forecasting inputs ----
from statsmodels.stats.outliers_influence import variance_inflation_factor
w = pd.read_csv("/home/asif/Desktop/Work/Dataset/AQ Dataset/Wunderground Data/Compressed/Dhaka.csv", index_col=0, parse_dates=True)
w = w[["Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure"]].resample("H").mean()
d = pd.concat([s[["Tungi_Dhaka"]].rename(columns={"Tungi_Dhaka": "PM2.5"}), w], axis=1)["2019":"2021"].resample("D").mean().dropna()
print("\n== forecasting inputs (daily, Tungi + Dhaka airport, 2019-2021), n=%d ==" % len(d))
print(d.corr().round(2).to_string())
X = (d - d.mean()) / d.std(); X["const"] = 1.0
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1] - 1)], index=d.columns)
print("\nVIF (same-day variables):\n" + vif.round(2).to_string())
Xm = X.drop(columns=["Dew Point"])
vif2 = pd.Series([variance_inflation_factor(Xm.values, i) for i in range(Xm.shape[1] - 1)], index=Xm.columns[:-1])
print("\nVIF without Dew Point:\n" + vif2.round(2).to_string())
d.corr().round(3).to_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/forecast_inputs_corr.csv")
