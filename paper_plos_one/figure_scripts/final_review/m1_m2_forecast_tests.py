# Usage: python m1_m2_forecast_tests.py   (read-only; writes m1_m2_test_predictions.csv next to this script)
"""M1/M2: rebuild the Bayesian Ridge forecast exactly as forecast_reproduce_persistence.py / AQ_forecast_model.ipynb,
check Table 4 (BR + persistence RMSE), then block-bootstrap CI of RMSE ratio BR/persistence, Diebold-Mariano (HAC,
squared-error loss) per horizon, and a PM-only BR ablation."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge

HERE = os.path.dirname(os.path.abspath(__file__))
def fill_days(x, days=3):
    ss = [x.shift(s, freq="D") for s in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna(pd.concat(ss, axis=1).mean(axis=1))

pm = pd.read_csv("/home/asif/PycharmProjects/Air_Quality_Analysis/exploration/Dhaka_city_and_nearby_readings.csv", index_col="time", parse_dates=True)
pm = pm[["Tungi_Dhaka"]]["2018":"2021"].apply(fill_days, args=[3]); pm.columns = ["Reading"]
w = pd.read_csv("/home/asif/Desktop/Work/Dataset/AQ Dataset/Wunderground Data/Compressed/Dhaka.csv", index_col="Time", parse_dates=True)
w.Temperature = w.Temperature.mask(w.Temperature < 0); w["Dew Point"] = w["Dew Point"].mask(w["Dew Point"] < 0)
w.Pressure = w.Pressure.mask(w.Pressure == 0); w.Humidity = w.Humidity.mask(w.Humidity == 0)
w = w[["Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure"]].resample("H").mean().apply(fill_days, args=[3])
daily = pd.concat((pm, w), axis=1).resample("D").mean().round(2)
aug = daily.copy()
for i in range(14):
    s = daily.shift(i); s.columns = s.columns + f" {i} days ago"; aug = pd.concat((aug, s), axis=1)
for i in range(1, 8):
    s = daily[["Reading"]].shift(-i); s.columns = s.columns + f" {i} day after"; aug = pd.concat((aug, s), axis=1)
aug = aug.dropna().drop(daily.columns, axis=1)
ycols = [f"Reading {i} day after" for i in range(1, 8)]
X, y = aug.drop(ycols, axis=1), aug[ycols]
nX, ny = MinMaxScaler().fit(X), MinMaxScaler().fit(y)
Xs = pd.DataFrame(nX.transform(X), columns=X.columns, index=X.index); ys = pd.DataFrame(ny.transform(y), columns=y.columns, index=y.index)
H = 365
Xtr, Xte = Xs.iloc[:-H], Xs.iloc[-H + 1::2]; ytr, yte = ys.iloc[:-H], ys.iloc[-H + 1::2]
print(f"samples {len(Xs)} ({aug.index.min().date()}..{aug.index.max().date()}); train {len(Xtr)} to {Xtr.index.max().date()}; "
      f"test {len(Xte)} ({Xte.index.min().date()}..{Xte.index.max().date()}); features {X.shape[1]}")

yt = yte.values; n = len(yt)
br = MultiOutputRegressor(BayesianRidge(alpha_1=1e-6, alpha_2=1e-4)).fit(Xtr, ytr); P_br = br.predict(Xte)
today = X.loc[Xte.index, "Reading 0 days ago"].values
P_pers = np.column_stack([(today - ny.data_min_[k]) / ny.data_range_[k] for k in range(7)])
pmcols = [c for c in X.columns if c.startswith("Reading")]
br_pm = MultiOutputRegressor(BayesianRidge(alpha_1=1e-6, alpha_2=1e-4)).fit(Xtr[pmcols], ytr); P_pm = br_pm.predict(Xte[pmcols])
rmse = lambda P: np.sqrt(((P - yt) ** 2).mean(axis=0))
T4_BR = np.array([0.077, 0.105, 0.114, 0.120, 0.123, 0.125, 0.125]); T4_P = np.array([0.077, 0.106, 0.111, 0.130, 0.139, 0.147, 0.144])
print("\nRMSE (normalised), days 1-7")
print("BR full      ", rmse(P_br).round(3), "| matches Table 4:", np.allclose(rmse(P_br).round(3), T4_BR))
print("Persistence  ", rmse(P_pers).round(3), "| matches Table 4:", np.allclose(rmse(P_pers).round(3), T4_P))
print("BR PM-only   ", rmse(P_pm).round(3), f"({len(pmcols)} features)")
print("scaler: target range %.1f, min %.1f ug/m3 (1 unit ~ %.0f ug/m3)" % (ny.data_range_[0], ny.data_min_[0], ny.data_range_[0]))

rng = np.random.default_rng(0); B, bl = 2000, 7
def boot_ratio(Pa, Pb):
    out = np.empty((B, 7))
    for b in range(B):
        st = rng.integers(0, n - bl + 1, size=n // bl + 1)
        idx = np.concatenate([np.arange(s, s + bl) for s in st])[:n]
        out[b] = np.sqrt(((Pa[idx] - yt[idx]) ** 2).mean(0)) / np.sqrt(((Pb[idx] - yt[idx]) ** 2).mean(0))
    return np.percentile(out, [2.5, 97.5], axis=0)
def dm(Pa, Pb, k, lags=4):
    d = (Pa[:, k] - yt[:, k]) ** 2 - (Pb[:, k] - yt[:, k]) ** 2
    r = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}); return r.params[0], r.pvalues[0]
ci1 = boot_ratio(P_br, P_pers); ci2 = boot_ratio(P_br, P_pm)
print(f"\n7-day moving-block bootstrap (B={B}, seed 0), 95% CI of RMSE ratio; Diebold-Mariano (HAC, maxlags=4) p")
print("day | BR/pers ratio  CI            DM p | BRfull/BRpm ratio CI          DM p")
for k in range(7):
    print(f" {k+1}  | {rmse(P_br)[k]/rmse(P_pers)[k]:.3f} [{ci1[0,k]:.3f}, {ci1[1,k]:.3f}]  {dm(P_br,P_pers,k)[1]:.3f} | "
          f"{rmse(P_br)[k]/rmse(P_pm)[k]:.3f} [{ci2[0,k]:.3f}, {ci2[1,k]:.3f}]  {dm(P_br,P_pm,k)[1]:.3f}")
print("persistence also vs PM-only BR, DM p by day:", [round(dm(P_pm, P_pers, k)[1], 3) for k in range(7)])
inv = lambda a: a * ny.data_range_ + ny.data_min_
print("test-target max %.1f vs training-target max %.1f ug/m3" % (inv(yt).max(), inv(ytr.values).max()))
rows = {}
for name, P in [("obs", yt), ("BR", P_br), ("BR_PMonly", P_pm), ("persistence", P_pers)]:
    for k in range(7): rows[f"{name}_d{k+1}"] = inv(P)[:, k]
out = pd.DataFrame(rows, index=Xte.index.rename("forecast_origin_UTCday")).round(2)
out.to_csv(os.path.join(HERE, "m1_m2_test_predictions.csv")); print("saved", os.path.join(HERE, "m1_m2_test_predictions.csv"), "(ug/m3)")
