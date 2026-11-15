"""Rebuild the forecasting dataset exactly as AQ_forecast_model.ipynb (cells 23-29, 35-48, 59, 64, 68),
verify it by re-training the deterministic Bayesian Ridge model (expect RMSE 0.077 day 1 ... 0.125 day 7),
then add a persistence baseline (forecast for day t+k = observed value on day t) on the same test days and metrics."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, median_absolute_error

def fill_days(x, days=3):                                         # notebook cell 28
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
print("shapes train/test:", Xtr.shape, Xte.shape, "| sample period", aug.index.min().date(), "to", aug.index.max().date(),
      "| train ends", Xtr.index.max().date(), "| test", Xte.index.min().date(), "to", Xte.index.max().date())

def metrics(yt, pred, name):                                      # notebook cell 114 (normalised scale)
    out = []
    for i in range(7):
        r, p = yt.iloc[:, i].values, pred[:, i]
        out.append(dict(model=name, day=i + 1, rmse=round(np.sqrt(mean_squared_error(r, p)), 3),
                        mape=round(100 * mean_absolute_percentage_error(r + 1e-1, p), 3) if False else round(mean_absolute_percentage_error(r + 1e-1, p), 3),
                        mdae=round(median_absolute_error(r, p), 3)))
    return out

br = MultiOutputRegressor(BayesianRidge(alpha_1=1e-6, alpha_2=1e-4)).fit(Xtr, ytr)
res = metrics(yte, br.predict(Xte), "BayesianRidge (re-trained)")
# persistence: today's observed reading, mapped to each target's MinMax scale
today = X.loc[Xte.index, "Reading 0 days ago"].values
pers = np.column_stack([(today - ny.data_min_[k]) / ny.data_range_[k] for k in range(7)])
res += metrics(yte, pers, "Persistence")
out = pd.DataFrame(res).pivot(index="day", columns="model")
print(out.to_string())
# raw-scale (ug/m3) RMSE for interpretability
inv = lambda a: a * ny.data_range_ + ny.data_min_
yt_raw = inv(yte.values)
for name, pred in [("BayesianRidge", br.predict(Xte)), ("Persistence", pers)]:
    print(name, "RMSE in ug/m3 by day:", np.round(np.sqrt(((inv(pred) - yt_raw) ** 2).mean(axis=0)), 1))
out.to_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/forecast_persistence_baseline.csv")
