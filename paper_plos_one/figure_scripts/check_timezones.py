import sys, numpy as np, pandas as pd, xarray as xr, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
w = pd.read_csv("/home/asif/Desktop/Work/Dataset/AQ Dataset/Wunderground Data/Compressed/Dhaka.csv", index_col=0, parse_dates=True)
print("Wunderground: hour of max mean temperature =", w["Temperature"].groupby(w.index.hour).mean().idxmax(),
      "| hour of min =", w["Temperature"].groupby(w.index.hour).mean().idxmin())
try:
    ds = xr.open_dataset("/home/asif/PycharmProjects/Air_Quality_Analysis/meteorology/meteoblue_data_2019_bd_and_neighbours.nc")
    v = [k for k in ds.data_vars if "emperature" in k][0]; t = ds[v].to_dataframe().reset_index()
    tc = [c for c in t.columns if "time" in c.lower()][0]; t[tc] = pd.to_datetime(t[tc])
    print("Meteoblue (%s): hour of max mean =" % v, t.groupby(t[tc].dt.hour)[v].mean().idxmax())
except Exception as e: print("meteoblue check failed:", e)
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
pm = read_bd_data_4_years()[1]
print("Berkeley Earth PM2.5 (UTC): hour of max mean =", pm.mean(axis=1).groupby(pm.index.hour).mean().idxmax(),
      "| min =", pm.mean(axis=1).groupby(pm.index.hour).mean().idxmin())
# day/night with and without UTC->local (+6 h) conversion
from data_preparation.additional_data_preparation import get_diurnal_period
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): dn = get_diurnal_period()
for label, idx in [("as in paper (UTC vs local sun times)", pm.index), ("corrected (UTC+6 local)", pm.index + pd.Timedelta(hours=6))]:
    x = pm.copy(); x.index = idx; x = x["2018":"2020"]
    lab = dn.reindex(x.index)
    st = x.stack().rename("v").reset_index(); st.columns = ["time", "zone", "v"]
    st["dn"] = lab.reindex(st["time"]).values; st["m"] = st["time"].dt.month
    out = st.groupby(["m", "dn"])["v"].median().unstack()
    print("\n" + label + "  (median, 2018-2020)\n", out.loc[[12, 1, 6, 7, 8]].round(2).to_string(),
          "\n  Dec night/day ratio:", round(out.loc[12, "night"] / out.loc[12, "day"], 2),
          "| all-month mean night/day:", round((out["night"] / out["day"]).mean(), 2))
