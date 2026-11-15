"""R2.M2c: sensitivity of the selected imputer (LightGBM with forward-lookup features, per zone)
to random single-hour gaps vs. REAL gap patterns (missing-data mask of Satkania, the zone with most gaps,
transplanted onto observed data of other zones). Metrics by gap length."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
from data_preparation.imputer import ImputeWithFeaturesLightGBMForwardLookup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io, contextlib
rng = np.random.default_rng(42)
s = read_bd_data_4_years()[1]
real_mask_src = s["Satkania_Chittagong"].isna().values            # real gap pattern, 13.6% missing

def run_lengths(mask):
    """length of the gap each masked position belongs to"""
    out = np.zeros(len(mask), int); i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            out[i:j] = j - i; i = j
        else: i += 1
    return out

def bucket(L):
    return np.select([L == 1, L <= 23, L <= 120], ["1 h", "2-23 h", "1-5 d"], "> 5 d")

rows = []
for zone in ["Dhaka_Dhaka", "Sylhet_Sylhet", "Chittagong_Chittagong", "Rajshahi_Rajshahi"]:
    x = s[[zone]].copy(); obs = x[zone].notna().values
    schemes = {"random 1-h": (rng.random(len(x)) < 0.10) & obs,
               "real pattern": np.roll(real_mask_src, 24 * 30) & obs}       # shifted 30 d so gaps fall on observed hours
    for name, m in schemes.items():
        xt = x.copy(); xt.values[m, 0] = np.nan
        with contextlib.redirect_stdout(io.StringIO()):
            imp = ImputeWithFeaturesLightGBMForwardLookup(xt, max_iter=5)
        a, p = x.values[m, 0], imp.values[m, 0]
        L = run_lengths(np.isnan(xt.values[:, 0]))[m]
        df = pd.DataFrame({"a": a, "p": p, "b": bucket(L)})
        rows.append([zone, name, "all", len(df), mean_absolute_error(a, p), np.sqrt(mean_squared_error(a, p)), r2_score(a, p)])
        for b, g in df.groupby("b"):
            if len(g) > 30:
                rows.append([zone, name, b, len(g), mean_absolute_error(g.a, g.p), np.sqrt(mean_squared_error(g.a, g.p)), r2_score(g.a, g.p)])
        print(zone, name, "done", flush=True)
res = pd.DataFrame(rows, columns=["zone", "scheme", "gap", "n", "MAE", "RMSE", "R2"]).round(3)
res.to_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/imputation_gap_sensitivity.csv", index=False)
print(res.to_string(index=False))
