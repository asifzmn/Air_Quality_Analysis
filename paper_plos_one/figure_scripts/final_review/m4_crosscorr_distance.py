# Usage: python m4_crosscorr_distance.py   (read-only; uses cross_correlation/evidence_df.csv and leader_follower_zone.ipynb)
"""M4/M5: all-window vs flagged-window correlation means per Indian state / BD division, Spearman rho with distance,
window spacing, and the flag rule (from the notebook source and re-derived from the CSV)."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
R = "/home/asif/PycharmProjects/Air_Quality_Analysis/cross_correlation/"
d = pd.read_csv(R + "evidence_df.csv", parse_dates=["start_date", "end_date"],
                usecols=["start_date", "end_date", "india_zone", "bangladesh_zone", "zero_lag_correlation", "max_correlation",
                         "max_correlation_lag", "transport_detected", "correlation_diff_pvalue", "season"])
print("windows:", len(d), "| flagged:", int(d.transport_detected.sum()))
# window geometry
p = d[(d.india_zone == d.india_zone.iloc[0]) & (d.bangladesh_zone == d.bangladesh_zone.iloc[0])].sort_values("start_date")
print("window length (end-start):", (d.end_date - d.start_date).value_counts().to_dict())
print("start spacing within one pair:", p.start_date.diff().value_counts().to_dict())
print("windows per pair (median):", int(d.groupby(["india_zone", "bangladesh_zone"]).size().median()), "| hours covered by each hour ~", round(167 / 23, 1), "windows")
# flag rule, from source
nb = json.load(open(R + "leader_follower_zone.ipynb"))
src = "".join(nb["cells"][11]["source"])
for key in ["if max_corr_lag > 0:", "se_diff =", "transport_detected = (p_value_diff< 0.05) or (max_corr > 0.5)", "if corr > max_corr:"]:
    print("notebook cell 11 contains:", repr(key), key in src)
# flag rule, re-derived
rule = (d.max_correlation_lag > 0) & ((d.correlation_diff_pvalue < 0.05) | (d.max_correlation > 0.5))
print("re-derived rule (lag>0 AND (p<0.05 OR r>0.5)) reproduces flags exactly:", bool((rule == d.transport_detected).all()))
print("flagged with lag 0:", int((d.transport_detected & (d.max_correlation_lag == 0)).sum()),
      "| zero-lag windows with r>0.5 (never flagged):", int(((d.max_correlation_lag == 0) & (d.max_correlation > 0.5)).sum()))
nz = d[d.max_correlation_lag > 0]
print("non-zero-lag windows:", len(nz), "| of these p<0.05:", int((nz.correlation_diff_pvalue < 0.05).sum()),
      "| zero-lag windows:", int((d.max_correlation_lag == 0).sum()), f"({100*(d.max_correlation_lag==0).mean():.1f}%)")
f = d[d.transport_detected]

def table(col):
    g, gf = d.groupby(col), f.groupby(col)
    return pd.DataFrame({"all_zero_r": g.zero_lag_correlation.mean(), "all_best_r": g.max_correlation.mean(),
                         "flag_zero_r": gf.zero_lag_correlation.mean(), "flag_best_r": gf.max_correlation.mean(),
                         "flag_lag_h": gf.max_correlation_lag.mean(), "pct_flagged": 100 * g.transport_detected.mean()})
# approximate state reference points (capital / centre); Dhaka 23.81N 90.41E
ref = {"West Bengal": (22.57, 88.36), "Bihar": (25.60, 85.10), "Jharkhand": (23.34, 85.31), "Odisha": (20.30, 85.80),
       "Chhattisgarh": (21.25, 81.63), "Uttar Pradesh": (26.85, 80.95), "Madhya Pradesh": (23.26, 77.41),
       "Telangana": (17.39, 78.49), "Haryana": (29.00, 76.10), "NCT": (28.61, 77.20)}
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))
ti = table("india_zone"); ti["km_to_Dhaka"] = [hav(ref[s], (23.81, 90.41)) for s in ti.index]
print("\nIndian states (leaders), sorted by distance:\n" + ti.sort_values("km_to_Dhaka").round(2).to_string())
for c in ["all_zero_r", "all_best_r", "flag_best_r", "pct_flagged"]:
    r = spearmanr(ti.km_to_Dhaka, ti[c]); print(f"Spearman rho(distance, {c}) = {r.correlation:.2f}, p = {r.pvalue:.4f}")
print("\nBangladeshi divisions (followers):\n" + table("bangladesh_zone").sort_values("all_best_r").round(2).to_string())
print("\nBy season (calendar DJF/MAM/JJA/SON):\n" + table("season").round(2).to_string())
