"""Validate Berkeley Earth (BE) point estimates against the US Embassy Dhaka reference monitor
(EPA ScienceHub doi:10.23719/1528454; Sarwar et al. 2023). Daily means, 2018-2021; diurnal profiles by season."""
import sys, csv, numpy as np, pandas as pd
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
V = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/validation_us_embassy/"
rows = [r for r in csv.reader(open(V + "csv/Figure_1_data__data.csv"))][2:]
emb = pd.Series({pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(float(r[0])) - 1): float(r[1]) for r in rows if r[0].strip()}).sort_index()["2018":"2021"]
pm = read_bd_data_4_years()[1]
def daily(series, local=True):
    s = series.copy()
    if local: s.index = s.index + pd.Timedelta(hours=6)     # UTC -> Asia/Dhaka
    return s.resample("D").mean().where(s.resample("D").count() >= 18)
out = []
for z in ["Dhaka_Dhaka", "Paltan_Dhaka", "Azimpur_Dhaka", "Tungi_Dhaka"]:
    for local in (True, False):
        b = daily(pm[z], local); j = pd.concat([emb.rename("emb"), b.rename("be")], axis=1).dropna()
        slope, icpt = np.polyfit(j.emb, j.be, 1)
        out.append(dict(zone=z, days="local" if local else "UTC", n=len(j), r=j.corr().iloc[0, 1], spearman=j.corr("spearman").iloc[0, 1],
                        mean_emb=j.emb.mean(), mean_be=j.be.mean(), MB=(j.be - j.emb).mean(), NMB_pct=100 * (j.be - j.emb).sum() / j.emb.sum(),
                        RMSE=np.sqrt(((j.be - j.emb) ** 2).mean()), slope=slope, intercept=icpt))
res = pd.DataFrame(out).round(3); print(res.to_string(index=False)); res.to_csv(V + "daily_validation_metrics.csv", index=False)

# seasonal means and winter/summer ratio (Dhaka_Dhaka, local days); seasons as in Sarwar et al.: winter DJF, pre-monsoon MAM, monsoon JJAS, post-monsoon ON
j = pd.concat([emb.rename("US Embassy"), daily(pm["Dhaka_Dhaka"]).rename("Berkeley Earth")], axis=1).dropna()
season = pd.Series(j.index.month, index=j.index).map({12: "winter", 1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon",
                                                       6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon"})
sm = j.groupby(season).mean().round(1); print("\nSeasonal means (overlapping days):\n", sm.to_string())
print("winter/monsoon ratio: Embassy %.2f | BE %.2f" % (sm.loc["winter", "US Embassy"] / sm.loc["monsoon", "US Embassy"], sm.loc["winter", "Berkeley Earth"] / sm.loc["monsoon", "Berkeley Earth"]))
print("share of days > 65: Embassy %.1f%% | BE %.1f%%" % ((j["US Embassy"] > 65).mean() * 100, (j["Berkeley Earth"] > 65).mean() * 100))
yr = j.groupby(j.index.year).mean().round(1); print("\nAnnual means:\n", yr.to_string())
j.to_csv(V + "daily_paired_dhaka.csv")

# diurnal profile check (Embassy hour = local clock): which shift of BE (UTC) best matches?
r3 = [r for r in csv.reader(open(V + "csv/Figure_3_data__data.csv"))][3:]
prof = pd.DataFrame({"winter": [float(r[1]) for r in r3[:24]], "monsoon": [float(r[3]) for r in r3[:24]]}, index=range(24))
s = pm["Dhaka_Dhaka"]["2018":"2021"]
print("\nDiurnal-profile correlation with Embassy, by shift applied to BE UTC hours:")
for sh in (0, 3, 6, 9):
    t = s.copy(); t.index = t.index + pd.Timedelta(hours=sh)
    w = t[t.index.month.isin([12, 1, 2])].groupby(t[t.index.month.isin([12, 1, 2])].index.hour).mean()
    m = t[t.index.month.isin([6, 7, 8, 9])].groupby(t[t.index.month.isin([6, 7, 8, 9])].index.hour).mean()
    print(f"  shift +{sh} h: winter r = {np.corrcoef(w, prof.winter)[0,1]:.3f}, monsoon r = {np.corrcoef(m, prof.monsoon)[0,1]:.3f}, "
          f"BE winter peak hour {w.idxmax()}, Embassy winter peak hour {prof.winter.idxmax()}")
