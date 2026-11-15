# Usage: python m7_income_partial.py   (read-only; replicates exploration/demographic_analysis.ipynb cells 23-25, 34, 37, 44, 54, 60)
"""M7: rebuild the district per-capita-income vs zone-mean PM2.5 dataset as in the notebook, check r = 0.31 (CI 0.00-0.57, n = 40),
then the partial correlation of income after controlling for distance from Dhaka and longitude."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
import statsmodels.api as sm
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
E = "/home/asif/PycharmProjects/Air_Quality_Analysis/data_preparation/Extra Data/"
series = read_bd_data_4_years()[1]                                                      # hourly, 55 zones (as in notebook)
average_reading = series.groupby(series.index.dayofyear).mean().mean()                  # notebook cell 37 (day-of-year climatology mean)
zone_income = pd.read_csv(E + "zone_income.csv", index_col="zone")
district_income = pd.read_csv(E + "district_income.csv", index_col="district", sep="\t+", engine="python")[["per_capita_income"]]
zmap = pd.merge(zone_income.reset_index(), district_income.reset_index(), on="per_capita_income", how="inner")[["zone", "district", "per_capita_income"]]
zmap = zmap.set_index("zone")                                                           # notebook cell 25 (merge on income value)
zone_not_district_city = ["Azimpur_Dhaka", "Paltan_Dhaka", "Tungi_Dhaka", "Bhairab Bazar_Dhaka", "Patiya_Chittagong", "Sandwip_Chittagong",
                          "Satkania_Chittagong", "Laksham_Chittagong", "Ishurdi_Rajshahi", "Sarishabari_Mymensingh", "Raipur_Chittagong",
                          "Ramganj_Chittagong", "Nagarpur_Dhaka"]                        # notebook cell 34
df = zmap.drop(zone_not_district_city).join(average_reading.rename("pm"))
print("zones after dropping non-district-city zones:", len(df), "(notebook cell 54: 42)")
def r_ci(x, y):
    r, p = stats.pearsonr(x, y); n = len(x); z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), p, n
r, lo, hi, p, n = r_ci(df.per_capita_income, df.pm)
print(f"n={n} (incl. Dhaka & Chittagong): r = {r:.3f} [{lo:.2f}, {hi:.2f}], p = {p:.3f}")
d40 = df.drop(["Dhaka_Dhaka", "Chittagong_Chittagong"])
r, lo, hi, p, n = r_ci(d40.per_capita_income, d40.pm)
print(f"n={n} (Dhaka & Chittagong excluded, as in paper): r = {r:.3f} [{lo:.2f}, {hi:.2f}], p = {p:.3f}   <- paper: 0.31 [0.00, 0.57], p = 0.05")
print("mean income of the 40 districts (paper threshold 3,968): %.1f; higher-income n = %d" % (d40.per_capita_income.mean(), (d40.per_capita_income > d40.per_capita_income.mean()).sum()))
# alternative zone-mean definitions, to show sensitivity
r2 = stats.pearsonr(d40.per_capita_income, series.mean()[d40.index])[0]
print(f"same 40 zones with plain hourly mean instead of day-of-year mean: r = {r2:.3f}")
# location controls (zone coordinates from geography_check.csv; distance to Dhaka zone)
g = pd.read_csv("/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts/geography_check.csv", index_col=0)
x = d40.join(g[["lat", "lon", "dist_dhaka_km"]])
def resid(v, Z): return sm.OLS(v, sm.add_constant(Z)).fit().resid
Z = x[["dist_dhaka_km", "lon"]]
pr, pp = stats.pearsonr(resid(x.per_capita_income, Z), resid(x.pm, Z))
dfree = len(x) - 2 - 2; t = pr * np.sqrt(dfree / (1 - pr ** 2)); pp = 2 * stats.t.sf(abs(t), dfree)
print(f"partial r(income, PM | distance from Dhaka, longitude) = {pr:.3f}, p = {pp:.3f} (df = {dfree})")
m = sm.OLS(x.pm, sm.add_constant(x[["per_capita_income", "dist_dhaka_km", "lon"]])).fit()
print("OLS pm ~ income + dist + lon: coef income = %.5f per BDT (p = %.3f); R2 = %.3f" % (m.params["per_capita_income"], m.pvalues["per_capita_income"], m.rsquared))
print("location model alone R2 = %.3f" % sm.OLS(x.pm, sm.add_constant(Z)).fit().rsquared)
print("r(income, distance from Dhaka) = %.2f; r(income, longitude) = %.2f" % (x.per_capita_income.corr(x.dist_dhaka_km), x.per_capita_income.corr(x.lon)))
hi_ = x.per_capita_income > x.per_capita_income.mean()
print("group means: higher-income dist %.0f km, lon %.2f | lower-income dist %.0f km, lon %.2f" %
      (x[hi_].dist_dhaka_km.mean(), x[hi_].lon.mean(), x[~hi_].dist_dhaka_km.mean(), x[~hi_].lon.mean()))
