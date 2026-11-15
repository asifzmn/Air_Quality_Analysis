"""Do forest cover, distance to the Bay of Bengal, and west-east position explain the zone-to-zone differences
in 4-year mean PM2.5 (55 zones, 2018-2021, local time, observed hours)?
Forest: LGED forest and natural parks polygons (HDX/OCHA, bgd_phy_forestnaturalparks_lged.zip).
Coast: Bangladesh outline (bdBounds.geojson) minus segments within 15 km of India/Myanmar (Natural Earth lowres).
Output: geography_check.csv (per zone) and printed correlations."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis")
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats

r = read_bd_data_4_years(); meta, s = r[0], r[1].copy()
s.index = s.index + pd.Timedelta(hours=6); s = s["2018":"2021"]
z = pd.DataFrame({"mean": s.mean(), "lat": meta.loc[s.columns, "Latitude"], "lon": meta.loc[s.columns, "Longitude"]})
UTM = 32646
pts = gpd.GeoSeries([Point(xy) for xy in zip(z.lon, z.lat)], index=z.index, crs=4326).to_crs(UTM)

forest = gpd.read_file("zip:///home/asif/Desktop/AQ Resources/bgd_phy_forestnaturalparks_lged.zip").to_crs(UTM)
forest = forest[forest.geometry.notna()]
fu = forest.geometry.buffer(0).unary_union
for km in (10, 25, 50):
    z[f"forest_km2_{km}"] = [pts[i].buffer(km * 1000).intersection(fu).area / 1e6 for i in z.index]
z["dist_forest_km"] = [pts[i].distance(fu) / 1000 for i in z.index]

ne = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
neigh = ne[ne.name.isin(["India", "Myanmar"])].to_crs(UTM).geometry.unary_union.buffer(15000)
bd = gpd.read_file("/home/asif/Desktop/Work/Dataset/Bangladesh/bdBounds.geojson").to_crs(UTM).geometry.unary_union
coast = bd.boundary.difference(neigh)
z["dist_coast_km"] = [pts[i].distance(coast) / 1000 for i in z.index]
dhaka = pts["Dhaka_Dhaka"]
z["dist_dhaka_km"] = [pts[i].distance(dhaka) / 1000 for i in z.index]
z.round(3).to_csv(HERE + "/geography_check.csv")

def ci(r, n):
    f = np.arctanh(r); se = 1 / np.sqrt(n - 3); return np.tanh(f - 1.96 * se), np.tanh(f + 1.96 * se)
def partial(x, y, c):
    X = np.column_stack([np.ones(len(c)), c])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)
n = len(z); print("n zones", n, "coast length km", round(coast.length / 1000))
for v in ["forest_km2_10", "forest_km2_25", "forest_km2_50", "dist_forest_km", "dist_coast_km", "lon", "lat", "dist_dhaka_km"]:
    pr, pp = stats.pearsonr(z[v], z["mean"]); sr, sp = stats.spearmanr(z[v], z["mean"]); lo, hi = ci(pr, n)
    pc, ppc = partial(z[v].values, z["mean"].values, z[["dist_dhaka_km"]].values)
    print(f"{v:16s} r={pr:+.2f} [{lo:+.2f},{hi:+.2f}] p={pp:.3g} | rho={sr:+.2f} p={sp:.3g} | partial|distDhaka r={pc:+.2f} p={ppc:.3g}")
# multiple regression (standardized)
X = z[["forest_km2_25", "dist_coast_km", "lon", "dist_dhaka_km"]]; Xs = (X - X.mean()) / X.std(); y = (z["mean"] - z["mean"].mean()) / z["mean"].std()
A = np.column_stack([np.ones(n), Xs.values]); b, *_ = np.linalg.lstsq(A, y.values, rcond=None)
res = y.values - A @ b; R2 = 1 - (res ** 2).sum() / ((y.values - y.mean()) ** 2).sum()
s2 = (res ** 2).sum() / (n - A.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
for name, bi, sei in zip(["const"] + list(X.columns), b, se):
    print(f"  beta {name:15s} {bi:+.2f} (t={bi/sei:+.2f}, p={2*stats.t.sf(abs(bi/sei), n-5):.3g})")
print("  R2", round(R2, 3))
near = z["forest_km2_25"] > 0
print("zones with forest within 25 km:", near.sum(), "mean", round(z.loc[near, "mean"].mean(), 1), "vs", round(z.loc[~near, "mean"].mean(), 1),
      "Welch p", round(stats.ttest_ind(z.loc[near, "mean"], z.loc[~near, "mean"], equal_var=False).pvalue, 3))
print(z.sort_values("mean")[["mean", "forest_km2_25", "dist_coast_km", "lon", "dist_dhaka_km"]].round(1).to_string())

# ---- table for the SI: Pearson r [95% CI], Spearman rho, partial r controlling for distance to Dhaka and longitude
rows = []
for v, lab in [("dist_dhaka_km", "Distance to Dhaka (km)"), ("lon", "Longitude (deg E)"), ("forest_km2_10", "Forest area within 10 km"),
               ("forest_km2_25", "Forest area within 25 km"), ("forest_km2_50", "Forest area within 50 km"),
               ("dist_forest_km", "Distance to nearest forest"), ("dist_coast_km", "Distance to coast")]:
    pr, pp = stats.pearsonr(z[v], z["mean"]); lo, hi = ci(pr, n); sr, _ = stats.spearmanr(z[v], z["mean"])
    ctrl = [c for c in ["dist_dhaka_km", "lon"] if c != v]
    pc, ppc = partial(z[v].values, z["mean"].values, z[ctrl].values)
    rows.append([lab, pr, lo, hi, pp, sr, pc, ppc])
T = pd.DataFrame(rows, columns=["var", "r", "lo", "hi", "p", "rho", "partial", "p_partial"])
T.round(3).to_csv(HERE + "/geography_check_table.csv", index=False); print(T.round(2).to_string())
