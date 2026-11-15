# Code for the PLOS ONE paper "Understanding and forecasting the pattern of air pollution in Bangladesh"

Asif Zaman and Moinul Zaber. This folder holds the forecasting notebooks and the scripts that reproduce the figures, tables and verification numbers of the revised manuscript (PONE-D-25-18374). The data-preparation code lives in the rest of this repository (`data_preparation/`, `exploration/`, `meteorology/`, `paths.py`); the scripts here import it, so the repository root must be on `sys.path`.

## Contents

| Path | What it is |
|---|---|
| `notebooks/AQ_forecast_model.ipynb` | Google Colab notebook: forecasting dataset, model search, evaluation, SHAP, architecture diagrams, loss curves |
| `notebooks/AQ_NASA_FIRMS.ipynb` | Google Colab notebook: NASA FIRMS fire download and aggregation |
| `figure_scripts/` | Scripts that produce the final figures and the numbers quoted in the text |
| `figure_scripts/final_review/` | Checks added in the final revision: forecast significance and PM2.5-only ablation, cross-border distance decay, income vs location |
| `data/` | Processed hourly PM2.5 series (55 zones, 8 divisions, 10 Indian states; UTC) and metadata, see `data/README.md` |
| `validation_us_embassy/scripts/` | Berkeley Earth vs U.S. Embassy Dhaka validation and a stdlib `.xlsx` to `.csv` converter |

## Scripts and outputs

| Script | Manuscript item |
|---|---|
| `fig2_flowchart.py` | Fig 2 (data collection and preparation) |
| `fig_overview_calendar.py` | Fig 3 |
| `fig3_fig8_fig25.py` | Figs 4, 8, 25 |
| `figs_new_final.py` | Figs 5, 11, 14, 21, 23 |
| `fig6_diurnal.py` | Fig 6 and night/day tests |
| `fig7_meteo_lags.py` | Fig 7, Table 9 |
| `fig9_fig22_mobility.py` | Figs 9, 22 (supersedes `fig9_mobility_lag.py`, `fig22_desea.py`) |
| `fig15_fig20.py` | Fig 16 (imputation benchmark) |
| `fire_rerun_continuous.py`, `fig20_fire_corrected.py` | Fig 20: fire–PM2.5 correlations on a continuous daily index (fire-free days set to zero) |
| `fig_stacked_bar.py` | Fig 17 |
| `fig7_fig17.py` | Fig 18 |
| `fig_si_meteo_heatmaps.py` | Fig 19 |
| `geography_check.py` | Geographic correlates of zone means (SI Table 8) |
| `check_pblh.py`, `check_naaqs_vif.py`, `check_imputation_gaps.py`, `check_timezones.py` | PBLH lags, NAAQS exceedance and VIF (Table 2), imputation sensitivity (Table 6), time-zone checks |
| `forecast_reproduce_persistence.py` | Rebuilds the forecasting dataset locally, Bayesian Ridge and persistence (Table 4) |
| `final_review/m1_m2_forecast_tests.py` | Block bootstrap, Diebold–Mariano tests, PM2.5-only model |
| `medians.py` | Mobility differences (5.2 and 1.9 µg/m³) |
| `plos_style.py` | Shared figure style |

Figures 1, 10, 13, 15, 24 and 26 come from the main repository (GeoPandas maps) or the Colab notebook (SHAP, architectures, Fig 27 loss curves).

## How to run

```bash
# Python 3.6 virtual environment of this repository
export PYTHONPATH=/path/to/Air_Quality_Analysis
cd paper_plos_one/figure_scripts
python fig_overview_calendar.py        # same pattern for every fig*/check* script
```

The scripts contain absolute paths for the data folder and some outputs (`/home/...`); edit them to your locations. `paths.py` sets the data root. Secrets (e.g. a Mapbox token for the old interactive maps) are read from a git-ignored `.env` file; none are needed for the paper's figures. The notebooks run in Google Colab after the prepared data are placed where their first cells expect them.

## Data sources and licences

Raw data are not redistributed here; obtain them from the providers. The processed Berkeley Earth series in `data/` are shared under CC BY-NC 4.0.

| Data | Licence / terms |
|---|---|
| Berkeley Earth hourly PM2.5 | CC BY-NC 4.0 (non-commercial, attribution) |
| Meteoblue history archive (2019) | Meteoblue terms; not redistributable |
| Weather Underground public history (Dhaka airport) | The Weather Company terms; non-commercial research use |
| ERA5 boundary-layer height (Copernicus Climate Data Store) | Copernicus licence; "Contains modified Copernicus Climate Change Service information" |
| NASA FIRMS (VIIRS NOAA-20) | NASA open data; acknowledge LANCE/FIRMS |
| Google COVID-19 Community Mobility Reports | Google terms (public reports) |
| Bangladesh Bureau of Statistics district indicators | Public government statistics |
| U.S. Embassy Dhaka PM2.5 (U.S. EPA; Sarwar et al. 2023), doi:10.23719/1528454 | U.S. government work, public |

## Notes (as stated in the paper)

- Berkeley Earth timestamps are UTC and are converted to local time (UTC+6) for all descriptive and multi-source analyses; the forecasting pipeline uses UTC days.
- The validation against the U.S. Embassy monitor is probably not independent (the monitor may feed Berkeley Earth), so r = 0.93 is an upper bound.
- Forecasting: scalers fitted on the full period; contiguous KFold for the grid search; validation and test days alternate; gains over persistence at days 4–7 are nominally significant only on day 6; a PM2.5-only model is as accurate as the full model; ordinary least squares, k-nearest neighbors and a decision tree were also tested and are not reported in the tables.
- SHAP explains auxiliary models from an earlier run; architecture diagrams show the development configuration.
- MAPE is not reported (an earlier implementation was invalid).
- Saidpur is assigned to Rajshahi division, as in the Berkeley Earth metadata, although it lies in Rangpur division; reassigning it changes the Rajshahi and Rangpur medians by 0.62 and 0.10 µg/m³.
