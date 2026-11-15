# Processed PM2.5 series used in the paper

Hourly PM2.5 (µg/m³) derived from Berkeley Earth's interpolated estimates, after the preparation described in the paper (complete hourly index, duplicate timestamps removed, no imputation). **Timestamps are UTC**; add 6 h for Bangladesh local time (the paper converts to local time for all descriptive and multi-source analyses).

| File | Content |
|---|---|
| `bd_zone_series_2018_2021_utc.csv` | 55 Bangladeshi zones, hourly, 2018–2021 (columns `Zone_Division`) |
| `bd_zone_metadata.csv` | Zone name, division, population, latitude, longitude |
| `bd_division_series_2018_2021_utc.csv` | Hourly means of the zones in each of the 8 divisions, 2018–2021 |
| `region_series_bd_india_2017_2022_utc.csv` | Hourly means for the 8 Bangladeshi divisions and 10 Indian states, 2017–2022 (cross-border analysis) |
| `region_metadata_bd_india.csv` | Region, country, population, mean coordinates, number of zones |

Division labels follow the Berkeley Earth metadata; Saidpur is listed under Rajshahi although it lies in Rangpur division (see the paper, Table 7 note).

**Source and licence:** Berkeley Earth air quality data (http://berkeleyearth.lbl.gov/air-quality/), licensed under CC BY-NC 4.0. These derived files are shared under the same terms: non-commercial use with attribution to Berkeley Earth and to the paper.
