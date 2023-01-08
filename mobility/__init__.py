import pandas as pd
import plotly.graph_objects as go
from data_preparation import get_metadata, get_series, clip_missing_prone_values, prepare_region_and_country_series
from exploration import grouped_box_month_year

bd_lockdown_dates = '2020-03-26', '2020-05-16', '2021-04-05', '2021-07-14', '2021-07-23', '2021-08-10'

if __name__ == '__main__':
    series_with_heavy_missing, metadata_with_heavy_missing = get_series()[:'2021'], get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_region_and_country_series(series,
                                                                                                         metadata)

    print(country_series)
    # series[['Kushtia','Dhaka','Delhi']].apply(grouped_box)
    # division_series[['Dhaka', 'NCT']].apply(grouped_box)
    country_series.apply(grouped_box_month_year)
    # series.apply(grouped_box)
