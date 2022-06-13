import pandas as pd
import plotly.graph_objects as go
from data_preparation import get_metadata, get_series, clip_missing_prone_values, prepare_division_and_country_series

bd_lockdown_dates = '2020-03-26','2020-05-16','2021-04-05','2021-07-14','2021-07-23','2021-08-10'

def grouped_box(x):
    fig = go.Figure()

    color_pal = ['#4AA02C', '#6AA121', '#7D0552', '#7D0500', '#2471A3']

    # for year in pd.DatetimeIndex(x.index).year.unique():
    for year in x.index.year.unique():
        fig.add_trace(go.Box(
            # y=x[str(year)],
            y=x[str(year)],
            x=pd.DatetimeIndex(x.index).month_name(),
            name=year,
            marker_color=color_pal[year - 2018]
        ))

    fig.update_layout(
        title=x.name,
        yaxis_title='PM2.5 Concentration',
        boxmode='group',
        yaxis=dict(
            range=[0, 180]),
        legend_orientation="h")

    fig.show()


if __name__ == '__main__':
    series_with_heavy_missing, metadata_with_heavy_missing = get_series()['2018':], get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_division_and_country_series(series,
                                                                                                               metadata)

    print(country_series)
    # series[['Kushtia','Dhaka','Delhi']].apply(grouped_box)
    # division_series[['Dhaka', 'NCT']].apply(grouped_box)
    country_series.apply(grouped_box)
    # series.apply(grouped_box)