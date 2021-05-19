import pandas as pd
import plotly.graph_objects as go
from data_preparation import get_metadata, get_series


def GroupedBox(x):
    fig = go.Figure()

    colorPal = ['#4AA02C', '#6AA121', '#7D0552', '#7D0500']

    for year in pd.DatetimeIndex(x.index).year.unique():
        fig.add_trace(go.Box(
            y=x[str(year)],
            x=pd.DatetimeIndex(x.index).month_name(),
            name=year,
            marker_color=colorPal[year - 2017]
        ))

    fig.update_layout(
        title=x.name,
        yaxis_title='PM2.5 Concentration',
        boxmode='group',
        yaxis=dict(
            range=[0, 250]),
        legend_orientation="h")

    fig.show()


if __name__ == '__main__':
    metadata, series = get_metadata(), get_series()
    print(metadata)

    series[['Kushtia']].apply(GroupedBox)
    # df.resample('6H').mean().apply(GroupedBox)
