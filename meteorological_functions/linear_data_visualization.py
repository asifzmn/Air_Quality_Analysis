from plotly.subplots import make_subplots

from data_preparation import get_diurnal_period
from meteorological_functions.meteoblue_data_preparation import get_factor_data
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def meteo_time_series(meteo_data, meteorological_variable):
    fig = go.Figure()
    print(meteorological_variable.factor_list)
    meteo_data = meteo_data.loc[:, meteorological_variable.factor_list]
    # wind_dir_names = wind_direction_factors(wind_direction_factor)[0]
    wind_dir_names = []
    for i, factor in enumerate(np.setdiff1d(meteo_data['factor'].values, wind_dir_names)):
        fig.add_trace(
            go.Scatter(x=pd.Series(meteo_data['time'].values), y=meteo_data.loc[:, factor], name=factor,
                       marker_color=meteorological_variable.color_list[i]))  # line_color='deepskyblue','dimgray'

    fig.update_traces(mode='lines+markers', marker=dict(line_width=0, symbol='circle', size=3))
    fig.update_layout(title=meteorological_variable.name, font_size=16, legend_font_size=16, template="ggplot2",
                      xaxis_title="Time", yaxis_title=meteorological_variable.unit, legend_orientation='h')
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


def meteo_time_series_subplots(data, meteorological_variable_type_list_linear):
    sub_plot_rows = len(meteorological_variable_type_list_linear)
    fig = make_subplots(rows=sub_plot_rows, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for j, meteorological_variable in enumerate(meteorological_variable_type_list_linear):
        meteo_data = data.mean(axis=0).resample(time="M").mean().loc[:, meteorological_variable.factor_list]
        for i, factor in enumerate(meteo_data['factor'].values):
            fig.add_trace(
                go.Scatter(x=pd.Series(meteo_data['time'].values), y=meteo_data.loc[:, factor], name=factor,
                           marker_color=meteorological_variable.color_list[i]), row=j + 1, col=1)
        fig.update_yaxes(title_text=f"{meteorological_variable.name} ({meteorological_variable.unit})", row=j + 1,
                         col=1)

    fig.update_layout(height=800, width=1600,
                      title_text="Stacked Subplots with Shared X-Axes", legend_orientation='h', font_size=15)

    fig.show()


def meteo_box_plot(meteo_data, time_series):
    factor = [['Surface Temperature', 'indianred'], ['Relative Humidity [2 m]', 'skyblue']][1]
    fig = go.Figure()
    for dis in meteo_data['district'].values: fig.add_trace(
        go.Box(y=meteo_data.loc[dis, :, factor[0]], name=dis, marker_color=factor[1]))
    fig.show()


def scatter_plot_factor_vs_pm_diurnal_seasonal(meteo_data, time_series):
    compare_factors = ['Temperature [2 m elevation corrected]', 'Relative Humidity [2 m]', 'Wind Speed [10 m]',
                       'Mean Sea Level Pressure [MSL]']
    compare_factors_rename = ['Temperature [Celcius]', 'Relative Humidity [Percentage]',
                              'Wind Speed [kilometer/second]', 'Mean Sea Level Pressure [hPa]']

    for factor, factor_rename in zip(compare_factors, compare_factors_rename):
        factor_meteo_data = get_factor_data(meteo_data, factor)
        if factor == 'Relative Humidity [2 m]':
            factor_meteo_data = factor_meteo_data * 100
        all_data = pd.concat((factor_meteo_data.stack(), time_series.stack(dropna=False)), axis=1).droplevel(1)
        all_data.columns = [factor_rename, "PM2.5 Reading"]
        all_data = all_data[all_data.index.month.isin([1, 2, 6, 7, 8, 12])]
        all_data['month'] = all_data.index.month
        all_data['season'] = all_data.apply(lambda x: 'summer' if x.month in (range(6, 9)) else 'winter', axis=1)

        all_data = all_data.join(get_diurnal_period()).sample(1310 * 3)
        # all_data = all_data.join(get_diurnal_period()['2019']).sample(1310 * 3)
        all_data["seasonal_diurnal"] = all_data['season'] + ' ' + all_data['diurnal_name']
        # color_map = {'summer day': 'light red','summer night': 'dark red'}
        color_map = {'winter day': '#82CAFF', 'winter night': '#151B54', 'summer day': '#E67451',
                     'summer night': '#551606'}

        # print(factor_meteo_data.stack().corr(time_series.stack(dropna=False)))
        fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename, color='seasonal_diurnal', opacity=0.45,
                         trendline='ols', color_discrete_map=color_map, template='seaborn')
        # fig = px.scatter(all_data, y="PM2.5 Reading", x=factor_rename,opacity=0.33,trendline='ols',
        # color_discrete_map=color_map)
        fig.update_layout(height=750, width=750, font=dict(size=18), legend_orientation='h', yaxis_range=[0, 175])
        fig.show()


def scatter_plot_factor_vs_pm(timeSeries, reading):
    # allData = reading.join(timeSeries)
    # allData = allData.resample('D').mean()
    allData = pd.concat((reading, timeSeries), axis=1)

    # print(allData.corr())

    meteo_factors = ['Temperature', 'Humidity', 'Wind Speed']

    for f in meteo_factors:
        fig = px.scatter(allData, y="Reading", x=f)
        fig.update_layout(height=750, width=750, font=dict(size=21))
        fig.show()

    # BoxPlotSeason(timeSeries)
    # BoxPlotHour(timeSeries)


def nominal_condition_stats(all_data):
    condition_group = (all_data.groupby('Condition').median()).loc[
        ['Haze', 'Fog', 'Rain', 'T-Storm', 'Drizzle', 'Thunder', 'Light Rain', 'Light Rain with Thunder',
         'Haze / Windy']]
    fig = go.Figure(data=[
        go.Bar(y=condition_group.Reading, x=condition_group.index, marker_color="grey")
    ])
    fig.show()

    direction_group = (all_data.groupby('Wind').median())

    fig = go.Figure(data=[
        go.Bar(y=direction_group.Reading, x=direction_group.index, marker_color="grey")
    ])
    fig.show()
