import numpy as np
import plotly.graph_objs as go

def PlotlyRosePlotBasic(info=None):
    if info is None:
        info = [[[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5], '11-14 m/s', 'rgb(106,81,163)'],
                [[55.5, 50.0, 45.0, 35.0, 20.0, 22.5, 37.5, 55.0], '8-11 m/s', 'rgb(158,154,200)'],
                [[40.0, 30.0, 30.0, 35.0, 7.5, 7.5, 32.5, 40.0], '5-8 m/s', 'rgb(203,201,226)'],
                [[20.0, 7.5, 15.0, 22.5, 2.5, 2.5, 12.5, 22.5], '< 5 m/s', 'rgb(242,240,247)']
                ]

    fig = go.Figure()
    for [r, name, marker_color] in info:
        fig.add_trace(go.Barpolar(
            r=r,
            name=name,
            marker_color=marker_color
        ))

    fig.update_traces(
        text=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
    fig.update_layout(
        title='Wind Speed Distribution in Laurel, NE',
        font_size=16,
        legend_font_size=16,
        polar_radialaxis_ticksuffix='%',
        polar_angularaxis_rotation=90,
        template="plotly_dark",
        polar_angularaxis_direction="clockwise"
    )
    fig.show()


def plotly_rose_plot(info, color_pal, all_districts):
    fig = go.Figure()
    # for [r,name] in info:fig.add_trace(go.Barpolar(r=r,name=name,marker_color=colorPal))
    for infod in info:
        for [r, name], colorp in zip(infod, color_pal):
            fig.add_trace(go.Barpolar(r=r, name=name, marker_color=colorp))

    district_count, wind_states = info.shape[0], info.shape[1]
    states = np.full((district_count, district_count * wind_states), False, dtype=bool)
    # for i in range(district_count):
    #     states[i][wind_states * i:wind_states * (i + 1)] = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label=district, method="update",
                         args=[{"visible": state}, {"title": "Wind direction in " + district}])
                    for district, state in zip(all_districts, states)]),
                active=0
            )
        ])

    fig.update_traces(
        text=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
    fig.update_layout(
        title='Wind', font_size=16, legend_font_size=16, polar_radialaxis_ticksuffix='', polar_angularaxis_rotation=90,
        template="plotly_dark", polar_angularaxis_direction="clockwise"
    )
    fig.show()


def wind_direction_factors(wind_direction_factor):
    wind_dir_names = np.array(wind_direction_factor.factor_list)
    color_pal = np.array(['#ffffb1', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB', '#1E90FF'])
    # color_pal = np.array(['#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0'])
    return wind_dir_names, color_pal


def get_cardinal_direction(x, seg=16):
    direction = np.floor((x - (360 / seg) / 2) / (360 / seg)).astype('int')
    direction[direction == -1] = seg - 1
    return np.roll(np.hstack((np.bincount(direction), np.zeros(seg - max(direction) - 1))), 1)


def WindGraphTeamEstimate(meteoData, alldis):
    colorPal = np.array(['#ffffff'])
    directions = np.array([[[get_cardinal_direction(meteoData), 'Wind']]], dtype=object)
    # print(directions)
    plotly_rose_plot(directions, colorPal, alldis)


def wind_graph_multi_zone(meteo_data,wind_direction_factor):
    wind_dir_names, color_pal = wind_direction_factors(wind_direction_factor)
    alldis = meteo_data['district'].values
    directions = np.array(
        # [[[getSides(meteoData.loc[dis, '2020-03-29':'2020-03-31', factor].values), factor]
        # for factor in wind_dir_names]
        [[[get_cardinal_direction(meteo_data.loc[dis, :, factor].values), factor] for factor in wind_dir_names]
         for dis in alldis], dtype=object)
    plotly_rose_plot(directions, color_pal, alldis)


def monthly_rose_plot(meteo_data,wind_direction_factor):
    meteo_data_wind_direction = meteo_data.loc[:, :, wind_direction_factors(wind_direction_factor)[0]]
    for month, data in meteo_data_wind_direction.groupby('time.month'):
        print(data.shape)
        # wind_graph_multi_zone(data)
        WindGraphTeamEstimate(data.stack(z=("time", "district", "factor")).values, ['bd'])

# def WindGraph(x): PlotlyRosePlot([[getSides(x), 'Wind', '#3f65b1']])