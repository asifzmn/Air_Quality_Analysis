import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import json
import math
import pandas as pd
import geojsoncontour
import geopandas as gpd
from scipy.interpolate import griddata
from numpy import linspace
import matplotlib.pyplot as plt


from data_preparation import get_category_info, get_metadata, get_series, read_all_granularity_data

access_token = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA'


def plotly_slider_mapbox(df, meta_data, timeformat, discrete=True):
    dfre = df.resample(timeformat[0]).mean()
    data, times = (dfre).stack().values, (dfre.index.strftime(timeformat[1]))

    meta_data = meta_data.reset_index()
    meta_data = pd.concat([meta_data] * len(times), ignore_index=True)

    if discrete:
        colorScale, categoryName, AQScale = get_category_info()
        meta_data['category'], meta_data['time'] = [categoryName[val] for val in
                                                    np.digitize(data, AQScale[1:-1])], np.repeat(times, df.shape[1])
        meta_data['zone'] = meta_data.index.values
        fig = px.scatter_mapbox(meta_data, lat="Latitude", lon="Longitude", hover_name='zone', animation_frame="time",
                                hover_data=["Zone"],
                                # zoom=6, height=750,width=750, color="category", color_discrete_sequence=colorScale)
                                zoom=6, height=750, width=750, color="category", color_discrete_sequence=colorScale)

    else:
        # colorRange,colorScale = [10,40],([(0, "yellow"), (0.5, "orange"), (1, "red")])
        colorRange, colorScale = [0, 1], ([(0, "white"), (0.5, "skyblue"), (1, "blue")])
        # colorRange,colorScale = [10,40],([(0, "yellow"), (0.5, "orange"), (1, "red")])

        print(meta_data.shape, data.shape)
        meta_data['category'] = data
        meta_data['time'] = np.repeat(times, df.shape[1])
        fig = px.scatter_mapbox(meta_data, lat="Latitude", lon="Longitude", animation_frame="time",
                                # hover_data=["Population"],
                                zoom=7.5, height=750, color="category", range_color=colorRange,
                                color_continuous_scale=colorScale)

    fig.update_traces(marker_size=18, marker_opacity=1)
    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(legend=dict(bordercolor='rgb(100,100,100)', borderwidth=2, x=0, y=0))
    fig.show()


def plotly_density_mapbox():
    quakes = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')
    print(quakes)
    fig = go.Figure(go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
                                     radius=10))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def heatmapgeoJson(meta_data, title):
    with open('/home/asif/Data/Dataset/Bangladesh/bdBounds.geojson') as file:
        bdBounds = json.load(file)

    rounding_num, correction_coeff, segments, regions = 0.015, 0.5, 500, 75
    meta_data["Longitude"] = np.round(meta_data["Longitude"] / rounding_num) * rounding_num
    meta_data["Latitude"] = np.round(meta_data["Latitude"] / (rounding_num * correction_coeff)) * (
            rounding_num * correction_coeff)

    z, y, x = meta_data['Value'], meta_data['Latitude'], meta_data['Longitude']
    xi, yi = linspace(x.min(), x.max(), segments), linspace(y.min(), y.max(), segments)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]))

    print(zi)
    step_size = math.ceil((np.nanmax(zi) - np.nanmin(zi)) / regions)
    cs = plt.contourf(xi, yi, zi, range(int(np.nanmin(zi)), int(np.nanmax(zi)) + step_size + 1, step_size))

    data_geojson = eval(geojsoncontour.contourf_to_geojson(contourf=cs, ndigits=10, ))

    data_geojson["features"] = [data_geojson["features"][i] for i in range(len(data_geojson["features"])) if
                                len(data_geojson["features"][i]['geometry']['coordinates']) > 0]

    arr_temp = np.ones([len(data_geojson["features"]), 2])

    for i in range(len(data_geojson["features"])):
        data_geojson["features"][i]['properties']["id"] = i
        arr_temp[i] = i, float(data_geojson["features"][i]["properties"]["title"].split('-')[0])

    df_contour = pd.DataFrame(arr_temp, columns=["Id", "Value"])
    center_coors = 23.6850, 90.3563  # BD
    # center_coors = 22.55, 88.36  # Three Country
    center_coors = 23.79, 84.89  # Two Country


    district = (go.Scattermapbox(
        mode="markers", showlegend=False,
        lon=meta_data['Longitude'], lat=meta_data['Latitude'], text=meta_data.index,
        marker=go.scattermapbox.Marker(size=6, color='#43BFC7')
    ))

    districtBorder = (go.Scattermapbox(
        mode="markers", showlegend=False,
        lon=meta_data['Longitude'], lat=meta_data['Latitude'],
        marker=go.scattermapbox.Marker(size=9, color='#504A4B')
    ))

    df_bd = pd.DataFrame({'Value': [0], 'Id': [19]})
    tracebd = go.Choroplethmapbox(
        geojson=bdBounds, featureidkey="properties.ID_0",
        locations=df_bd.Id, z=df_bd.Value,
        marker=dict(opacity=0.15), colorscale='greys', showscale=False
    )

    trace = go.Choroplethmapbox(
        geojson=data_geojson, z=df_contour.Value,
        locations=df_contour.Id, featureidkey="properties.id",
        marker=dict(opacity=0.15), marker_line_width=0,
        zmin=0, zmax=250,
        # zmin=25, zmax=35,
        # colorscale=[(0, '#46d246'), (0.05, '#46d246'), (0.05, '#ffff00'), (0.14, '#ffff00'),
        #             (0.14, '#ffa500'), (0.22, '#ffa500'), (0.22, '#ff0000'), (0.6, '#ff0000'),
        #             (.6, '#800080'), (1, '#800080')],  # Discreet

        colorscale=[(0, '#46d246'), (0.045, '#1b701b'), (0.055, '#ffff00'), (0.135, '#7f7f00'),
                    (0.145, '#ffa500'), (0.22, '#7f5200'), (0.25, '#ff0000'), (0.55, '#7f0000'),
                    (0.65, '#800080'), (1, '#400040')],  # Discreet with continuous in group
    )

    layout = go.Layout(
        title=title, title_x=0.4,
        width=1250,
        height = 1000,
        margin=dict(t=80, b=0, l=0, r=0),
        font=dict(color='dark grey', size=18),
        mapbox=dict(
            center=dict(lat=center_coors[0], lon=center_coors[1]),
            # zoom=6.5,
            zoom=5.5,
            style="carto-positron"
        )
    )

    figure = dict(data=[tracebd, trace, districtBorder, district], layout=layout)
    # print(figure)
    iplot(figure)


if __name__ == '__main__':
    metaFrame, df = get_metadata(), get_series()['2018-01':'2021-12'].resample('H').mean()
    metadata_all, series_all, metadata_region_all, region_series_all, metadata_country_all, country_series_all = read_all_granularity_data()
    metaFrame, df = metadata_region_all, region_series_all['2018-01':'2021-12'].resample('H').mean()

    for i, row in df.iloc[24 * 2 + 6:24 * 2 + 18].iterrows():
        # if row.isnull().any(): continue
        if row.isnull().any(): row = row.fillna(1)
        metadata = metaFrame.assign(Value=row.values)
        # print(row)
        # print(metadata)
        # title = (str(i.date()-pd.Timedelta(days=7))+" "+str(i.date()))
        # title = (str(i.date()))
        title = (str(i))
        heatmapgeoJson(metadata, title)
        # exit()

    metadata, series, = get_metadata(), get_series()
    df = pd.DataFrame(data=np.transpose(series.values), index=series.index,
                      columns=[d.replace(' ', '') for d in metadata.index.values]).apply(pd.to_numeric)
    df = df.fillna(df.rolling(1830, min_periods=1, ).mean())['2017':]
    plotly_slider_mapbox(df, metadata, ['M', '%Y %B'])
    # plotly_density_mapbox()
    exit()

    # dataVector = get_series()[:'2020-06-09']
    # metaFrame = get_metadata()
    # meteoData = get_all_meteo_data_()
    # time = meteoData.coords['time'].values
    # # data = meteoData.loc[:,:,'Surface Temperature'].values
    # data = meteoData.loc[:, :, 'Relative Humidity [2 m]'].values
    #
    # meteoVar = pd.DataFrame(data=np.transpose(data), index=time)
    # # print(meteoVar)
    # # print(dataVector)
    # # SliderMapCommon(meteoVar, metaFrame, ['12H', '%Y-%B-%D  %H:00:00'], discrete=False)
    # plotly_slider_mapbox(dataVector, metaFrame, ['12H', '%Y-%B-%D  %H:00:00'], discrete=False)
