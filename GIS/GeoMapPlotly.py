import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_preparation import get_category_info, get_metadata, get_series

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


if __name__ == '__main__':
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
