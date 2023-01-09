import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_preparation.spatio_temporal_filtering import read_bd_data


def plotly_marker_scatter_mapbox_with_arrow(geo_points):
    # mapbox_access_token = open(".mapbox_token").read()
    mapbox_access_token = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA'

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=geo_points.Latitude,
        lon=geo_points.Longitude,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15, color= 'black'
        )
    ))

    fig.add_trace(go.Scattermapbox(
        lat=geo_points.Latitude,
        lon=geo_points.Longitude,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=12, color=geo_points.color
        ),
        text=geo_points.index,
    ))


    l = 1.1/3  # the arrow length
    widh = 0.035  # 2*widh is the width of the arrow base as triangle

    # A = np.array([locations['lon'][5], locations['lat'][5]])
    A = np.array([geo_points.Longitude.iloc[5], geo_points.Latitude.iloc[5]])
    # B = np.array([locations['lon'][0], locations['lat'][0]])
    B = np.array([geo_points.Longitude.iloc[0], geo_points.Latitude.iloc[0]])
    v = B - A
    w = v / np.linalg.norm(v)
    u = np.array([-v[1], v[0]])  # u orthogonal on  w

    P = B - l * w
    S = P - widh * u
    T = P + widh * u

    fig.add_trace(go.Scattermapbox(lon=[S[0], T[0], B[0], S[0]],
                                lat=[S[1], T[1], B[1], S[1]],
                                mode='lines',
                                fill='toself',
                                fillcolor='blue',
                                line_color='blue'))

    fig.add_trace(go.Scattermapbox(lon=[geo_points.Longitude.iloc[0], geo_points.Longitude.iloc[5]],
                                lat=[geo_points.Latitude.iloc[0], geo_points.Latitude.iloc[5]],
                                mode='lines',
                                fill='toself',
                                fillcolor='blue',
                                line_color='blue'))

    fig.update_layout(
        height=900,
        width=900,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=23.81,
                lon=90.41
            ),
            pitch=0,
            zoom=6
        )
    )

    fig.show()

metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data()
metadata.color = 'grey'
plotly_marker_scatter_mapbox_with_arrow(metadata)