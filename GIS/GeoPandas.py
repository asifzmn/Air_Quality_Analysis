import json
import math
import pandas as pd
import geojsoncontour
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from matplotlib import cm
from matplotlib.lines import Line2D
from plotly.offline import iplot
from data_preparation import get_metadata, get_category_info, get_series, aq_directory
from scipy.interpolate import griddata
from numpy import linspace
import plotly.graph_objects as go


def Distance(dis1, dis2):
    metaFrame = get_metadata()
    origin = metaFrame.loc[dis1]['Latitude'], metaFrame.loc[dis1]['Longitude']
    dest = metaFrame.loc[dis2]['Latitude'], metaFrame.loc[dis2]['Longitude']

    # x = geodesic(origin, dest).meters,geodesic(origin, dest).kilometers,geodesic(origin, dest).miles
    return geodesic(origin, dest).kilometers


def compass_bearing():
    pass


def angleFromCoordinate(dis1, dis2):
    metaFrame = get_metadata()
    lat1, long1, lat2, long2 = metaFrame.loc[dis1]['Latitude'], metaFrame.loc[dis1]['Longitude'], metaFrame.loc[dis2][
        'Latitude'], metaFrame.loc[dis2]['Longitude']
    lat1, lat2, long1, long2 = math.radians(lat1), math.radians(lat2), math.radians(long1), math.radians(long2)

    dLon = (long2 - long1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)

    return (math.degrees(bearing) + 360) % 360


def draw_pie(ax, ratios, X=0, Y=0, size=1000):
    xy, start = [], 0
    colors = ["BURLYWOOD", "DARKBLUE"]

    for ratio in ratios:
        pieSegemnt = np.linspace(2 * math.pi * start, 2 * math.pi * (start + ratio), 30)
        x = [0] + np.cos(pieSegemnt).tolist()
        y = [0] + np.sin(pieSegemnt).tolist()
        print(x)
        print(y)
        xy.append(list(zip(x, y)))
        start += ratio

    for i, xyi in enumerate(xy):
        # print(X,Y)
        ax.scatter([X], [Y], marker=(xyi), s=size, facecolor=colors[i])


def Pie(ax, ratiodata):
    metaFrame = get_metadata()
    for i, txt in enumerate(metaFrame.index): draw_pie(ax, ratiodata[0], metaFrame.iloc[i]['Longitude'],
                                                       metaFrame.iloc[i]['Latitude'], size=800)

    lines = [Line2D([0], [0], color=c, linewidth=5, linestyle='-') for c in ["BURLYWOOD", "DARKBLUE"]]
    labels = ['Good or Moderate', 'Unhealthy or Hazardous']
    plt.legend(lines, labels, loc='lower left', prop={'size': 15})


def Arrows(ax, vec):
    colorGen = cm.get_cmap('Purples', 256)
    # newcolors = colorGen(np.linspace(0, 256, 7))
    # pal = colorGen(np.linspace(0, 1, 15))[4:11][::-1]
    pal = colorGen(np.linspace(0, 1, 5))[1:5]
    # pal = []

    metaFrame = get_metadata()
    for x in range(len(vec)):
        for y in range(len(vec)):
            # if vec[x][y] == 0.0: continue
            if vec.iloc[x][y] <= 0: continue
            # c = 'b' if vec[x][y] > 0 else 'r'
            ax.arrow(metaFrame.iloc[x]['Longitude'], metaFrame.iloc[x]['Latitude'],
                     metaFrame.iloc[y]['Longitude'] - metaFrame.iloc[x]['Longitude'],
                     metaFrame.iloc[y]['Latitude'] - metaFrame.iloc[x]['Latitude'],  # width=.01 * vec.iloc[x][y],
                     # metaFrame.iloc[y]['Latitude'] - metaFrame.iloc[x]['Latitude'], width=.001 * 3,
                     head_width=.075, head_length=0.1, length_includes_head=True, zorder=1,
                     color=pal[int(vec.iloc[x][y])],
                     # head_width=.045, head_length=0.1, length_includes_head=True, zorder=1, color=pal[-1],
                     ls='-')
            # head_width=.033, head_length=0.081, length_includes_head=True, zorder=0, color = 'grey',ls = '-')


def setMap(x=2):
    plt.rcParams['figure.figsize'] = (8 * x, 10 * x)
    # df_admin = gpd.read_file('/media/az/Study/Air Analysis/Maps/districts.geojson')
    df_admin = gpd.read_file(aq_directory + 'Maps/districts.geojson')
    return df_admin.plot(color='#E5E4E2', edgecolor='#837E7C')


def MapScatter(ax, data=None):
    colorScale, categoryName, AQScale = get_category_info()

    # data = np.arange(15, 15 + 22 * 9, 9)
    # color_ton = [colorScale[val] for val in np.digitize(data, AQScale[1:-1])]

    # colorScale = ['#C38EC7','#E42217','#FFD801','#5EFB6E','#5CB3FF','#34282C']
    # color_ton = [colorScale[val] for val in np.digitize(data, AQScale[1:-1])]

    # ax.scatter(x=data.Longitude.astype('float64'), y=data.Latitude.astype('float64'), zorder=1, alpha=1,
    #            c=data.color, s=150, marker='H', edgecolor='#3D3C3A', linewidth=1)

    for idx, row in data.iterrows():
        marker_size, marker_color, = 150, '#566D7E'
        if idx in ['Kishorganj', 'Nagarpur', 'Dhaka']: marker_size, marker_color = 450, '#5b567e'
        ax.scatter(x=row.Longitude, y=row.Latitude, zorder=1, alpha=1,
                   c=row.color, s=marker_size, marker=row.symbol, edgecolor='#3D3C3A', linewidth=1)


def MapLegend(ax, legendData):
    lines = [Line2D([0], [0], color=c, linewidth=5, linestyle='-') for c in legendData[1].color]
    # lines = [Line2D([], [], color='#566D7E', marker=s, linestyle='None', markersize=15) for s in legendData[1].symbol]
    ax.legend(lines, legendData[1].category, loc='lower left', prop={'size': 15}, title=legendData[0])


def MapAnnotate(ax, data):
    for idx, row in (data.iterrows()): ax.annotate(idx, (row.Longitude - len(idx) * .015, row.Latitude - .1),
                                                   fontsize=15)


def mapPlot(data, legendData, save=None):
    ax = setMap()
    MapScatter(ax, data)
    MapAnnotate(ax, data)
    MapLegend(ax, legendData)
    plt.tight_layout()
    plt.show()
    if save is not None:
        plt.savefig(save + '.png', dpi=300)
        plt.clf()


def mapArrow(data, mat, times, save=None):
    ax = setMap()
    MapScatter(ax, data)
    MapAnnotate(ax, data)
    Arrows(ax, mat)
    # plt.title('From '+str(times[0])+' to '+str(times[-1]))
    plt.title(times)
    plt.tight_layout()
    plt.show()

    if save is not None:
        plt.savefig(save + '.png', dpi=300)
        plt.clf()


if __name__ == '__main__':
    # meteoData = xr.open_dataset('meteoData.nc')['meteo']

    metaFrame, df = get_metadata(), get_series()['2018-01':'2021-12'].resample('H').mean()
    # metaFrame, df = LoadMetadata(), getFactorData(meteoData, 'Temperature [2 m]')

    from data_preparation import get_metadata, get_series, clip_missing_prone_values, \
        prepare_region_and_country_series

    series_with_heavy_missing, metadata_with_heavy_missing = get_series(), get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_region_and_country_series(series,
                                                                                                         metadata)

    # df, metaFrame= region_series.groupby(country_series.index.month).mean() , metadata_region
    # df, metaFrame= region_series.groupby(country_series.index.year).median() , metadata_region

    exit()

    ax = setMap()
    # MapScatter(ax)
    # Pie(ax,[[.5,.5]])
    # Arrows(ax,mat)
    # MapAnnotate(ax)
    plt.show()
    exit()
