# import re
# import geojsoncontour
# from scipy.spatial.distance import cdist
# from pandas_profiling import ProfileReport
# from CoronaBeforeAfter import GroupedBox
# from GeoMapPlotly import SliderMapCommon
# from Correlation_Measures import *

from collections import Counter
from sklearn.cluster import KMeans
from itertools import combinations
from plotly.subplots import make_subplots
from GIS.GeoPandas import mapArrow, mapPlot
from covid_mobility import grouped_box
from cross_correlation import CrossCorrelation
from data_exporting import latex_custom_table_format, paper_comparison, missing_data_fraction
# from meteorological_functions import get_factor_data, get_cardinal_direction, plotly_rose_plot
from meteorological_functions import get_factor_data
from related.GeoMapMatplotLib import MapPlotting
from visualization import *
import plotly.graph_objects as go
import xarray as xr
import more_itertools

month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']

save = '/home/az/Pictures/'
colorScale, categoryName, AQScale = get_category_info()


def diurnality(x):
    if (x >= 6) and (x < 12):
        return 'Morning'
    elif (x >= 12) and (x < 17):
        return 'Afternoon'
    elif (x >= 17) and (x < 20):
        return 'Evening'
    else:
        return 'Night'


def city_analysis(s):
    print(s.mean())
    # print(s.resample('M').mean().values)

    # s = pd.concat([s,s.index.to_series().dt.hour],axis=1)
    # s['index'] = s['index'].apply(diurnally)
    # s.columns = ['reading','daytime']
    # print(s.groupby('daytime').mean())


def MakeProfileReport(timeseries, name='AirQuality'):
    prof = MakeProfileReport(timeseries, minimal=False, title=name)
    prof.to_file(output_file=name + '.html')


def extreme_correlation(df):
    corrDistrict = np.corrcoef(np.transpose(df.to_numpy()))
    correlatedDistricts = (np.transpose(np.where(corrDistrict > 0.999)))
    identity = np.transpose([np.arange(len(df)), np.arange(len(df))])
    closeIndexReal = (sorted(list(set(map(tuple, correlatedDistricts)).difference(set(map(tuple, identity))))))
    print(np.transpose(
        [df.columns.values[np.transpose(closeIndexReal)[0]], df.columns.values[np.transpose(closeIndexReal)[1]]]))


def heat_map_driver(data, vmin, vmax, title, cmap):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(data, mask=mask, cmap=cmap, square=True, linewidths=.5, vmin=vmin,
                         vmax=vmax, xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title)
    plt.show()


def triangular_heatmap_correlation(title, df):
    plt.figure(figsize=(6, 6))
    heat_map_driver(df.corr(), 0, 1, str(title.year) + '/' + str(title.month), 'Purples')


def kde_zone_pair(df):
    for comb in list(combinations(range(len(df.columns.values)), 2)):
        with sns.axes_style('white'):
            sns.jointplot(df[df.columns.values[comb[0]]], df[df.columns.values[comb[1]]],
                          data=df, kind='kde')
        plt.show()


def bucketing(reading, bins): return reading.apply(
    lambda x: pd.cut(x, bins).value_counts().sort_index()).apply(
    lambda x: x / x.sum())


def ratio_map_plotting(reading, time_stamp):
    norm = bucketing(reading, [55.5, 500])
    MapPlotting(meta_data, timeseries.mean().values, ratiodata=norm, title=str(time_stamp.year))


def missing_data_info(df):
    print((df.isna().sum()))
    print(df.isnull().any(axis=1).sum())  # any null reading of a district for a time stamp
    print(df.isnull().all(axis=1).sum())  # all null reading of a district for a time stamp

    # HeatMapDriver(dfm.corr(), -1, 1, '', 'RdBu')


def shifted_series(df, dis, lag_range, offset, rs):
    plt.figure(figsize=(9, 3))
    df[dis[0]].plot(linestyle='-', linewidth=1, c="Blue")
    df[dis[1]].plot(linestyle='-', linewidth=1, c="Red")
    df[dis[1]].shift(offset).plot(linestyle=':', linewidth=3, c="palevioletred")
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Concentration')
    plt.legend([dis[0], dis[1], dis[1] + " Shifted"],
               loc='upper left')

    # loc = 'shiftedSeries/'+df.columns.values[comb[0]]+df.columns.values[comb[1]]+'Reading.png'
    # plt.savefig(loc,dpi=300)
    # plt.clf()
    plt.show()

    f, ax = plt.subplots(figsize=(9, 3))
    ax.plot(list(range(-lag_range, lag_range + 1)), rs, marker='o', )
    # ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    # ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.axvline(offset, color='k', linestyle='--', label='Peak synchrony')
    ax.set(title='Offset for {dis[0]} to {dis[1]} = {offset} hours',
           ylim=[-1, 1], xlabel='Offset', ylabel='Pearson r')
    plt.legend()

    # loc = 'shiftedSeries/' + df.columns.values[comb[0]] + df.columns.values[comb[1]] + 'Offset.png'
    # plt.savefig(loc, dpi=300)
    # plt.clf()
    plt.show()


def meteo_analysis(df):
    meteo_data = xr.open_dataset('Files/meteo_data.nc')['meteo']

    factors, districts = pd.Series(
        ['Temperature [2 m]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure', 'Precipitation',
         'Cloud Cover High', 'Cloud Cover Medium', 'Cloud Cover Low', 'Sunshine Duration', 'Shortwave Radiation',
         'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
         'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]',
         'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
         'Wind Direction [700 mb]', 'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]',
         'Temperature [850 mb]', 'Temperature [700 mb]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]',
         'Soil Moisture [0-10 cm down]']), df.columns.values

    for lag in range(0, 3, 3):
        z = factors.apply(lambda x: df.shift(lag).corrwith(get_factor_data(meteo_data, x), axis=0))
        z.index = factors.values
        print(z)

        fig = go.Figure(data=go.Heatmap(
            z=z, y=factors, x=districts,
            colorscale='icefire',
            zmin=-1, zmax=1,
            reversescale=True
        ))
        fig.update_layout(
            autosize=False, width=1800, height=450 * 3,
            title="PM2.5 correaltion with meteorological factors",
            xaxis_title="District", yaxis_title="Factors",
            font=dict(size=21, color="#3D3C3A"
                      )
        )
        fig.show(config={'displayModeBar': False, 'responsive': True})


def missing_bar(df):
    single_district_data = df['Dhaka'].values
    idx_pairs = np.where(np.diff(np.hstack(([False], (np.isnan(single_district_data)), [False]))))[0].reshape(-1, 2)
    counts = (Counter(idx_pairs[:, 1] - idx_pairs[:, 0]))
    sorted_counts = dict(sorted(counts.items()))

    # x = ['One', 'Two', 'Three', 'Four', 'More']
    # y = list(sorted_counts.values())[:4] + [sum(list((sorted_counts.values()))[4:])]

    y, x = list(sorted_counts.values()), list(sorted_counts.keys())
    # x = np.arange(len(y))+1

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_traces(marker_color='#3090C7', marker_line_color='#2554C7', marker_line_width=1.2, opacity=0.8)
    fig.update_layout(title_text='Missing Value Length')
    fig.show()


def FillMissingDataFromHours(x, hours=1):
    ss = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromYears(y):
    ss = [y.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    return y.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def correlation_seasonal(corrArray, rows=2, cols=2, title=''):
    sub_titles = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']
    # sub_titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=sub_titles, vertical_spacing=0.1)
    # fig = fig.update_yaxes(side='right')

    import plotly.figure_factory as ff

    annotList = []
    for i in range(rows):
        for j in range(cols):
            fig1 = ff.create_annotated_heatmap(z=np.flip(np.tril((corrArray[i][j])), 0),
                                               y=meta_data.index.to_list()[::-1], x=meta_data.index.to_list(),
                                               annotation_text=np.flip(np.tril(
                                                   np.absolute(np.random.randint(3, size=(
                                                       timeseries.shape[1], timeseries.shape[1])))),
                                                   0), colorscale='purpor', zmin=0, zmax=1)
            fig.add_trace(fig1.dataset[0], i + 1, j + 1)

            annot = list(fig1.layout.annotations)
            # annotList.extend(annot)

            for k in range(len(annot)):
                annot[k]['xref'] = 'x' + str(i * rows + j + 1)
                annot[k]['yref'] = 'y' + str(i * rows + j + 1)

    fig.update_layout(annotations=annotList)

    # for i in range(len(fig.layout.annotations)):
    #     fig.layout.annotations[i].font.size = 9

    for i, t in enumerate(sub_titles):
        fig['layout']['annotations'][i].update(text=t, font=dict(
            family="Courier New, monospace",
            size=27,
            color="#7f7f7f"
        ))

    fig.update_layout(title_text=title, height=7500, width=1500)
    fig.show()


def stacked_bar(timeseries):
    def ranking(timeseries): return timeseries.mean().sort_values()

    def cut_and_count(x): return pd.cut(x, AQScale, labels=categoryName).value_counts() / x.count()

    ranking = ranking(timeseries)
    category_frquency = timeseries.apply(cut_and_count)
    category_frquency = category_frquency[ranking.index]

    # category_frquency = category_frquency.T
    # category_frquency['grp'] = category_frquency['Hazardous'] + category_frquency['Very Unhealthy'] + category_frquency['Unhealthy']
    # category_frquency['grp'] = category_frquency['Good'] + category_frquency['Moderate'] + category_frquency['Unhealthy for Sensitive Groups']
    # category_frquency = category_frquency.sort_values(by=['grp']).T
    # category_frquency = category_frquency.drop('grp')

    datas = [go.Bar(x=category_frquency.columns.values, y=row,
                    marker_color=colorScale[categoryName.tolist().index(idx)],
                    name=idx, opacity=.666) for idx, row in category_frquency.iterrows()]
    fig = go.Figure(data=datas)
    fig.update_layout(
        width = 300,
        legend_orientation="h",
        font=dict(size=24),
        barmode='stack',
        template='plotly_white',
        legend={"x": 0, "y": -.3}
    )
    fig.show()


def changes_in_months(timeseries):
    monthly_average = pd.concat(
        [timeseries[timeseries.index.month == i + 1].stack().droplevel(1).resample('Y').median() for i in range(12)],
        axis=1)
    monthly_average.columns = month_names
    pct_change = monthly_average.pct_change().fillna(0) * 100

    fig = go.Figure()

    month_colors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]

    for row, color in zip(month_names, month_colors):
        name = row
        fig.add_trace(go.Scatter(
            x=pct_change.index,
            y=pct_change[name],
            name=name,
            line=dict(color=color)
        ))

    fig.show()


def changes_in_districts(timeseries):
    yearly_average = timeseries.resample('Y').median()

    print(yearly_average)
    pct_change = yearly_average.pct_change().fillna(0) * 100
    print(pct_change.to_string())

    fig = go.Figure()

    for id, row in get_metadata().iterrows():
        name = id.replace(' ', '_')
        fig.add_trace(go.Scatter(
            x=pct_change.index,
            y=pct_change[name],
            name=name
        ))
    fig.show()


def overall_stats(timeseries):
    allData = timeseries.stack()
    # print(allData.droplevel(1))
    print(allData.describe())


def frequency_clustering(df):
    def ElbowPlot(dataPoints):
        res, n_cluster = list(), range(1, 10)
        for n in n_cluster:
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(dataPoints)
            res.append(kmeans.inertia_)
            # res.append(np.average(np.min(cdist(dataPoints, kmeans.cluster_centers_, 'euclidean'), axis=1)))

        res = [-KMeans(n_clusters=i).fit(dataPoints).score(dataPoints) for i in n_cluster]

        fig = go.Figure(data=go.Scatter(x=list(n_cluster), y=res,
                                        line=dict(color='#4C9BE2', width=3),
                                        marker=dict(color='#1A69B0', size=15)))
        fig.update_layout(font=dict(size=30), xaxis_title='Number of clusters', yaxis_title='Inertia')
        fig.show()

    metaFrame = get_metadata()
    norm = bucketing(df, AQScale).T
    # dataPoints, n_clusters = norm.values[:, :], 3
    dataPoints, n_clusters = df.resample('M').mean().T, 3

    # ElbowPlot(dataPoints)

    alg = KMeans(n_clusters=n_clusters)
    alg.fit(dataPoints)

    labels = pd.DataFrame(alg.labels_, index=metaFrame.index, columns=['label'])

    # districtMeanandLabels = pd.concat([df.mean(), labels], axis=1)
    districtMeanandLabels = labels.assign(mean=df.mean())

    districtMeanAndGroup = districtMeanandLabels.groupby('label').mean().sort_values('mean').round(2).assign(
        # color=['#C38EC7', '#3BB9FF', '#8AFB17', '#EAC117', '#F70D1A', '#7D0541','#FFFFFF','#000000'][:n_clusters])
        # color=['#3BB9FF', '#8AFB17', '#EAC117', '#F70D1A', '#7D0541','#FFFFFF','#000000'][:n_clusters])
        color=['#ffb2b2', '#FF0000', '#990000'], symbol=['o', 'o', 'o'])
    # color=['#8AFB17', '#EAC117', '#F70D1A'], symbol=['P', '*', 's'])
    # color=['#8AFB17', '#EAC117', '#F70D1A', '#170A14'], symbol=['P', '*', 's','o'])
    districtMeanAndGroup.columns = ['category', 'color', 'symbol']

    labels['color'] = labels.apply(lambda x: districtMeanAndGroup.loc[x['label']]['color'], axis=1)
    labels['symbol'] = labels.apply(lambda x: districtMeanAndGroup.loc[x['label']]['symbol'], axis=1)
    # labels['color'] = '#566D7E'

    metaFrame = pd.concat([metaFrame, labels], axis=1)

    representative = districtMeanandLabels.groupby('label').apply(lambda x: x['mean'].idxmax())
    # representative = representative[districtMeanAndGroup.index]
    print(representative)

    mapPlot(metaFrame, ('Average Reading', districtMeanAndGroup))

    # BoxPlotHour(df[representative])
    # BoxPlotSeason(df[representative])
    # PairDistributionSummary(df[representative])
    # PLotlyTimeSeries(df['2019-09':'2019-12'][representative])


def representative_district_analysis(df):
    respresentativeDistricts = ['Kishorganj', 'Bogra', 'Nagarpur', 'Jessore', 'Nawabganj', 'Dhaka']

    cross_corr_columns = ['Nawabganj', 'Sherpur', 'Kishorganj', 'Kushtia', 'Nagarpur',
                          'Narsingdi', 'Satkhira', 'Pirojpur', 'Lakshmipur']
    CrossCorrelation(timeseries[cross_corr_columns])
    CrossCorrelation(timeseries)
    paper_comparison(df)

    city_analysis(timeseries['Dhaka']['2017'])
    city_analysis(df['Narayanganj']['2017-02':'2018-02'])
    city_analysis(df['Mymensingh']['2019-02':'2019-04'])

    latex_custom_table_format(timeseries.describe().T)
    missing_data_fraction(timeseries)

    df = df.fillna('0')

    box_plot_series(df)
    violin_plot_year(df)

    df[respresentativeDistricts].apply(grouped_box)
    BoxPlotSeason(df[respresentativeDistricts])
    BoxPlotHour(df[respresentativeDistricts])
    pair_distribution_summary(df[respresentativeDistricts])


def pair_distribution(timeseries):
    zones = ['Dhaka', 'Nagarpur', 'Kishorganj', 'Jamalpur', 'Kushtia', 'Barisal', 'Narsingdi', 'Tongi', 'Narayanganj']
    g = sns.PairGrid(timeseries[zones].resample('D').mean(), diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.close("all")
    sns.set()
    # save_data()
    # sns.set_style("whitegrid")
    meta_data, timeseries = get_metadata(), get_series()

    # PairDistribution(timeseries)
    # day_night_distribution(timeseries)
    # changes_in_districts(timeseries)
    # changes_in_months(timeseries)
    # print(Ranking(timeseries))

    # SimpleTimeseries(timeseries)
    # overall_stats(timeseries)
    # StackedBar(timeseries)
    # PairDistributionSummary(timeseries.iloc[:,:30])
    # MissingDataHeatmap(timeseries)
    # MissingDataFraction(timeseries)
    # FrequencyClustering(timeseries)
    # prepare_color_table()

    # df.describe().T.to_csv('GenralStats.csv')

    series_with_heavy_missing, metadata_with_heavy_missing = get_series(), get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_region_and_country_series(series,
                                                                                                         metadata)
    # day_night_distribution(country_series)
    # PLotlyTimeSeries(country_series)
    stacked_bar(country_series)
    stacked_bar(region_series)