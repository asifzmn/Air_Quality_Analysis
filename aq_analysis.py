# import re
# import geojsoncontour
# from collections import Counter
# from scipy.spatial.distance import cdist
# from pandas_profiling import ProfileReport
# from CoronaBeforeAfter import GroupedBox
# from GeoMapPlotly import SliderMapCommon
# from Correlation_Measures import *
# from DataPreparation import *
from sklearn.cluster import KMeans
from itertools import combinations
from plotly.subplots import make_subplots
from GIS.GeoPandas import mapArrow, mapPlot
from meteorological_variables import *
from related.GeoMapMatplotLib import MapPlotting
from visualization_modules import *
import more_itertools

month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']

colorScale, categoryName, AQScale = get_category_info()
save = '/home/az/Pictures/'


def latex_simple_table_format(df):
    new_df = df.groupby([df.index.year]).agg(['min', 'mean', 'max'])
    new_df.index.set_names(['Year'], inplace=True)
    print(new_df.T.to_latex(col_space=3))


def latex_custom_table_format(stats):
    stats.to_csv('Files/general_stats.csv')
    # stats['count'] = stats['count'].astype('int')

    # stats = stats.iloc[:,[0,1,2,3,5,7]].round(1)

    latex_data = stats.to_latex(col_space=3).replace("\\\n", "\\ \hline\n").replace('\\toprule', '\\toprule\n\\hline')
    substring = latex_data[
                latex_data.index('\\begin{tabular}{') + len('\\begin{tabular}{') - 1:latex_data.index('}\n') + 1]
    latex_data = latex_data.replace(substring, '|'.join(substring))

    for axisName in stats.columns: latex_data = latex_data.replace(axisName, f"\\textbf{{{axisName}}}")
    # for axisName in stats.index: latex_data = latex_data.replace(axisName, f"\\textbf{{{axisName}}}")

    latex_data = latex_data.replace('25\%', "\\textbf{Q1}").replace('50\%', "\\textbf{Q2}").replace('75\%',
                                                                                                    "\\textbf{Q3}")
    latex_data = latex_data.replace('{Tungi}para', "{Tungipara}")

    print(latex_data)


def diurnality(x):
    if (x >= 6) and (x < 12):
        return 'Morning'
    elif (x >= 12) and (x < 17):
        return 'Afternoon'
    elif (x >= 17) and (x < 20):
        return 'Evening'
    else:
        return 'Night'


def CityAnalysis(s):
    print(s.mean())
    # print(s.resample('M').mean().values)

    # s = pd.concat([s,s.index.to_series().dt.hour],axis=1)
    # s['index'] = s['index'].apply(diurnally)
    # s.columns = ['reading','daytime']
    # print(s.groupby('daytime').mean())


def PaperComparision():
    paperData = pd.read_csv('/home/az/Desktop/AQ Overall Comparision.csv', sep='\t')
    latex_custom_table_format(paperData)


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


def HeatMapDriver(data, vmin, vmax, title, cmap):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(data, mask=mask, cmap=cmap, square=True, linewidths=.5, vmin=vmin,
                         vmax=vmax, xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title)
    plt.show()


def KDE(df):
    for comb in list(combinations(range(len(df.columns.values)), 2)):
        with sns.axes_style('white'): sns.jointplot(df[df.columns.values[comb[0]]], df[df.columns.values[comb[1]]],
                                                    data=df, kind='kde')
        plt.show()


def TriangularHeatmap(title, df):
    plt.figure(figsize=(6, 6))
    HeatMapDriver(df.corr(), 0, 1, str(title.year) + '/' + str(title.month), 'Purples')


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


def PLotlyTimeSeries(df, missing=None):
    fig = go.Figure()
    for d in df: fig.add_trace(go.Scatter(x=df.index, y=df[d], name=d))

    fig.update_traces(mode='markers+lines', marker=dict(line_width=0, symbol='circle', size=5))
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


def Bucketing(reading, bins): return reading.apply(
    lambda x: pd.cut(x, bins).value_counts().sort_index()).apply(
    lambda x: x / x.sum())


def ratioMapPlotting(reading, timeStamp):
    norm = Bucketing(reading, [55.5, 500])
    MapPlotting(meta_data, timeseries.mean().values, ratiodata=norm, title=str(timeStamp.year))


def MissingDataInfo(df):
    print((df.isna().sum()))
    print(df.isnull().any(axis=1).sum())  # any null reading of a district for a time stamp
    print(df.isnull().all(axis=1).sum())  # all null reading of a district for a time stamp

    # HeatMapDriver(dfm.corr(), -1, 1, '', 'RdBu')


def MissingBar(df):
    singleDistrictData = df['Dhaka'].values
    idx_pairs = np.where(np.diff(np.hstack(([False], (np.isnan(singleDistrictData)), [False]))))[0].reshape(-1, 2)
    counts = (Counter(idx_pairs[:, 1] - idx_pairs[:, 0]))
    sortedCounts = dict(sorted(counts.items()))

    # x = ['One', 'Two', 'Three', 'Four', 'More']
    # y = list(sortedCounts.values())[:4] + [sum(list((sortedCounts.values()))[4:])]

    y, x = list(sortedCounts.values()), list(sortedCounts.keys())
    # x = np.arange(len(y))+1

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_traces(marker_color='#3090C7', marker_line_color='#2554C7', marker_line_width=1.2, opacity=0.8)
    fig.update_layout(title_text='Missing Value Length')
    fig.show()


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


def CrossCorr(time_del, time_stamp, df, lagRange=3):
    def crosscorr(datax, datay, lag=0, wrap=False):
        return datax.astype('float64').corr(datay.shift(lag).astype('float64'))

    # r_window_size = 24
    #
    # rolling_r = df.iloc[:, 0].rolling(window=r_window_size, center=True).corr(df.iloc[:, 1])
    # f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # df.iloc[:, 0:2].rolling(window=r_window_size, center=True).median().plot(ax=ax[0])
    # ax[0].set(xlabel='Frame', ylabel='Series')
    # rolling_r.plot(ax=ax[1])
    # ax[1].set(xlabel='Frame', ylabel='Pearson r')
    # plt.suptitle("Rolling window correlation")
    # plt.show()
    # print(list(combinations(range(len(df.shape)), 2)))

    lags, lagDir, lagmatrix = [], [], np.zeros((df.shape[1], df.shape[1]))
    for comb in list(combinations(range((df.shape[1])), 2)):
        rs = [crosscorr(df[df.columns.values[comb[0]]], df[df.columns.values[comb[1]]], lag) for lag in
              range(-lagRange, lagRange + 1)]
        offset = - int(np.floor(len(rs) / 2) - np.argmax(rs))
        lags.append(offset)
        lagmatrix[comb[1]][comb[0]], lagmatrix[comb[0]][comb[1]] = offset, -offset

        if offset == 0: continue

        # print(df.columns.values[comb[0]],df.columns.values[comb[1]],offset)
        # ShiftedSeries(df,[df.columns.values[comb[0]],df.columns.values[comb[1]]],lagRange,offset,rs)

        # if offset < 0:
        #     dir = (angleFromCoordinate(df.columns.values[comb[0]], df.columns.values[comb[1]]))
        # else:
        #     dir = (angleFromCoordinate(df.columns.values[comb[1]], df.columns.values[comb[0]]))
        # lagDir.append(dir)

    # if len(lagDir)>0:
    #     WindGraphTeamEstimate(np.array(lagDir), ['Overall'])
    #     st = ((timeStamp.to_pydatetime()))
    #     en = ((timeStamp.to_pydatetime())+timedelta(hours=48))
    #     WindGraphTeam(dfm.loc[:,st:en,:])
    # print(df.columns.values[comb[0]],df.columns.values[comb[1]],offset)

    if list(Counter(lags).keys()) == [0]: return lagmatrix, lagDir

    mat = pd.DataFrame(data=lagmatrix.astype('int'), columns=df.columns.values, index=df.columns.values)
    # print(mat.dtypes)
    # HeatMapDriver(mat, -lagRange, lagRange, str(timeStamp),
    #               sns.diverging_palette(15, 250, s=90, l=50, n=90, center="light"))
    # MapPlotting(metaFrame, df.mean().values, vec=lagmatrix, title=str(timeStamp))
    print(Counter(lags))
    mapArrow(np.zeros(meta_data.shape[0]).astype('int'), mat, df.index.date)

    return lagmatrix, lagDir


def WindGraphTeamEstimate(meteoData, alldis):
    colorPal = np.array(['#ffffff'])
    directions = np.array([[[get_cardinal_direction(meteoData), 'Wind']]])
    plotly_rose_plot(directions, colorPal, alldis)


def MeteoAnalysis(df):
    meteoData = xr.open_dataset('Files/meteoData.nc')['meteo']

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
        z = factors.apply(lambda x: df.shift(lag).corrwith(get_factor_data(meteoData, x), axis=0))
        z.index = factors.values

        print(z)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            y=factors,
            x=districts,
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


def FillMissingDataFromHours(x, hours=1):
    ss = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromYears(y):
    ss = [y.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    # ss = [x.shift(shft, freq='D') for shft in [-365, 365]]
    return y.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def CorrationSeasonal(corrArray, rows=2, cols=2, title=''):
    subpTitles = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']
    # subpTitles = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subpTitles, vertical_spacing=0.1)
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

    for i, t in enumerate(subpTitles):
        fig['layout']['annotations'][i].update(text=t, font=dict(
            family="Courier New, monospace",
            size=27,
            color="#7f7f7f"
        ))

    fig.update_layout(title_text=title, height=7500, width=1500)
    fig.show()


def BoxPlotDistrict(df):
    plt.figure(figsize=(20, 8))
    ax = sns.boxplot(data=df, color="grey")
    for i, c in enumerate(colorScale): ax.axhspan(AQScale[i], AQScale[i + 1], facecolor=c, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", size=21)
    ax.set(ylim=(0, 250))
    plt.show()


def BoxPlotWeek(df):
    weekDay = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    dayColors = ["#ff0000"] * 2 + ["#8fcadd"] * 5
    seasons = ['Weekday', 'Weekend']
    my_pal = dict(zip((np.unique(df.index.day_name())), dayColors))

    for d in df.columns.values[3:5]:
        sns.boxplot(x=df.index.day_name(), y=d, data=df, palette=my_pal)  # weekday_name,month,day,hour
        for area, color in zip(seasons, np.unique(dayColors)): plt.scatter([], [], c=color, alpha=0.66, s=150,
                                                                           label=str(area))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Day Type')
        plt.ylim(0, 350)
        plt.show()


def cutAndCount(x): return pd.cut(x, AQScale, labels=categoryName).value_counts() / x.count()


def StackedBar(timeseries):
    ranking = Ranking(timeseries)
    category_frquency = timeseries.apply(cutAndCount)
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
        legend_orientation="h",
        font=dict(size=24),
        barmode='stack',
        template='plotly_white',
        legend={"x": 0, "y": -.3}
    )
    fig.show()


def CrossCorrelation(df):
    def HeatmapCrossCorr(df):
        plt.figure(figsize=(9, 9))
        sns.heatmap(df, vmin=0, vmax=3, cmap='Purples')
        plt.tight_layout()
        plt.show()

    def allLagRange(x, df, lag, readings):
        def bestCorr(x, ss): return ss.apply(lambda y: y.corr(x)).argmax()

        print(x)
        ss = pd.concat([x.shift(shift) for shift in np.arange(lag + 1)], axis=1)
        readings = [df[reading[0]:reading[-1]] for reading in readings]
        new_df = pd.concat([reading.apply(bestCorr, ss=ss) for reading in readings], axis=1)
        new_df.columns = [str(reading.index.date[0]) + ' to ' + str(reading.index.date[-1]) for reading in readings]
        ser, ser.name = new_df.stack(), x.name
        return ser

    pth = berkeley_earth_data
    window, step, lag = 5, 2, 3
    fileName, indices, freq = 'lagTimeMatrix_', ['Leader', 'Date', 'Follower'], str(window) + 'D'

    # df = df.rolling(center=True, window=4).mean()
    df = df['2019']
    # print(df['2017'].isnull().sum().sum())
    # print(df['2018'].isnull().sum().sum())
    # print(df['2019'].isnull().sum().sum())

    readings = np.array(list(more_itertools.windowed(df.index[lag:-lag], n=window * 24, step=step * 24)))

    # lagTimeMatrix = df.apply(allLagRange, df=df, lag=lag, readings=readings).stack()
    # lagTimeMatrix.index = lagTimeMatrix.index.rename(indices)
    # lagTimeMatrix.to_csv(pth + fileName + freq)

    lagTimeMatrix = pd.read_csv(pth + fileName + freq, index_col=indices)
    lagTimeMatrix = lagTimeMatrix.unstack('Date')
    lagTimeMatrix.columns = lagTimeMatrix.columns.droplevel(0)

    print(lagTimeMatrix.shape)
    print(lagTimeMatrix.stack().value_counts())
    metaFrame = get_metadata().assign(symbol='H')

    for key, df in lagTimeMatrix.iloc[:, :].items():
        if df.nunique() > 1:
            print(key)
            if key == '2019-01-27 to 2019-02-01' or key == '2019-10-26 to 2019-10-31':
                print(df[df != 0])
                df[df != 0].to_csv(key + '.csv')
                HeatmapCrossCorr(df.unstack())
                mapArrow(metaFrame, df.unstack(), df.name)


def save_data():
    meta_data, timeseries = get_metadata(), get_series()['2017':'2019']
    meta_data.to_csv('zone_data.csv')
    timeseries.to_csv('pm_time_series.csv')


def Ranking(timeseries):
    return timeseries.mean().sort_values()


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


def MissingDataFraction(timeseries):
    missing_percentage = (timeseries.isnull().sum() / len(timeseries) * 100).round(2)
    missing_percentage.to_csv('missing_percentage_1.csv')


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
    norm = Bucketing(df, AQScale).T
    # dataPoints, n_clusters = norm.values[:, :], 3
    dataPoints, n_clusters = df.resample('M').mean().T, 3

    # ElbowPlot(dataPoints)

    alg = KMeans(n_clusters=n_clusters)
    alg.fit(dataPoints)

    labels = pd.DataFrame(alg.labels_, index=metaFrame.index, columns=['label'])

    # print(labels)
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


def PairDistribution(timeseries):
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
    # print(Ranking(timeseries))

    # changes_in_districts(timeseries)
    # changes_in_months(timeseries)

    # SimpleTimeseries(timeseries)
    # overall_stats(timeseries)
    # StackedBar(timeseries)
    # PairDistributionSummary(timeseries.iloc[:,:30])
    # MissingDataHeatmap(timeseries)
    # MissingDataFraction(timeseries)
    # FrequencyClustering(timeseries)

    # df.describe().T.to_csv('GenralStats.csv')

    # respresentativeDistricts = ['Kishorganj', 'Bogra', 'Nagarpur', 'Jessore', 'Nawabganj', 'Dhaka']

    # cross_corr_columns = ['Nawabganj','Sherpur','Kishorganj','Kushtia','Nagarpur',
    # 'Narsingdi','Satkhira','Pirojpur','Lakshmipur']
    # CrossCorrelation(timeseries[cross_corr_columns])
    # CrossCorrelation(timeseries)
    # PaperComparision()

    # CityAnalysis(timeseries['Dhaka']['2017'])
    # CityAnalysis(df['Narayanganj']['2017-02':'2018-02'])
    # CityAnalysis(df['Mymensingh']['2019-02':'2019-04'])

    # ColorTable()

    # latex_custom_table_format(timeseries.describe().T)
    # MissingDataFraction(timeseries)

    # df = df.fillna('0')

    # BoxPlotDistrict(df)
    # ViolinPLot(df)

    # df[disRep].apply(GroupedBox)
    # BoxPlotSeason(df[respresentativeDistricts])
    # BoxPlotHour(df[respresentativeDistricts])
    # PairDistributionSummary(df[respresentativeDistricts])