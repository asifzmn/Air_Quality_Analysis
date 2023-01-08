from collections import Counter
from itertools import combinations
from GIS.GeoPandas import mapArrow, mapPlot
# from meteorology.meteoblue_data_preparation import get_factor_data
from exploration import *
import plotly.graph_objects as go
import xarray as xr
import more_itertools


def CrossCorr(df):
    def crosscorr(datax, datay, lag=0, wrap=False):
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy)

        # print(datax.dtype())

        return datax.corr(datay.shift(lag))

    # r_window_size = 120
    # # Interpolate missing data.
    # # df_interpolated = df.interpolate()
    # # Compute rolling window synchrony
    # rolling_r = df.iloc[:,0].rolling(window=r_window_size, center=True).corr(df.iloc[:,1])
    # f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # df.rolling(window=30, center=True).median().plot(ax=ax[0])
    # ax[0].set(xlabel='Frame', ylabel='Smiling Evidence')
    # rolling_r.plot(ax=ax[1])
    # ax[1].set(xlabel='Frame', ylabel='Pearson r')
    # plt.suptitle("Smiling data and rolling window correlation")
    # plt.show()

    d1 = df.iloc[:, 0]
    d2 = df.iloc[:, 1]
    # seconds = 5
    # fps = 3
    lagRange = 15
    rs = [crosscorr(d1, d2, lag) for lag in range(-lagRange, lagRange + 1)]
    offset = np.floor(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    print(offset)

    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    # ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[.1, .31], xlim=[0, 301], xlabel='Offset',
    ax.set(title=f'Offset = {offset} frames\nS1 leads <{max(rs)}> S2 leads', xlabel='Offset',
           ylabel='Pearson r')
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    plt.legend()
    plt.show()


def CrossCorr(time_del, time_stamp, df, lagRange=3):
    meta_data = None

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


def correlation_heatmap_with_linear_meteorological_variables_with_lag(df):
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

    for lag in range(0, 6, 1):
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

        # # print((all_data[factor_rename].shift(30*1)))
        # lag_range = list(range(0,24))
        # lag_series = pd.Series(data=[abs(all_data['PM2.5 Reading'].shift(30*i).corr(all_data[factor_rename]))
        # for i in lag_range],index=lag_range)
        # fig = px.scatter(lag_series)
        # fig.show()
        # print(lag_series)
        # print(lag_series.idxmax())
        # print()

# for dis1 in metaFrame.index.values:
#     for dis2 in metaFrame.index.values:
#         lagRange = 3
#         rs = [crosscorr(df[dis1], df[dis2], lag) for lag in range(-lagRange, 0)]
#         offset = - int(np.floor(len(rs) / 2) - np.argmax(rs))
#         if offset<0:
#             print(offset)
#             print(angleFromCoordinate(dis1, dis2))
