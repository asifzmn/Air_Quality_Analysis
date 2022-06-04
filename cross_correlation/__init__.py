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
