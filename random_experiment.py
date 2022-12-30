# BoxPlotYear(df)
# BoxPlotDistrict(df)
# MissingDataHeatmap(df)
# MissingBar(df)
# PLotlyTimeSeries(df)
# MeteoAnalysis(df)
# SliderMapCommon(df, metaFrame, ['M', '%Y %B'])
# SliderMapCommon(df['2017':'2020'], metaFrame, ['Y', '%Y %B'])
# SliderMapCommon(df['2020'], metaFrame, ['D', '%Y %B %D'],True)
# StackedBar(df)
# MapPlotting(metaFrame, df[date].mean().values,vec=lagMatrix)
# TriangularHeatmap(df)
# SeasonalDecomposition(monthData[-1][2].iloc[:,3])

# PLotlyTimeSeries(df[[cols]],missing[[cols]])
# PLotlyTimeSeries(missing[[cols]],missing)

# df = df.apply(FillMissingDataFromHours, args=[2])
# df = df.apply(FillMissingDataFromDays, args=[3])
# df = df.apply(FillMissingDataFromYears)

# dfm = GetAllMeteoData()

# for freq,data in df['2020'].resample('W').mean().iterrows():
#     print(data)
#     mapPlot(data,str(freq))

# changePoints = np.hstack((([0],np.argwhere(np.diff((df.notnull().all(axis=1)).values)).squeeze(),
# [df.shape[0]-1]))).reshape(-1,2)
# # print((changePoints[:,1]-changePoints[:,0])/(24*30))
# cleanData = (changePoints[(changePoints[:,1]-changePoints[:,0])/(24*30)>1])
# for cl in cleanData:print(df.index[cl[0]],df.index[cl[1]-1])

# for cols in df.columns.values:
#     days = 7
#     ss = [df[cols].shift(shft, freq='H') for shft in np.delete(np.arange(-days,days+1),
#     [-3+7,-2+7,-1+7,0+7,1+7,2+7,3+7])*24]
#     df[cols] = df[cols].fillna((pd.concat(ss, axis=1).mean(axis=1)))

# totalEstimate = []
# for i, [timeDel, timeStamp, reading] in enumerate(readings[:]):
#     print(i, timeStamp)
#     # ratioMapPlotting(reading)
#     # BoxPlotHour(reading)
#     l1, l2 = CrossCorr(timeDel, timeStamp, reading, lagRange=2)
#     totalEstimate.extend(l2)
#     # BoxPlotSeason(reading)
#     # TriangularHeatmap(timeStamp,reading.astype('float64'))
# # WindGraphTeamEstimate(np.array(totalEstmare), ['Overall'])

# corrArray = np.array([df['2017-12':'2017-12'].corr().values,
#                       df['2018-03':'2018-03'].corr().values,
#                       df['2018-06':'2018-06'].corr().values,
#                       df['2018-09':'2018-09'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
# CorrationSeasonal(corrArray)
#
# corrArray = np.array([df['2018-01':'2018-01'].corr().values,
#                       df['2018-04':'2018-04'].corr().values,
#                       df['2018-07':'2018-07'].corr().values,
#                       df['2018-10':'2018-10'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
# CorrationSeasonal(corrArray)
#
# corrArray = np.array([df['2018-02':'2018-02'].corr().values,
#                       df['2018-05':'2018-05'].corr().values,
#                       df['2018-08':'2018-08'].corr().values,
#                       df['2018-11':'2018-11'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
# CorrationSeasonal(corrArray)
#
# corrArray = np.array([df['2017-12':'2018-02'].corr().values,
#                       df['2018-03':'2018-05'].corr().values,
#                       df['2018-06':'2018-08'].corr().values,
#                       df['2018-09':'2018-11'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
# CorrationSeasonal(corrArray)

# corrArray = (np.array([[df['2017-'+str(month+1)].corr().values]
# for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
# CorrationSeasonal(corrArray,rows=6,title = '2017')
#
# corrArray = (np.array([[df['2018-'+str(month+1)].corr().values]
# for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
# CorrationSeasonal(corrArray,rows=6,title = '2018')
#
# corrArray = (np.array([[df['2019-'+str(month+1)].corr().values]
# for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
# CorrationSeasonal(corrArray,rows=6,title = '2019')

# [yearData, monthData, weekData, dayData] = [
#     np.array([[timeDel, timeStamp, reading] for timeStamp, reading in
#               df.groupby(pd.Grouper(freq=timeDel)) if not reading.isnull().any().any()], dtype=object) for timeDel
#     in
#     (['Y', 'M', 'W', '3D'])]
# readings = dayData
# print(len(readings))
#

# # b = np.array([0, 4, 8, 12, 16, 20, 24])
# b = np.array([pd.to_datetime(0, format='%H'), pd.to_datetime(12, format='%H')])
# # l = ['Late Night', 'Early Morning', 'Morning', 'Noon', 'Eve', 'Night']
# l = ['Day',  'Night']
# df1['session'] = pd.cut(df.index, bins=b, labels=l)

# print([df1[2].mean().mean() for df1 in yearData])
# popYear = [157977153, 159685424, 161376708, 163046161]
# print(np.corrcoef([df1[2].mean().mean() for df1 in yearData][1:],popYear[1:]))
# print(metaFrame[['Population','avgRead']].corr())

# for label, content in df.items():print(label,content)

# print(df.to_markdown())
# print(df.to_html())
