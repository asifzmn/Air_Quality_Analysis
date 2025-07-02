import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.seasonal import seasonal_decompose

from data_preparation import *
import plotly.graph_objects as go

from data_preparation.additional_data_preparation import get_diurnal_period
from data_preparation.spatio_temporal_filtering import read_bd_data, get_bd_data_4_years, read_bd_data_4_years

color_scale, category_name, aq_scale = get_category_info()

colorScale, categoryName, AQScale = get_category_info()


def overall_stats(timeseries):
    all_data = timeseries.stack()
    # print(all_data.droplevel(1))
    print(all_data.describe())


def yearly_seasonal_decomposition_bd(series):
    series_bd_weekly = series.resample('W', label='left', loffset=pd.DateOffset(days=1)).mean()
    # .resample('W').mean().fillna(country_series.Bangladesh.mean())
    result = seasonal_decompose(series_bd_weekly)
    seasonal_component = result.seasonal.resample('H').ffill().reindex(series.index).ffill()
    stationary = (series - seasonal_component).to_frame()
    # stationary_diff = stationary.diff()
    # PLotlyTimeSeries(stationary)
    # PLotlyTimeSeries(stationary_diff)
    return stationary


def seasonal_decomposition(df):
    result = seasonal_decompose(df)
    print(result)
    result.plot()
    plt.show()

    # x = df - result.seasonal
    # diff = np.diff(x)
    # plt.plot(diff)
    # plt.show()
    #
    # plot_acf(diff, lags=10)
    # plot_pacf(diff, lags=10)
    # plt.show()


def missing_data_heatmap(df):
    import plotly.graph_objects as go

    missing = df.T.isnull().astype(int)
    # print(missing.sum(axis=1))

    bvals, dcolorsc = np.array([0, .5, 1]), []
    tickvals = [np.mean(bvals[k:k + 2]) for k in range(len(bvals) - 1)]
    ticktext, colors = ['Present', 'Missing'], ['#C6DEFF', '#2B3856']

    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # normalized values
    for k in range(len(colors)): dcolorsc.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])

    fig = go.Figure(data=go.Heatmap(
        z=missing,
        y=missing.index,
        x=missing.columns,
        colorscale=dcolorsc,
        colorbar=dict(thickness=75, tickvals=tickvals, ticktext=ticktext),
    ))
    fig.update_layout(
        title="Missing Data Information",
        yaxis_title="Zone",
        xaxis_title="Time",
        font=dict(
            size=15,
            color="#3D3C3A"
        ),
        # legend = dict(font = dict(family = "Courier", size = 50, color = "black")),
        # legend_font_size=27,
        height=900
    )
    fig.show()


def day_night_distribution_zonal(time_series, sampling_hours=1):
    # time_series = time_series.iloc[6:-18]
    time_series = time_series.resample(str(sampling_hours) + 'H').mean()
    time_series = time_series.fillna(-1)
    time_series = time_series.stack().reset_index().set_index('time')
    time_series = time_series.replace({-1: None})
    time_series.columns = ['zone', 'reading']

    # time_series['diurnal_name'] = np.tile(
    #     np.hstack((np.repeat('Day', 12 * 30 // sampling_hours), np.repeat('Night', 12 * 30 // sampling_hours))),
    #     ((time_series.shape[0] * sampling_hours) // (24 * 30)))

    time_series = time_series.join(get_diurnal_period())
    print(time_series)

    # print(timeseries.head(3000).to_string())

    fig = go.Figure()

    fig.add_trace(go.Violin(x=time_series['zone'][time_series['diurnal_name'] == 'day'],
                            y=time_series['reading'][time_series['diurnal_name'] == 'day'],
                            legendgroup='Yes', scalegroup='Yes', name='Day',
                            side='negative', line_color='orange')
                  )
    fig.add_trace(go.Violin(x=time_series['zone'][time_series['diurnal_name'] == 'night'],
                            y=time_series['reading'][time_series['diurnal_name'] == 'night'],
                            legendgroup='No', scalegroup='No', name='Night',
                            side='positive', line_color='blue')
                  )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay', font_size=27, legend_orientation='h',
                      xaxis_title="Zone", yaxis_title="PM2.5 Concentration", )
    fig.show()


def day_night_distribution_monthly(time_series, sampling_hours=1):
    # time_series = time_series.iloc[6:-18]
    time_series = time_series.resample(str(sampling_hours) + 'H').mean()
    time_series = time_series.fillna(-1)
    time_series = time_series.stack().reset_index().set_index('time')
    time_series = time_series.replace({-1: None})
    time_series.columns = ['zone', 'reading']

    # time_series['diurnal_name'] = np.tile(
    #     np.hstack((np.repeat('Day', 12 * 30 // sampling_hours), np.repeat('Night', 12 * 30 // sampling_hours))),
    #     ((time_series.shape[0] * sampling_hours) // (24 * 30)))

    time_series = time_series.join(get_diurnal_period())
    time_series['month'] = time_series.index.month_name()
    print(time_series)

    # print(timeseries.head(3000).to_string())

    fig = go.Figure()

    fig.add_trace(go.Violin(x=time_series['month'][time_series['diurnal_name'] == 'day'],
                            y=time_series['reading'][time_series['diurnal_name'] == 'day'],
                            legendgroup='Yes', scalegroup='Yes', name='Day',
                            # side='negative', line_color='orange')
                            side='negative', line_color='#C0C6C7')
                  )
    fig.add_trace(go.Violin(x=time_series['month'][time_series['diurnal_name'] == 'night'],
                            y=time_series['reading'][time_series['diurnal_name'] == 'night'],
                            legendgroup='No', scalegroup='No', name='Night',
                            side='positive', line_color='#040720')
                  )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay', font_size=27, legend_orientation='h',
                      xaxis_title="Zone", yaxis_title="PM2.5 Concentration (µgm<sup>-3</sup>)", height=1000, template='plotly_white')
    fig.show()


def pair_distribution_summary(timeseries, sampling_hours=1):
    # df = df[: str(max(df.index.date) + timedelta(days=-1))].resample(str(samplingHours) + 'H').mean()
    timeseries = timeseries.resample(str(sampling_hours) + 'H').mean()
    timeseries['daytime'] = np.tile(
        np.hstack((np.repeat('Day', 12 // sampling_hours), np.repeat('Night', 12 // sampling_hours))),
        ((timeseries.shape[0] * sampling_hours) // 24))
    print(timeseries)

    timeseries = timeseries.sort_values(by=['daytime'], ascending=False)
    g = sns.pairplot(timeseries, hue='daytime', palette=["#2B3856", "#FFF380"], plot_kws={"s": 9})

    for ax in plt.gcf().axes: ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    for ax in plt.gcf().axes: ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
    plt.show()


def BoxPlotYear(df):
    for district in df:
        ax = sns.boxplot(x=df.index.year, y=district, data=df, color="#00AAFF")  # weekday_name,month,day,hour
        pltSetUpAx(ax, "Year", "PM Reading", 'Yearly average reading in ' + district, ylim=(0, 200))


def BoxPlotSeason(timeseries):
    month_colors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]
    seasons = ['Spring', 'Winter', 'Summer', 'Autumn']
    seasonPalette = dict(zip((np.unique(timeseries.index.month)), month_colors))

    for zone in timeseries:
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x=timeseries.index.month, y=zone, data=timeseries,
                         palette=seasonPalette)  # weekday_name,month,day,hour
        for seasonName, color in zip(seasons, np.unique(month_colors)):
            plt.scatter([], [], c=color, alpha=0.66, s=150, label=str(seasonName))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=.5, title='Season')
        pltSetUpAx(ax, xlabel="Month", ylabel="PM Reading", title='Seasonality of months in ' + zone, ylim=(0, 300))


def BoxPlotHour(timeseries):
    color_map = LinearSegmentedColormap.from_list('DayNight',
                                                  ["#075077", "#075077", "#6f5a66", "#6f5a66",
                                                   "#eae466", "#eae466", "#eae466", "#eae466",
                                                   "#6f5a66", "#6f5a66", "#075077", "#075077"], N=24)
    hourPalette = dict(zip((np.unique(timeseries.index.hour)), color_map(np.arange(24))))
    for zone in timeseries.columns.values[:]:
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x=timeseries.index.hour, y=zone, data=timeseries,
                         palette=hourPalette)  # weekday_name,month,day,hour
        pltSetUpAx(ax, xlabel="Hour of day", ylabel="PM Reading", title='HourSeasonality' + zone, ylim=(0, 300))


def pltSetUp(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save=None):
    if not xlabel is None: plt.xlabel(xlabel)
    if not ylabel is None: plt.ylabel(ylabel)
    if not title is None: plt.title(title)
    if not xlim is None: plt.xlim(xlim[0], xlim[1])
    if not ylim is None: plt.ylim(ylim[0], ylim[1])

    if save is None:
        plt.show()
    else:
        plt.savefig(f"{save} {title}.png", dpi=300)
        plt.clf()


# def pltSetUpAx(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save='Box_plots/Save'):
def pltSetUpAx(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save=None):
    if not xlabel is None: ax.set_xlabel(xlabel)
    if not ylabel is None: ax.set_ylabel(ylabel)
    if not title is None: ax.set_title(title)
    if not xlim is None: ax.set_xlim(xlim[0], xlim[1])
    if not ylim is None: ax.set_ylim(ylim[0], ylim[1])

    if save is None:
        plt.show()
    else:
        plt.savefig(f"{save} {title}.png", dpi=300)
        plt.clf()
        print(f"{save} {title}.png")


def custom_time_series(df):
    resampled_df = df.resample('M')

    aggregated_value = pd.concat(
        [sampledSeries.stack().apply(['min', 'mean', 'median', 'max']) for _, sampledSeries in resampled_df], axis=1).T
    aggregated_value.index = [time for time, _ in resampled_df]

    print(aggregated_value[['min', 'median', 'max']])


    # aggregated_value = aggregated_value.iloc[:-1]
    # print(aggregated_value.to_string())
    # aggregated_value.to_csv('aggregated_value.csv')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=aggregated_value.index.tolist() + aggregated_value.index[::-1].tolist(),
        y=aggregated_value['max'].tolist() + aggregated_value['min'].tolist()[::-1],
        fill='toself', showlegend=False,
        fillcolor='rgba(127, 127, 127,0.3)',
        line_color='rgba(255,255,255,0)',
    ))

    # fig.add_trace(go.Scatter(
    #     x=aggregated_value.index, y=aggregated_value['mean'].tolist(),
    #     line_color='rgba(0, 38, 255,.45)', line_width=5, name='Mean',
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=aggregated_value.index, y=aggregated_value['median'].tolist(),
    #     line_color='rgba(255, 216, 1,.75)', line_width=5, name='Median',
    # ))

    # dhaka_series = df["Dhaka"].resample('M').apply(['mean', 'median'])
    # barisal_series = df["Barisal"].resample('M').apply(['mean', 'median'])
    #
    # fig.add_trace(go.Scatter(
    #     x=dhaka_series.index.tolist(), y=dhaka_series['mean'].tolist(),
    #     line_color='#2C3539', line_width=2, name='Dhaka_Mean',
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=dhaka_series.index.tolist(), y=dhaka_series['median'].tolist(),
    #     line_color='#737CA1', line_width=2, name='Dhaka_Median',
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=barisal_series.index.tolist(), y=barisal_series['mean'].tolist(),
    #     line_color='#E238EC', line_width=2, name='Barisal_Mean',
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=barisal_series.index.tolist(), y=barisal_series['median'].tolist(),
    #     line_color='#C12283', line_width=2, name='Barisal_Median',
    # ))

    # for country_name in df:
    #     print(country_name)
    #     country_monthly_series = df[country_name].resample('M').mean()
    #
    #     fig.add_trace(go.Scatter(
    #         x=country_monthly_series.index.tolist(), y=country_monthly_series,
    #         line_color=country_color_dict[country_name], line_width=2, name=country_name,
    #     ))

    region_names = ['Bangladesh']
    color_codes = ['royalblue']

    # region_names = ['Dhaka', 'Khulna', 'Barisal', 'Chittagong', 'Rajshahi']
    # color_codes = ['red', 'orange', 'blue', 'green', 'yellow']

    # region_names = ['Dhaka', 'Khulna', 'Chittagong', 'Rajshahi']
    # color_codes = ['red', 'blue', 'green', 'orange']

    for zone, color in zip(region_names, color_codes):
        print(zone)
        # country_monthly_series = df[zone].resample('WS').mean()
        # country_monthly_series = resampled_df[zone].mean()
        country_monthly_series = resampled_df[zone].median()

        fig.add_trace(go.Scatter(
            x=country_monthly_series.index.tolist(), y=country_monthly_series,
            line_color=color, line_width=3, name=zone,
        ))

    # # annotate_points = ['2017-01-01', '2018-01-14', '2019-01-13', '2017-07-02', '2018-07-15', '2019-07-14']
    # annotate_points = ['2017-01-31', '2018-01-31', '2019-01-31', '2017-07-31', '2018-07-31', '2019-07-31']
    # # annotate_points = ['2017-01-31', '2018-01-31', '2019-01-31', '2017-07-31', '2018-07-31', '2019-07-31','2017-12-31', '2018-12-31', '2019-12-31']
    # for annotate_point in annotate_points:
    #     print(aggregated_value.loc[annotate_point, 'median'])
    #     fig.add_annotation(x=annotate_point, y=aggregated_value.loc[annotate_point, 'median'],
    #                        text=aggregated_value.loc[annotate_point, 'median'])

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E4E2')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E4E2')

    fig.update_traces(mode='lines')
    fig.update_layout(
        height=1050,
        yaxis_title="PM2.5 Concentration",
        xaxis_title="Time",
        font=dict(size=27),
        legend_orientation="h",
        template='plotly_white'
    )
    fig.show()


def violin_plot_year(df):
    fig = go.Figure()
    # df = df.resample('M').mean()
    years = ['2018', '2019', '2020', '2021']

    for year in years:
        fig.add_trace(go.Violin(y=df[year].stack(),
                                name=year, box_visible=True,
                                meanline_visible=True, line_color='#566D7E', ))

    fig.update_layout(font=dict(size=21), width=900,
                      title="Air quality of Bangladesh over years",
                      yaxis_title="PM2.5 Concentration",
                      xaxis_title="Year",
                      )
    fig.show()


def box_plot_series(df):
    plt.figure(figsize=(20, 8))
    ax = sns.boxplot(data=df, color="grey")
    for i, c in enumerate(color_scale): ax.axhspan(aq_scale[i], aq_scale[i + 1], facecolor=c, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", size=21)
    ax.set_yticklabels(ax.get_yticklabels(), size=21)
    ax.set_ylabel('reading', fontsize=15)
    ax.set(ylim=(0, 250))
    plt.show()


def box_plot_week(df):
    day_colors = ["#ff0000"] * 2 + ["#8fcadd"] * 5
    seasons = ['Weekday', 'Weekend']
    my_pal = dict(zip((np.unique(df.index.day_name())), day_colors))

    for d in df:
        sns.boxplot(x=df.index.day_name(), y=d, data=df, palette=my_pal)  # weekday_name,month,day,hour
        for area, color in zip(seasons, np.unique(day_colors)): plt.scatter([], [], c=color, alpha=0.66, s=150,
                                                                            label=str(area))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Day Type')
        plt.ylim(0, 350)
        plt.show()


def PLotlyTimeSeries(df, missing=None):
    fig = go.Figure()
    for d in df: fig.add_trace(go.Scatter(x=df.index, y=df[d], name=d))

    fig.update_traces(mode='markers+lines', marker=dict(line_width=0, symbol='circle', size=5))
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()

def grouped_box_month_year(x):
    fig = go.Figure()

    # color_pal = ['#4AA02C', '#6AA121', '#7D0552', '#7D0500', '#2471A3']
    # color_pal = ['#4AA02C', '#6AA121', '#7D0552', '#7D0500']
    # color_pal = ['#FF69B4', '#FF6EB4', '#44C5FF', '#64C0FF']
    color_pal = ['#726E6D', '#726E6D', '#040720', '#040720']

    boxplot_stats = {}

    print(x)
    print(x.index.year.unique())
    # for year in pd.DatetimeIndex(x.index).year.unique():
    for year in x.index.year.unique():

        year_data = x[str(year)]
        monthly_data = {}
        for month_idx, month_name in enumerate(pd.DatetimeIndex(x.index).month_name().unique()):
            month_values = year_data[pd.DatetimeIndex(year_data.index).month_name() == month_name]
            if not month_values.empty:
                monthly_data[month_name] = month_values

                # Calculate boxplot statistics
                q1 = month_values.quantile(0.25)
                median = month_values.median()
                q3 = month_values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = max(month_values.min(), q1 - 1.5 * iqr)
                upper_bound = min(month_values.max(), q3 + 1.5 * iqr)

                # Store statistics
                if year not in boxplot_stats:
                    boxplot_stats[year] = {}
                boxplot_stats[year][month_name] = {
                    'min': lower_bound,
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'max': upper_bound
                }




        fig.add_trace(go.Box(
            # y=x[str(year)],
            y=x[str(year)],
            x=pd.DatetimeIndex(x.index).month_name(),
            name=year,
            marker_color=color_pal[year - 2018]
        ))

    fig.update_layout(
        # title=x.name,
        yaxis_title='PM2.5 Concentration (µgm<sup>-3</sup>)',
        xaxis_title='Zone',
        boxmode='group',
        yaxis=dict(
            range=[0, 250]),
        legend_orientation="h",
        height=1050,
        font_size=24,
        template='plotly_white'
    )

    fig.show()

    print("\nBoxplot Statistics:")
    for year in boxplot_stats:
        print(f"\nYear: {year}")
        for month, stats in boxplot_stats[year].items():
            print(f"  {month}:")
            print(f"    Min: {stats['min']:.2f}")
            print(f"    Q1: {stats['q1']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Q3: {stats['q3']:.2f}")
            print(f"    Max: {stats['max']:.2f}")


def box_plot_basic_experiment():
    df = region_series.resample('3D').max()
    plt.figure(figsize=(9, 6))
    ax = sns.boxplot(data=df.T, color="blue")
    pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'), ylim=(0, 500))
    # pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'))# df = df.resample('3D').max()
    # plt.figure(figsize=(9, 6))
    # ax = sns.boxplot(data=df.T, color="blue")
    # pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'), ylim=(0, 500))
    # # pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'))


def make_category_frequency(timeseries):
    def cut_and_count(x): return pd.cut(x, AQScale, labels=categoryName).value_counts() / x.count()

    def ranking(timeseries): return timeseries.mean().sort_values()

    ranking = ranking(timeseries)
    category_frequency = timeseries.apply(cut_and_count)
    category_frequency = category_frequency[ranking.index] * 100
    return category_frequency


def stacked_bar(timeseries):
    category_frquency = make_category_frequency(timeseries)

    print(category_frquency.round(2).to_string())

    # category_frquency = category_frquency.T
    # category_frquency['grp'] = category_frquency['Hazardous'] + category_frquency['Very Unhealthy'] + category_frquency['Unhealthy']
    # category_frquency['grp'] = category_frquency['Good'] + category_frquency['Moderate'] + category_frquency['Unhealthy for Sensitive Groups']
    # category_frquency = category_frquency.sort_values(by=['grp']).T
    # category_frquency = category_frquency.drop('grp')

    data = [go.Bar(y=category_frquency.columns.values, x=row,
                    text=row.round(1),  # Show percentage labels
                    textposition='inside',  # Place inside the bars
                    marker_color=colorScale[categoryName.tolist().index(idx)],
                    orientation='h',  # Set horizontal orientation
                    name=idx, opacity=.666) for idx, row in category_frquency.iterrows()]
    fig = go.Figure(data=data)
    fig.update_layout(
        width=1800,
        height=900,
        legend_orientation="h",
        font=dict(size=24),
        barmode='stack',
        template='plotly_white',
        yaxis_title='Percentage of occurrence'
        # legend={"x": 0, "y": -.3}
    )
    fig.show()

#                                     Sylhet  Chittagong  Rangpur  Barisal  Mymensingh  Rajshahi  Khulna  Dhaka
# Good                              7.21        5.31     3.54     4.01        3.38      2.34    3.16   3.76
# Moderate                         39.33       40.15    41.13    38.99       38.53     37.04   36.80  33.90
# Unhealthy for Sensitive Groups   18.05       18.02    17.21    17.29       17.02     17.39   17.51  17.43
# Unhealthy                        33.75       34.24    36.17    36.10       37.10     38.79   37.16  36.42
# Very Unhealthy                    1.61        2.23     1.92     3.58        3.90      4.36    5.25   8.16
# Hazardous                         0.04        0.04     0.03     0.04        0.06      0.08    0.12   0.34


if __name__ == '__main__':
    metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data_4_years()

    # box_plot_series(region_series)
    # stacked_bar(region_series)

    # custom_time_series(country_series)
    grouped_box_month_year(country_series['Bangladesh'])
    # day_night_distribution_monthly(country_series[["Bangladesh"]])

    # # plt.close("all")
    # # sns.set()
    # # # sns.set_style("whitegrid")

    # prepare_color_table()
    # missing_data_heatmap(region_series)
    # # BoxPlotHour(series)
    # # violin_plot_year(country_series)

    # day_night_distribution(country_series)
    # day_night_distribution_monthly(country_series[["Bangladesh"]])
    # PLotlyTimeSeries(country_series)
    # print(region_series.columns)
    # box_plot_week(country_series)

    # print(metaFrame[['Population','avgRead']].corr())
    # popYear = [157977153,159685424,161376708,163046161]
