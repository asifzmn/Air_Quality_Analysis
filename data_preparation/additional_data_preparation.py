import pandas as pd
from paths import confirmed, deaths, mobility_path, sun_rise_set_time_2017_2020


def prepare_covid_dataset():
    covid_period_start, covid_period_end = '2020-03', '2022-02'
    confirmed_ts_bd, deaths_ts_bd = pd.read_csv(confirmed), pd.read_csv(deaths)

    confirmed_ts_bd = confirmed_ts_bd[confirmed_ts_bd['Country/Region'] == 'Bangladesh'].iloc[:, 4:].T
    deaths_ts_bd = deaths_ts_bd[deaths_ts_bd['Country/Region'] == 'Bangladesh'].iloc[:, 4:].T

    covid_cases = pd.concat((confirmed_ts_bd, deaths_ts_bd), axis=1)
    covid_cases.columns = ['confirmed', 'deaths']
    covid_cases.index = pd.to_datetime(covid_cases.index)
    covid_cases = covid_cases.diff().fillna(0)
    return covid_cases.loc[covid_period_start:covid_period_end], covid_period_start, covid_period_end


def get_mobility_data():
    mobility = pd.concat(
        [pd.read_csv(mobility_path + f'{year}_BD_Region_Mobility_Report.csv') for year in [2020, 2021, 2022]])
    mobility = mobility.iloc[:, 8:].set_index('date')
    mobility.index = pd.to_datetime(mobility.index)
    mobility.columns = mobility.columns.str.split('_percent').str[0]
    return mobility


def get_diurnal_period():
    sun_time = pd.read_csv(sun_rise_set_time_2017_2020, sep='\t')
    sun_time['Sunrise_date'] = sun_time['Date '] + ' ' + sun_time['Sunrise ']
    sun_time['Sunset_date'] = sun_time['Date '] + ' ' + sun_time['Sunset ']
    sun_time = sun_time[['Sunrise_date', 'Sunset_date']].apply(pd.to_datetime).apply(lambda x: x.dt.round('H'))
    sun_time_matrix = sun_time.apply(
        lambda x: pd.Series(['day' if x.Sunrise_date.hour <= i <= x.Sunset_date.hour else 'night' for i in range(24)]),
        axis=1)
    sun_time_series = sun_time_matrix.stack().reset_index(drop=True)

    print(pd.date_range('2017', '2021', freq='H')[:-1])
    sun_time_series.index = pd.date_range('2017', '2021', freq='H')[:-1]
    sun_time_series.name = 'diurnal_name'
    return sun_time_series
