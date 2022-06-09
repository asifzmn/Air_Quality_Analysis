import pandas as pd
from paths import confirmed, deaths, mobility_path


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
