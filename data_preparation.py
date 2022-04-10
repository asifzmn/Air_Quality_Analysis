from datetime import datetime, timedelta
from os import listdir, getcwd
from os.path import join, isfile
from urllib.request import urlopen
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from paths import *

metadata_attributes = ['Zone', 'Division', 'Population', 'Latitude', 'Longitude']
rename_dict = {'Azimpur': 'Dhaka', 'Tungi': 'Tongi'}


def get_common_id(id=1): return ['study_area', 'SouthAsianCountries', 'allbd'][id]


def get_save_location(): return berkeley_earth_data_prepared + get_common_id() + '/'


def get_zones_info(): return pd.read_csv(zone_data_path + get_common_id() + '.csv').sort_values('Zone', ascending=True)


def get_category_info():
    colorScale = np.array(['#46d246', '#ffff00', '#ffa500', '#ff0000', '#800080', '#6a2e20'])
    lcolorScale = np.array(['#a2e8a2', '#ffff7f', '#ffd27f', '#ff7f7f', '#ff40ff', '#d38370'])
    dcolorScale = np.array(['#1b701b', '#7f7f00', '#7f5200', '#7f0000', '#400040', '#351710'])
    categoryName = np.array(
        ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    AQScale = np.array([0, 12.1, 35.5, 55.5, 150.5, 250.5, 500])
    return colorScale, categoryName, AQScale


def make_header_only(meta_data):
    meta_data = meta_data.assign(Country='Bangladesh')
    meta_data = meta_data.reset_index()[['Country', 'Division', 'Zone']]
    meta_data.to_csv('headerFile.csv', index=False)


def read_all_bd_zones_and_make_header():
    alldatapath = berkeley_earth_data + 'total/'
    allFiles = [alldatapath + f for f in listdir(alldatapath)]
    allDistrictMetaData = []

    for file in allFiles:
        data = read_file_as_text(file)
        allDistrictMetaData.append(np.array([d.split(':')[1][1:-1] for d in data[:9]]))

    allDistrictMetaData = pd.DataFrame(np.array(allDistrictMetaData)[:, [4, 2]], columns=['Division', 'Zone']).assign(
        Country='Bangladesh')
    allDistrictMetaData = allDistrictMetaData[['Country', 'Division', 'Zone']]
    allDistrictMetaData.to_csv(berkeley_earth_data + 'zones/allbd.csv', index=False)


def handle_mislabeled_duplicates(series):
    duplicates = (series.duplicated(subset=['time'], keep='last'))
    duplicated_series = series[duplicates]
    firsts = series.loc[duplicated_series.index].time
    lasts = series.loc[duplicated_series.index + 1].time + timedelta(hours=1)
    firsts, lasts = firsts.reset_index(), lasts.reset_index()

    # print((firsts.time == lasts.time).value_counts())
    # print(duplicated_series.time.dt.year.unique())
    # print(duplicated_series.to_string())
    # print(series.duplicated(keep=False).value_counts())


def web_crawl():
    for idx, zone_info in get_zones_info().iterrows():
        zone_file = join(raw_data_path + get_common_id(), zone_info['Zone'] + '.txt')
        url = f'{dataset_url}{zone_info["Country"]}/{zone_info["Division"]}/{zone_info["Zone"]}.txt'
        print(url)
        data = [line.decode('unicode_escape')[:-1] for line in urlopen(url)]
        with open(zone_file, 'w') as file: file.write('\n'.join(map(str, data)))


def read_file_as_text(file):
    return str(open(file, 'rb').read().decode('unicode_escape')).split("\n")


def data_cleaning_and_preparation():
    zone_reading, zone_metadata = [], []

    for idx, row in get_zones_info().iterrows():
        file = join(raw_data_path + get_common_id(), row['Zone'] + '.txt')
        meta_data_list = read_file_as_text(file)
        zone_metadata.append(pd.Series([d.split(':')[1][1:-1] for d in meta_data_list[:9]]).loc[[2, 4, 5, 6, 7]])

        series = pd.read_csv(file, skiprows=10, sep='\t', header=None, usecols=[0, 1, 2, 3, 4])
        series.columns = ['Year', 'Month', 'Day', 'Hour', 'PM25']
        series['time'] = pd.to_datetime(series[series.columns[:4]])
        series = series[['time', 'PM25']]

        if get_common_id() != 'SouthAsianCountries':
            duplicates = (series.duplicated(subset=['time'], keep='first'))
            series.loc[duplicates, 'time'] = series.loc[duplicates, 'time'] + timedelta(hours=1)
        else:
            series = series.groupby('time').PM25.mean().reset_index()

        series = series.set_index('time').reindex(pd.date_range('2017-01-01', '2021-01-01', freq='h')[:-1])
        series.columns = [zone_metadata[-1].iloc[0]]
        zone_reading.append(series)

    metadata = pd.concat(zone_metadata, axis=1).T
    metadata.columns = metadata_attributes
    metadata = metadata.sort_values('Zone').set_index('Zone')
    metadata[metadata_attributes[2:]] = metadata[metadata_attributes[2:]].apply(pd.to_numeric).round(5)
    metadata.to_csv(get_save_location() + 'metadata.csv')

    time_series = pd.concat(zone_reading, axis=1)
    time_series.index.name = 'time'
    time_series.to_csv(get_save_location() + 'time_series.csv')


def get_series():
    # return pd.read_csv(get_save_location() + 'time_series.csv',index_col='time',parse_dates=[0])
    return pd.read_csv(get_save_location() + 'time_series.csv', index_col='time', parse_dates=[0]).rename(
        columns=rename_dict).sort_index(axis=1)


def get_metadata():
    # return pd.read_csv(get_save_location() + 'metadata.csv', index_col='Zone', parse_dates=[0])
    return pd.read_csv(get_save_location() + 'metadata.csv', index_col='Zone', parse_dates=[0]).rename(
        index=rename_dict).sort_index(axis=0)


def get_diurnal_period():
    sun_time = pd.read_csv(aq_directory + 'sun_rise_set_time_2017_2020.csv', sep='\t')
    sun_time['Sunrise_date'] = sun_time['Date '] + ' ' + sun_time['Sunrise ']
    sun_time['Sunset_date'] = sun_time['Date '] + ' ' + sun_time['Sunset ']
    sun_time = sun_time[['Sunrise_date', 'Sunset_date']].apply(pd.to_datetime).apply(lambda x: x.dt.round('H'))
    sun_time_matrix = sun_time.apply(
        lambda x: pd.Series(['day' if x.Sunrise_date.hour <= i <= x.Sunset_date.hour else 'night' for i in range(24)]),
        axis=1)
    sun_time_series = sun_time_matrix.stack().reset_index(drop=True)
    print(sun_time_series)

    print(pd.date_range('2017', '2021', freq='H')[:-1])
    sun_time_series.index = pd.date_range('2017', '2021', freq='H')[:-1]
    sun_time_series.name = 'diurnal_name'
    return sun_time_series


if __name__ == '__main__':
    web_crawl()
    # data_cleaning_and_preparation()
    timeseies = get_series()
    meta_data = get_metadata()

    # print(metaFrame[['Population','avgRead']].corr())
    # popYear = [157977153,159685424,161376708,163046161]

    exit()
