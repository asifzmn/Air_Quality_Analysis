import pandas as pd
from data_preparation import get_compressed_save_location, get_metadata, get_series, read_all_granularity_data


def get_compressed_save_location_bd(): return get_compressed_save_location() + 'bd/'


def save_data():
    meta_data, timeseries = get_metadata(), get_series()['2017':'2019']
    meta_data.to_csv('zone_data.csv')
    timeseries.to_csv('pm_time_series.csv')


def filter_country_bd(meta_data, time_series):
    meta_data = meta_data[meta_data.Country == 'Bangladesh']
    time_series = time_series[meta_data.index]
    return meta_data, time_series


def get_bd_data():
    metadata_all, series_all, metadata_region_all, region_series_all, metadata_country_all, \
    country_series_all = read_all_granularity_data()
    metadata, series = filter_country_bd(metadata_all, series_all)
    metadata_region, region_series = filter_country_bd(metadata_region_all, region_series_all)
    metadata_country, country_series = metadata_country_all.loc[["Bangladesh"]], country_series_all[["Bangladesh"]]
    return metadata, series, metadata_region, region_series, metadata_country, country_series


def get_bd_data_4_years():
    metadata, series, metadata_region, region_series, metadata_country, country_series = get_bd_data()
    series_4_year, region_series_4_year, country_series_4_year = series['2018':'2021'], region_series['2018':'2021'], \
                                                                 country_series['2018':'2021']
    return metadata, series_4_year, metadata_region, region_series_4_year, metadata_country, country_series_4_year


def read_bd_meta_data():
    return pd.read_csv(get_compressed_save_location_bd() + "metadata.csv", index_col='index')


def read_bd_series(file_suffix=''):
    return pd.read_csv(get_compressed_save_location_bd() + f"series{file_suffix}.csv", index_col='time',
                       parse_dates=[0])


def read_bd_data(file_suffix=''):
    files_path = get_compressed_save_location_bd()
    metadata = read_bd_meta_data()
    series = read_bd_series(file_suffix)
    metadata_region = pd.read_csv(files_path + "metadata_region.csv", index_col='Region')
    region_series = pd.read_csv(files_path + f"region_series{file_suffix}.csv", index_col='time', parse_dates=[0])
    metadata_country = pd.read_csv(files_path + "metadata_country.csv", index_col='Country')
    country_series = pd.read_csv(files_path + f"country_series{file_suffix}.csv", index_col='time', parse_dates=[0])
    return metadata, series, metadata_region, region_series, metadata_country, country_series


def read_bd_data_interval(start, end):
    metadata, series, metadata_region, region_series, metadata_country, country_series = read_bd_data()
    series_4_year, region_series_4_year, country_series_4_year = series[start:end], \
                                                                 region_series[start:end], country_series[start:end]
    return metadata, series_4_year, metadata_region, region_series_4_year, metadata_country, country_series_4_year


def read_bd_data_4_years():
    return read_bd_data(file_suffix='_4_year')
