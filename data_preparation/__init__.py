from paths import *
from os import listdir, getcwd
from os.path import join, isfile
from urllib.request import urlopen
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

metadata_attributes = ['Country', 'Zone', 'Region', 'Population', 'Latitude', 'Longitude']
rename_dict = {'Azimpur': 'Dhaka', 'Tungi': 'Tongi'}


def get_common_id(id=3): return ['study_area', 'SouthAsianCountries', 'allbd', 'bd_and_neighbours', 'usa_west'][id]


def get_save_location(): return berkeley_earth_data_prepared + get_common_id() + '/'


def get_compressed_save_location(): return berkeley_earth_data_compressed + get_common_id() + '/'


def get_zones_info(): return pd.read_csv(zone_data_path + get_common_id() + '.csv').sort_values('Zone', ascending=True)


def get_category_info():
    color_scale = np.array(['#46d246', '#ffff00', '#ffa500', '#ff0000', '#800080', '#6a2e20'])
    l_color_scale = np.array(['#a2e8a2', '#ffff7f', '#ffd27f', '#ff7f7f', '#ff40ff', '#d38370'])
    d_color_scale = np.array(['#1b701b', '#7f7f00', '#7f5200', '#7f0000', '#400040', '#351710'])
    category_name = np.array(
        ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    aq_scale = np.array([0, 12.1, 35.5, 55.5, 150.5, 250.5, 500])
    return color_scale, category_name, aq_scale


def prepare_color_table():
    color_scale, category_name, aq_scale = get_category_info()
    range_str = [str(a) + " - " + str(b) for a, b in zip(aq_scale, aq_scale[1:])]
    data = {'category_name': category_name, 'color_scale': color_scale, 'range_str': range_str}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Category</b>", "<b>Concentration Range (&#956;gm<sup>-3</sup>) </b>"],
            line_color='grey', fill_color='silver',
            align='center', font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[df.category_name, df.range_str],
            line_color='grey', fill_color=[df.color_scale],
            align='center', font=dict(color='black', size=9)
        ))
    ])
    fig.update_layout(width=333)
    fig.show()


def web_crawl():
    for idx, zone_info in get_zones_info().iterrows():
        zone_file = join(raw_data_path + get_common_id(), zone_info['Division'] + '_' + zone_info['Zone'] + '.txt')
        url = f'{berkeley_earth_dataset_url}{zone_info["Country"]}/{zone_info["Division"]}/{zone_info["Zone"]}.txt'
        print(url)
        data = [line.decode('unicode_escape')[:-1] for line in urlopen(url)]
        with open(zone_file, 'w') as file: file.write('\n'.join(map(str, data)))


def make_header_only(meta_data):
    meta_data = meta_data.assign(Country='Bangladesh')
    meta_data = meta_data.reset_index()[['Country', 'Division', 'Zone']]
    meta_data.to_csv('headerFile.csv', index=False)


def read_all_bd_zones_and_make_header():
    all_data_path = berkeley_earth_data + 'total/'
    allFiles = [all_data_path + f for f in listdir(all_data_path)]
    all_district_meta_data = []

    for file in allFiles:
        data = read_file_as_text(file)
        all_district_meta_data.append(np.array([d.split(':')[1][1:-1] for d in data[:9]]))

    all_district_meta_data = pd.DataFrame(np.array(all_district_meta_data)[:, [4, 2]],
                                          columns=['Division', 'Zone']).assign(
        Country='Bangladesh')
    all_district_meta_data = all_district_meta_data[['Country', 'Division', 'Zone']]
    all_district_meta_data.to_csv(berkeley_earth_data + 'zones/allbd.csv', index=False)


def handle_mislabeled_duplicates(series):
    duplicates = (series.duplicated(subset=['time'], keep='last'))
    duplicated_series = series[duplicates]
    firsts = series.loc[duplicated_series.index].time
    lasts = series.loc[duplicated_series.index + 1].time + timedelta(hours=1)
    firsts, lasts = firsts.reset_index(), lasts.reset_index()
    print(firsts, lasts)
    # print((firsts.time == lasts.time).value_counts())
    # print(duplicated_series.time.dt.year.unique())
    # print(duplicated_series.to_string())
    # print(series.duplicated(keep=False).value_counts())


def read_file_as_text(file):
    return str(open(file, 'rb').read().decode('unicode_escape')).split("\n")


def data_cleaning_and_preparation():
    zone_reading, zone_metadata = [], []

    for idx, row in get_zones_info().iterrows():
        # file = join(raw_data_path + get_common_id(), row['Zone'] + '.txt')
        file = join(raw_data_path + get_common_id(), row['Division'] + '_' + row['Zone'] + '.txt')
        meta_data_list = read_file_as_text(file)
        zone_metadata.append(pd.Series([d.split(':')[1][1:-1] for d in meta_data_list[:9]]).loc[[0, 2, 4, 5, 6, 7]])

        series = pd.read_csv(file, skiprows=10, sep='\t', header=None, usecols=[0, 1, 2, 3, 4])
        series.columns = ['Year', 'Month', 'Day', 'Hour', 'PM25']
        series['time'] = pd.to_datetime(series[series.columns[:4]])
        series = series[['time', 'PM25']]

        # series = series.groupby('time').PM25.mean().reset_index()  # usa
        # series = series.set_index('time').reindex(pd.date_range('2004-01-01', '2022-01-01', freq='h')[:-1])  # usa

        # if get_common_id() != 'SouthAsianCountries' and get_common_id() != 'bd_and_neighbours':
        if get_common_id() in ['study_area', 'allbd']:
            duplicates = (series.duplicated(subset=['time'], keep='first'))
            series.loc[duplicates, 'time'] = series.loc[duplicates, 'time'] + timedelta(hours=1)
        else:
            series = series.groupby('time').PM25.mean().reset_index()

        # print(row)
        series = series.set_index('time').reindex(pd.date_range('2017-01-01', '2022-05-01', freq='h')[:-1])
        series.columns = [zone_metadata[-1].iloc[1]]
        zone_reading.append(series)

    metadata = pd.concat(zone_metadata, axis=1).T
    metadata.columns = metadata_attributes
    metadata = metadata.replace({'Mymensingh Division': 'Mymensingh', 'Rangpur Division': 'Rangpur'})  # BD Regions
    metadata = metadata.sort_values('Zone').set_index('Zone')
    metadata[metadata_attributes[3:]] = metadata[metadata_attributes[3:]].apply(pd.to_numeric).round(5)
    metadata = metadata.reset_index()
    metadata.index = metadata.Zone + "_" + metadata.Region
    metadata.index.name = 'index'
    metadata.to_csv(get_save_location() + 'metadata.csv')

    time_series = pd.concat(zone_reading, axis=1)
    time_series.index.name = 'time'
    time_series.columns = metadata.index
    time_series.to_csv(get_save_location() + 'time_series.csv')


def clip_missing_prone_values(metadata, series):
    region_missing_counts = metadata.groupby('Region').apply(
        lambda regional_zone: series[regional_zone.index].isna().all(axis=1)).sum(axis=1).sort_values()
    region_valid_data = region_missing_counts[region_missing_counts < 10000].index
    metadata = metadata[metadata.Region.isin(region_valid_data)]
    series = series[metadata.index]
    return region_missing_counts, metadata, series


def prepare_region_and_country_series(series, metadata):
    def process_by_region(divisional_zone, series):
        return series[divisional_zone.index].mean(axis=1)

    # used for bd_and_neighbors to remove zones east of Bangladesh
    # metadata = metadata[(metadata.Country != 'Myanmar') & (metadata.Region != 'Tripura')]

    metadata_region_group = metadata.groupby('Region')
    metadata_country_group = metadata.groupby('Country')

    region_series = metadata_region_group.apply(process_by_region, series=series).T.round(2)
    country_series = metadata_country_group.apply(process_by_region, series=series).T.round(2)

    metadata_region = metadata_region_group.agg({
        'Country': lambda x: x.sample(), 'Population': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})

    metadata_country = metadata_country_group.agg({
        'Population': 'sum', 'Latitude': 'mean', 'Longitude': 'mean', })

    metadata_region['Count'] = metadata_region_group.Region.count()
    metadata_country['Count'] = metadata_country_group.Country.count()

    return region_series, metadata_region, country_series, metadata_country


def get_series():
    return pd.read_csv(get_save_location() + 'time_series.csv', index_col='time', parse_dates=[0])
    # return pd.read_csv(get_save_location() + 'time_series.csv', index_col='time', parse_dates=[0]).rename(
    #     columns=rename_dict).sort_index(axis=1)


def get_metadata():
    return pd.read_csv(get_save_location() + 'metadata.csv', index_col='index', parse_dates=[0])
    # return pd.read_csv(get_save_location() + 'metadata.csv', index_col='Zone', parse_dates=[0]).rename(
    #     index=rename_dict).sort_index(axis=0)


# def read_region_and_country_series_ignore_myanmar():
#     region_series, metadata_region, country_series, metadata_country = read_region_and_country_series()

def save_region_and_country_series(region_series, country_series, metadata_region, metadata_country):
    region_series.to_csv("region_series.csv")
    country_series.to_csv("country_series.csv")
    metadata_region.to_csv("metadata_region.csv")
    metadata_country.to_csv("metadata_country.csv")


def save_all_granularity_data():
    metadata_all_with_heavy_missing, series_all_with_heavy_missing = get_metadata(), get_series()
    # division_missing_counts, metadata_all, series_all = clip_missing_prone_values(metadata_all_with_heavy_missing,
    #                                                                               series_all_with_heavy_missing)
    metadata_all, series_all = metadata_all_with_heavy_missing, series_all_with_heavy_missing
    region_series_all, metadata_region_all, country_series_all, metadata_country_all = \
        prepare_region_and_country_series(series_all, metadata_all)
    save_region_and_country_series(region_series_all, country_series_all, metadata_region_all, metadata_country_all)


def read_region_and_country_series():
    files_path = get_compressed_save_location()
    region_series = pd.read_csv(files_path + "region_series.csv", index_col='time', parse_dates=[0])
    metadata_region = pd.read_csv(files_path + "metadata_region.csv", index_col='Region')
    country_series = pd.read_csv(files_path + "country_series.csv", index_col='time', parse_dates=[0])
    metadata_country = pd.read_csv(files_path + "metadata_country.csv", index_col='Country')
    return region_series, metadata_region, country_series, metadata_country


def read_all_granularity_data():
    metadata_all_with_heavy_missing, series_all_with_heavy_missing = get_metadata(), get_series()
    # division_missing_counts, metadata_all, series_all = clip_missing_prone_values(metadata_all_with_heavy_missing,
    #                                                                               series_all_with_heavy_missing)
    metadata_all, series_all = metadata_all_with_heavy_missing, series_all_with_heavy_missing

    region_series_all, metadata_region_all, country_series_all, metadata_country_all = read_region_and_country_series()
    return metadata_all, series_all, metadata_region_all, region_series_all, metadata_country_all, country_series_all


if __name__ == '__main__':
    # web_crawl()
    # data_cleaning_and_preparation()
    # save_all_granularity_data()

    # timeseries = get_series()
    # meta_data = get_metadata()

    exit()
