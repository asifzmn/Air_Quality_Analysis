from data_preparation import *
from exploration import *


def district_aggregation(x, time_series, zone_dis):
    zone_district_group = (zone_dis[zone_dis.District == x.name].Zone.to_list())
    single_district_series = (time_series[zone_district_group].mean(axis=1))
    single_district_series.name = x.name
    return single_district_series


if __name__ == '__main__':
    zone_district = pd.read_csv('/home/asif/Desktop/berkelyearth_bd_zone_district.csv')

    # dataSummaries = LoadData()
    time_series = get_series()
    metaFrame = get_metadata()

    time_series = time_series["2019":"2020"]
    district_series = zone_district.groupby("District").apply(district_aggregation, timeseies=time_series,
                                                              zone_dis=zone_district).T
    # print(district_series)
    # SimpleTimeseries(district_series)
    print(district_series.describe().sort_values('50%', axis=1).columns)
    # print(district_series.describe().sort_values('50%', axis=1).to_string())
