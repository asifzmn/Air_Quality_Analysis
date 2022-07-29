import plotly.express as px
import plotly.graph_objects as go
from data_preparation import *
from meteorological_functions.wunderground_data_preparation import prepare_data, text_to_angle
from visualization import missing_data_heatmap


def VectorAnalysis():
    timeSeries = prepare_data()
    print(timeSeries.sample(15).to_string())

    timeSeries['wind_angle'] = timeSeries.apply(lambda x: text_to_angle.get(x.Wind), axis=1)
    print(timeSeries['wind_angle'].resample('M').agg(pd.Series.mode))
    # print(timeSeries['wind_angle'].resample('M').apply(lambda x: x.value_counts().iloc[:3]))
    # print(timeSeries.Wind.value_counts())
    # print(timeSeries.Wind.value_counts().sort_index())


if __name__ == '__main__':
    from data_preparation import get_metadata, get_series, clip_missing_prone_values, \
        prepare_region_and_country_series

    series_with_heavy_missing, metadata_with_heavy_missing = get_series(), get_metadata()
    division_missing_counts, metadata, series = clip_missing_prone_values(metadata_with_heavy_missing,
                                                                          series_with_heavy_missing)
    region_series, metadata_region, country_series, metadata_country = prepare_region_and_country_series(series,
                                                                                                         metadata)

    reading_data = region_series["2019":"2021"].Dhaka
    reading_data.name = "Reading"

    raw_data = prepare_data()

    # print(raw_data.Condition.value_counts())
    # ConditionStats(raw_data)
    # ModelPreparation(raw_data, reading_data)
    # FactorAnalysis(raw_data, reading_data)
    # VectorAnalysis()
