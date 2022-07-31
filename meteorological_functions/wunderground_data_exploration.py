import plotly.express as px
import plotly.graph_objects as go
from data_preparation import *
from meteorological_functions.wunderground_data_preparation import prepare_data, text_to_angle
from data_preparation import get_metadata, get_series, clip_missing_prone_values, read_region_and_country_series

# from visualization import missing_data_heatmap

if __name__ == '__main__':
    region_series, metadata_region, country_series, metadata_country = read_region_and_country_series()
    reading_data = region_series["2019":"2021"].Dhaka
    reading_data.name = "Reading"

    raw_data = prepare_data()

    # print(raw_data.Condition.value_counts())
    # ConditionStats(raw_data)
    # ModelPreparation(raw_data, reading_data)
    # FactorAnalysis(raw_data, reading_data)
    # VectorAnalysis()
