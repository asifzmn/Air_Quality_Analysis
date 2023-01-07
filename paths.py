import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

berkeley_earth_dataset_url = 'http://berkeleyearth.lbl.gov/air-quality/maps/cities/'

# aq_directory = '/home/asif/Datasets/AQ Dataset/'  # change this main dataset location Desktop
aq_directory = '/home/asif/Data/Dataset/AQ Dataset/'  # change this main dataset location Laptop
# aq_directory = getcwd() + '/Project Data/'

aq_directory_local = 'Files/'

berkeley_earth_data = aq_directory + 'Berkeley Earth Data/'
raw_data_path = berkeley_earth_data + 'raw/'
zone_data_path = berkeley_earth_data + 'zones/'
berkeley_earth_data_prepared = berkeley_earth_data + 'prepared/'

berkeley_earth_data_compressed = berkeley_earth_data + 'compressed/'

meteoblue_data_path = aq_directory + 'Meteoblue Scrapped Data/'
meteoblue_data_path_2019 = aq_directory + 'Meteoblue Data/MeteoBlue Data 2019'
wunderground_data_path = aq_directory + 'Wunderground Data/'
wunderground_data_path_compressed = aq_directory + 'Wunderground Data/Compressed/'

sun_rise_set_time_2017_2020 = aq_directory + 'Day Night Time/sun_rise_set_time_2017_2020.csv'

mobility_path = aq_directory + 'Mobility/Region_Mobility_Report_CSVs/'

covid_data = aq_directory + 'Covid_Data/'
confirmed = covid_data + 'time_series_covid19_confirmed_global.csv'
deaths = covid_data + 'time_series_covid19_deaths_global.csv'
recovered = covid_data + 'time_series_covid19_recovered_global.csv'

comp_file = '/home/az/Desktop/AQ Overall Comparision.csv'
