# Air_Quality_Analysis

There are several python pakages which are the representation each phase of our experiments.

## scrapers
Start with this module. Berkeley Earth air quality data and other meteorological data can be collected from here. 

## data_preparation
This module contains preparation of berkeley earth dataset.

## visualization
Various spatio-temporal visualization modules.

## meteorological_functions
This module is more complex in design due to fact that there are meteorological data from two sources and multiple atmospheric factors account for three dimensional time series data (time, location, factor). 
First Meteoblue data for 2019 only read for individual zonal files. Then some simple cleaning and processing was done and exported into a single file of Xarray. Xarray works somewhat similar to Pandas. For data reliability and computational efficiency the data was averaged by regions just like performed for pm data

## cross_correlation
Computing pollution effect of one zone to another zone by wind transportation 

## covid_mobility
Effect of mobility changes on air qulaity

## forecasting_model
The forecasting model can be found at Google Colab.