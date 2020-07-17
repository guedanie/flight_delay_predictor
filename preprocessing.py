import pandas as pd
import numpy as np

import os.path
import requests
import io

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingClassifier

import wrangle
import model

def read_csv():
    df = wrangle.prep_flight_data()

    return df

def to_date_time(df):
    df.fl_date = pd.to_datetime(df.fl_date)
    df["day_of_week"] = df.fl_date.dt.dayofweek
    df["month"] = df.fl_date.dt.month
    df["avg_month_delay"] = df.groupby("month").arr_delay.transform("mean")
    df["avg_day_of_week_delay"] = df.groupby("day_of_week").arr_delay.transform("mean")
    return df

def create_new_features(df):
    df["airport_avg_delay"] = df.groupby("origin").arr_delay.transform("mean")
    df["dest_airport_avg_delay"] = df.groupby("dest").arr_delay.transform("mean")
    df["carrier_avg_delay"] = df.groupby("op_carrier").arr_delay.transform("mean")
    df["observation"] = df.fl_date.astype(str) + "_" + df.op_carrier + "_" + df.op_carrier_fl_num.astype(str)
    
    df["crs_dep_time_dt"] = df['crs_dep_time'].astype(str).apply(lambda x: x.zfill(4))
    df["crs_dep_time_dt"] = pd.to_datetime(df.crs_dep_time_dt, format= "%H%M")
    df["dep_time_mean_delay"] = df.groupby("crs_dep_time_dt").arr_delay.transform("mean")
    

    return df


def create_target_variable(df):
    df['is_delay'] = df['arr_delay'].apply(lambda x: True if x > 0 else False)
    return df

def split_data(df_modeling):

    train, test = train_test_split(df_modeling, random_state = 123, train_size= 0.75)
    train, validate = train_test_split(train, random_state = 123, train_size= 0.75)

    return train, validate, test

def return_values(scaler, train, validate, test):
    '''
    Helper function used to updated the scaled arrays and transform them into usable dataframes
    '''
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

# Linear scaler
def min_max_scaler(train,validate, test):
    '''
    Helper function that scales that data. Returns scaler, as well as the scaled dataframes
    '''
    scaler = MinMaxScaler().fit(train)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, train, validate, test)
    return scaler, train_scaled, validate_scaled, test_scaled

def mvp_modeling_prep(modeling = False, features_for_modeling=[], target_variable=''):
    df = wrangle.prep_flight_data()
    df = to_date_time(df)
    df = create_new_features(df)
    df = create_target_variable(df)

    features_for_modeling += ["observation"]
    features_for_modeling += [target_variable]

    df_modeling = df[features_for_modeling]

    df_modeling = df_modeling.set_index("observation")

    if modeling == False:
        return df_modeling

    else:

        train, validate, test = split_data(df_modeling)

        X_train = train.drop(columns=target_variable)
        y_train = train[target_variable]
        X_validate = validate.drop(columns=target_variable)
        y_validate = validate[target_variable]
        X_test = test.drop(columns=target_variable)
        y_test = test[target_variable]
        
        scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)


        return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test


#############################################################################
#                       Weather data preprocessing                          #
#############################################################################


def prep_flight_data_weather():
    df = wrangle.prep_flight_data()
    
    df["crs_dep_time"] = df['crs_dep_time'].astype(str).apply(lambda x: x.zfill(4))
    df["fl_datetime"] = df.fl_date.astype(str) + " " + df.crs_dep_time.astype(str)
    df.fl_datetime = pd.to_datetime(df.fl_datetime, format="%Y-%m-%d %H%M")
    df = df.set_index("fl_datetime")
    df.index = df.index.map(lambda t: t.replace(minute=30))

    features_to_keep = ["fl_date", "op_carrier", "op_carrier_fl_num", "crs_dep_time", "origin", "dest", "dep_delay", "arr_delay", "weather_delay", "Airline"]
    df = df[features_to_keep]
    df = df.reset_index()

    df["unique_id"] = df.fl_datetime.astype(str) + "_" + df.origin

    df = df.sort_values(["origin", "fl_datetime", "op_carrier"])

    return df


def prep_weather_data():
    weather_data = pd.read_csv("US_WeatherEvents_2016-2019.csv")
    
    # we only wand data for the following cities
    top_airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA"]
    
    airport_codes = pd.read_csv("airport_codes.csv")
    airport_codes = airport_codes[airport_codes.Country == "United States"]

    weather_data = weather_data.merge(airport_codes, how="left", left_on="AirportCode", right_on="ICAO")

    top_airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA"]

    for i in top_airports:
        weather_data.loc[weather_data['IATA'] == i, 'is_top'] = True 
        
    weather_data.is_top = weather_data.is_top.fillna(False)
    weather_data = weather_data[weather_data.is_top]
    
    weather_data["StartTime(UTC)"] = pd.to_datetime(weather_data["StartTime(UTC)"])

    weather_data = weather_data.set_index("StartTime(UTC)")

    weather_data = weather_data["2018"]

    return weather_data

def filter_weather_data(weather_data):
    
    weather = pd.DataFrame()
    unique_cities = weather_data.City_y.unique()
    # Given that not all cities have the same starting point, we need to basically individually filter 
    # for each city, reset their index and impude the missing values. The biggest issue is the fact that
    # only precipitation data is reported, so normal weather conditions are not present. As 
    # such we inpude them here
    for city in unique_cities:
        city_data = weather_data[weather_data.City_y == city]
        r = pd.date_range(start=city_data.index.min(), end=city_data.index.max(), freq="H")
        city_data = city_data.reindex(r, copy=True)
        city_data["EndTime(UTC)"] = city_data["EndTime(UTC)"].fillna(method="ffill")
        for i in range(city_data["EndTime(UTC)"].unique().shape[0]):
            date = city_data["EndTime(UTC)"].unique()[i]
            mask = (city_data["EndTime(UTC)"] == date) & (city_data.index >= date)
            city_data["Type"] = city_data["Type"].mask(mask, city_data["Type"].fillna("Clear"))
        mask = city_data.Type != "Clear"
        city_data = city_data.mask(mask, city_data.fillna(method="ffill"))
        city_data = city_data.fillna(method="ffill")  
        
        weather = pd.concat([weather, city_data])

        features_to_keep = ["Type", "Severity", "TimeZone", "IATA", "City_y", "State"]

    weather = weather[features_to_keep]

    weather = weather.sort_values(["City_y", "State"])

    weather.rename(columns={"City_y":"city"}, inplace=True)

    # We will now normalize the data by changing all time stamps to :30 minutes, to makes it easier to merge
    # with our flights df 

    weather.index = weather.index.map(lambda t: t.replace(minute=30))

    time_zones = weather.TimeZone.unique()
    weather_df = pd.DataFrame()
    for zone in time_zones:
        mask = weather.TimeZone == zone
        time_zone_df = weather[mask].tz_localize('utc').tz_convert(zone).tz_localize(None)
        weather_df = pd.concat([weather_df, time_zone_df])

    weather_df["unique_id"] = weather_df.index.astype(str) + "_" + weather_df.IATA
    weather_df = weather_df.reset_index().rename(columns={"index": "date"})
    
    return weather_df

def merge_flight_weather_data():
    df = prep_flight_data_weather()

    weather_data = prep_weather_data()

    weather_data = filter_weather_data(weather_data)

    merged_df = df.merge(weather_data[["Type", "Severity", "unique_id"]], how="left", right_on="unique_id", left_on="unique_id")

    merged_df.Type = merged_df.Type.fillna("Clear")
    merged_df.Severity = merged_df.Severity.fillna("Light")

    return merged_df

def weather_modeling_prep(modeling=False, features_for_modeling=[], target_variable=''):
    merged_df = merge_flight_weather_data()
    merged_df = to_date_time(merged_df)
    merged_df = create_new_features(merged_df)
    merged_df = create_target_variable(merged_df)
    
    # add weather features
    merged_df["avg_weather_delay"] = merged_df.groupby("Type").arr_delay.transform("mean")
    merged_df["type_severity"] = merged_df.Type + "_" + merged_df.Severity
    merged_df["avg_type_severity"] = merged_df.groupby("type_severity").arr_delay.transform("mean")

    if modeling == False:
        return merged_df

    else:
        
        features_for_modeling += ["observation"]
        features_for_modeling += [target_variable]

        merged_df = merged_df[features_for_modeling]
        merged_df = merged_df.set_index("observation")

        train, validate, test = split_data(merged_df)

        X_train = train.drop(columns=target_variable)
        y_train = train[target_variable]
        X_validate = validate.drop(columns=target_variable)
        y_validate = validate[target_variable]
        X_test = test.drop(columns=target_variable)
        y_test = test[target_variable]

        scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)

        return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test