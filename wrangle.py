import pandas as pd
import numpy as np

import os.path
import requests
import io

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def prep_flight_data():

    if os.path.exists("clean_data.csv") == False:
        df = pd.read_csv("2018.csv")
        top_airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA"]
        for i in top_airports:
            df.loc[df['ORIGIN'] == i, 'is_top'] = True 

        df.is_top = df.is_top.fillna(False)
        df = df[df.is_top]
        df[df.WEATHER_DELAY.isnull()].ARR_DELAY.mean()
        columns_to_drop = ["Unnamed: 27", "is_top", 'CANCELLATION_CODE', 'TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CANCELLED','DIVERTED','AIR_TIME', 'CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME']
        df.drop(columns = columns_to_drop, inplace=True)
        df = df[df.ARR_DELAY.notnull()]
        df = df[df.DEP_DELAY.notnull()]
        df = df.fillna(0.0)
        df.columns = map(str.lower, df.columns)

        # lets add the airline names

        airline_code = pd.read_csv("airline_codes.csv")

        airline_code.drop(columns="Unnamed: 0", inplace=True)
        df = df.merge(airline_code, how="left", left_on="op_carrier", right_on = "Code")
        df.drop(columns="Code", inplace=True)

        df.to_csv("clean_data.csv")

    
    else:
        df = pd.read_csv("clean_data.csv")
        df.drop(columns=["Unnamed: 0"], inplace=True)

    return df

# ------------------------------ #
#          Weather Data          #
# ------------------------------ #

def prep_flight_data_weather():
    df = prep_flight_data()

    df["crs_dep_time"] = df['crs_dep_time'].astype(str).apply(lambda x: x.zfill(4))
    df["fl_datetime"] = df.fl_date.astype(str) + " " + df.crs_dep_time.astype(str)
    df.fl_datetime = pd.to_datetime(df.fl_datetime, format="%Y-%m-%d %H%M")
    df = df.set_index("fl_datetime")
    df.index = df.index.map(lambda t: t.replace(minute=30))

    features_to_keep = ["fl_date", "op_carrier", "op_carrier_fl_num", "crs_dep_time", "origin", "dest", "crs_arr_time", "dep_delay", "arr_delay", "weather_delay", "Airline"]
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

    weather_data.drop(columns=["Unnamed: 0"],inplace=True)

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
    
    if os.path.exists("merged_data.csv") == False:

        df = prep_flight_data_weather()

        weather_data = prep_weather_data()

        weather_data = filter_weather_data(weather_data)

        merged_df = df.merge(weather_data[["Type", "Severity", "unique_id"]], how="left", right_on="unique_id", left_on="unique_id")

        merged_df.Type = merged_df.Type.fillna("Clear")
        merged_df.Severity = merged_df.Severity.fillna("Light")

        merged_df.to_csv("merged_data.csv")

        return merged_df

    else:

        merged_data = pd.read_csv("merged_data.csv")
        return merged_data