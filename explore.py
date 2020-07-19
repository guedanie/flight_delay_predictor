import pandas as pd
import numpy as np

import os.path
import requests
import io

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingClassifier

import wrangle
import model

def create_weather_delay_barplot(weather_df):
    plt.figure(figsize=(15,5))
    sns.barplot(data=weather_df,x="Type", y="arr_delay", ci=False)
    plt.title("What type of precipitation leads to most delays?")
    plt.ylabel("Number of minutes delayed")
    plt.xlabel("Type of Weather")

def create_weather_delay_boxplot(weather_df):
    plt.figure(figsize=(15,5))
    sns.boxplot(data=weather_df,x="Type", y="arr_delay")
    plt.title("What type of precipitation leads to most delays?")
    plt.ylabel("Number of minutes delayed")
    plt.xlabel("Type of Weather")

def weather_severity_bar_graph(weather_df):

    plt.figure(figsize=(15,5))
    sns.barplot(data=weather_df, y="arr_delay", x="Severity")
    plt.title("Does weather severity play a role in delays?")

def delay_by_severity(weather_df):
    severity = weather_df.Severity.unique()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8))

    for count, i in enumerate(severity):
        
        data = weather_df[weather_df.Severity == i]
        plt.subplot(3, 3, count + 1)
        sns.barplot(data=data, x="Type", y="arr_delay", ci=False)
        plt.title(f"{i}")
        fig.tight_layout()
        
    plt.show()
            

def data_distribution(df):
    # Is arr delay evenly distributed?

    plt.figure(figsize=(15,5))
    sns.distplot(df.arr_delay)
    plt.title("Is arrival delays evenly distributed?")
    plt.ylabel("Frequency")
    plt.xlabel("Minutes Delayed")

def delay_by_airline_minutes(df):
    # Delays in minutes
    df.groupby("op_carrier").arr_delay.mean().sort_values().plot.bar(figsize=(15,5))
    plt.title("Which airline is delayed the most?")
    plt.ylabel("Minutes Delayed")
    plt.xlabel("Airline Code")

def delay_by_airline_flights(df):
    # Delays by flights (count)

    df[df.arr_delay > 0].groupby("op_carrier").arr_delay.count().sort_values().plot.bar(figsize=(15,5))
    plt.title("Which airline has the most delays?")
    plt.ylabel("Number of Flights")
    plt.xlabel("Airline Code")

def delay_by_airport_minutes(df):
    # Delays in minutes
    df.groupby("origin").arr_delay.mean().sort_values().plot.bar(figsize=(15,5))
    plt.title("Which airport experiences the most delays, in minutes?")
    plt.ylabel("Minutes Delayed")
    plt.xlabel("Airline Code")

def delay_by_airport_flights(df):
    df[df.arr_delay > 0].groupby("origin").arr_delay.count().sort_values().plot.bar(figsize=(15,5))
    plt.title("Which airport has the most number of flights delayed?")
    plt.ylabel("Number of Flights")
    plt.xlabel("Airline Code")

def most_common_reason_for_delay(df):
    df.iloc[:,12:17].mean().sort_values().plot.bar(figsize=(15,5))
    plt.title("What is the most common reason for a delay?")
    plt.ylabel("Avg minutes delayed")
    plt.xlabel("Reason for delay")

def delays_by_destination(df):
    top_airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA"]
    for i in top_airports:
        df.loc[df['dest'] == i, 'is_top'] = True
    
    df.is_top = df.is_top.fillna(False)

    print(f"Top airport's had an average of {df[df.is_top].arr_delay.mean():.0f} minutes when flying to other top airports")
    print(f"Top airport's had an average of {df[~df.is_top].arr_delay.mean():.0f} minutes when flying to non-top airports")

def destination_delay_statistical_testing(df):
    subgroup_1 = df[df.is_top].arr_delay
    subgroup_2 = df[~df.is_top].arr_delay

    tstat, p = stats.ttest_ind(subgroup_1, subgroup_2)

    print(f"p value: {p:.2f}")

# ------------------- #
#     Time Series     #
# ------------------- #

def time_of_the_year_delays(df):
    df.fl_date = pd.to_datetime(df.fl_date)
    df = df.set_index("fl_date")
    monthly = df.resample("M").mean()

    plt.figure(figsize=(15,5))
    sns.lineplot(data=monthly, x=monthly.index, y="arr_delay")
    plt.title("Does time of year make a difference in avg minutes delayed?")
    plt.ylabel("Arrival delayed, in minutes")
    plt.xlabel("Date")

def day_of_the_week_delays(df):

    df = df.reset_index()

    df["day_name"] = df.fl_date.dt.day_name()

    plt.figure(figsize=(15,5))
    sns.barplot(data=df, x="day_name", y="arr_delay")
    plt.title("Does the day of the week matter?")
    plt.ylabel("Arrival delay, in minutes")

def time_of_the_day_delays(df):
    df["crs_dep_time"] = df['crs_dep_time'].astype(str).apply(lambda x: x.zfill(4))

    df["crs_dep_time"] = pd.to_datetime(df.crs_dep_time, format= "%H%M")

    df.set_index("crs_dep_time").arr_delay.resample("1H").mean().plot.line(figsize=(15,5))
    plt.title("Are there any hours where delays are more common?")
    plt.ylabel("Arrival delay, in minutes")
    plt.xlabel("Dept Time")

def daily_time_of_day_delays(df):
    # can we look at it by day of the week?
    df["day_name"] = df.fl_date.dt.day_name()

    fig = plt.figure(figsize=(11,8))
    days_of_week = df.day_name.unique()
    for count, day in enumerate(days_of_week):    
        plt.subplot(5, 2, 1+count)
        df[df.day_name == day].set_index("crs_dep_time").arr_delay.resample("1H").mean().plot.line()
        plt.title(day)

        fig.tight_layout()

    plt.show()

def additional_trends(df):
    df['is_delay'] = df['arr_delay'].apply(lambda x: 'True' if x > 0 else 'False')

    list_of_values = ["distance", "taxi_in", "air_time", "taxi_out", "carrier_delay", "nas_delay", "security_delay", "late_aircraft_delay"]

    f = plt.figure(figsize=(25,20))
    list_of_values = ["distance", "taxi_in", "air_time", "taxi_out"]

    for count, element in enumerate(list_of_values):
        f.add_subplot(4,5, count+1)
        sns.barplot(data=df, x="is_delay", y=element, ci=False)

        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def distance_scatterplot(df):
    plt.figure(figsize=(15,5))
    sns.scatterplot(data=df, y="arr_delay", x="distance")
    plt.title("Do longer flights experience more or less delays?")
    plt.xlabel("Distance")
    plt.ylabel("Arrival Delays, in minutes")

def distance_boxplot(df):
    df["distance_bins"] = pd.cut(df.distance, 3, labels=["short_flights", "medium_flights", "long_flights"])

    plt.figure(figsize=(15,5))
    sns.boxplot(data=df, x="distance_bins", y="arr_delay")
    plt.title("Does distance affect arrival delays?")
    plt.ylabel("Arrival delay, in minutes")
    plt.xlabel("Flight Distance")    

# ------------------------- #
#     Stats Analysis        #
# ------------------------- #

def anova_test_difference_in_day_of_week(df):

    df["day_name"] = df.fl_date.dt.day_name()

    monday = df[df.day_name == "Monday"].arr_delay
    tuesday = df[df.day_name == "Tuesday"].arr_delay
    wed = df[df.day_name == "Wednesday"].arr_delay
    thur = df[df.day_name == "Thursday"].arr_delay
    fri = df[df.day_name == "Friday"].arr_delay
    sat = df[df.day_name == "Saturday"].arr_delay
    sun = df[df.day_name == "Sunday"].arr_delay

    # ANOVA
    stat, pvalue = stats.f_oneway(monday, tuesday, wed, thur, fri, sat, sun)

    print(f"p value: {pvalue}")
