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

