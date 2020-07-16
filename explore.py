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
            