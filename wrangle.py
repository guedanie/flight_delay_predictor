import pandas as pd
import numpy as np

import os.path
import requests
import io

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def prep_flight_data():
    df = pd.read_csv("2018.csv")
    top_airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA"]
    for i in top_airports:
        df.loc[df['ORIGIN'] == i, 'is_top'] = True 

    df.is_top = df.is_top.fillna(False)
    df = df[df.is_top]
    df[df.WEATHER_DELAY.isnull()].ARR_DELAY.mean()
    df.drop(columns = ["Unnamed: 27", "is_top", "CANCELLATION_CODE"], inplace=True)
    df = df[df.ARR_DELAY.notnull()]
    df = df[df.DEP_DELAY.notnull()]
    df = df.fillna(0.0)
    df.columns = map(str.lower, df.columns)

    # lets add the airline names

    airline_code = pd.read_csv("airline_codes.csv")

    airline_code.drop(columns="Unnamed: 0", inplace=True)
    df = df.merge(airline_code, how="left", left_on="op_carrier", right_on = "Code")
    df.drop(columns="Code", inplace=True)

    return df
