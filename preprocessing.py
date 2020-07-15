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
    return df

def create_new_features(df):
    df["airport_avg_delay"] = df.groupby("origin").arr_delay.transform("mean")
    df["dest_airport_avg_delay"] = df.groupby("dest").arr_delay.transform("mean")
    df["carrier_avg_delay"] = df.groupby("op_carrier").arr_delay.transform("mean")
    df["observation"] = df.fl_date.astype(str) + "_" + df.op_carrier + "_" + df.op_carrier_fl_num.astype(str)
    
    df["crs_dep_time"] = df['crs_dep_time'].astype(str).apply(lambda x: x.zfill(4))
    df["crs_dep_time"] = pd.to_datetime(df.crs_dep_time, format= "%H%M")
    df["dep_time_mean_delay"] = df.groupby("crs_dep_time").arr_delay.transform("mean")

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

        train, validate,test = split_data(df_modeling)

        X_train = train.drop(columns=target_variable)
        y_train = train[target_variable]
        X_validate = validate.drop(columns=target_variable)
        y_validate = validate[target_variable]
        X_test = test.drop(columns=target_variable)
        y_test = test[target_variable]
        
        scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)


        return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test



