import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
from utils.map import df_map_features


def data_formatting(df: DataFrame, y_target: str, features: list):
    # Shaving of unused cols/feature
    df = pd.concat([df[features], df[y_target]], axis=1)

    # Replace "?" data with numpy NaN
    df = df.replace("?", np.nan)

    # Drop rows with missing values
    df_nafree = df.dropna()

    # Creating dataset containing only information
    # of incomplete feature values
    df_only_na = df[~df.index.isin(df_nafree.index)].reset_index(drop=True)
    df_only_na = df_only_na.replace(np.nan, "0")

    df.dropna(inplace=True)

    # Map the data consisting of words to numbers to get processed of the algorithm
    # Config data used to train and test the model
    df_train = df[features]
    df_train = dict_mapping(df_train)

    # Config data used to evaluate the model on unknown and incomplete data
    df_eval = df_only_na[features]
    df_eval = dict_mapping(df_eval)

    # Config data considered being the target output of the model and eval model
    df_target = df[[y_target]]
    df_target = dict_mapping(df_target)

    df_target_eval = df_only_na[[y_target]]
    df_target_eval = dict_mapping(df_target_eval)

    # Set the object types to float values to get processed by the algorithm
    train_vals = df_train.astype(float)
    train_vals = train_vals.values

    eval_vals = df_eval.astype(float)
    eval_vals = eval_vals.values

    target_vals = df_target.astype(int)
    target_vals = target_vals.values.flatten()

    target_eval_vals = df_target_eval.astype(int)
    target_eval_vals = target_eval_vals.values.flatten()

    return train_vals, eval_vals, target_vals, target_eval_vals


def dict_mapping(df: dict):
    # Replacing feature attributes with numbers
    for feature in df_map_features:
        if feature in df.columns:
            df = df.replace(df_map_features[feature][0])
    return df
