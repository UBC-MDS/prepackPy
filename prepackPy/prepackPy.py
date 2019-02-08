#!/usr/bin/env python

import numpy as np
import warnings
import pandas as pd


def splitter(X, target_index, split_size, seed): 
    """
    split a dataset into train and test sets

    Parameters
    ----------
    X: numpy array, pandas dataframe
        an input dataset
    target_index: integer
        a column index of target variable in X
    split_size: float
        a proportion of the dataset to include in the test split
    seed: integer
        a random state that will make code reproducible

    Returns
    -------
    X_train: numpy array
        splitted features for model training
    X_test: numpy array 
        splitted features for model testing
    y_train: 1d numpy array
        splitted target for model training
    y_test: 1d numpy array
        splitted target for model testing
    """
    return X_train, y_train, X_test, y_test

def stdizer(X, method, method_args, col_index=None):
    """
    standardize features

    Parameters
    ----------
    X: numpy array, pandas dataframe
        an input dataset
    method: string
        a method of standardization 
        one column of a standardized X can be calucated using the equation below:

        X_std[,i] = (X[,i] - first_value)/second_value

        allowable methods include:
        1. "mean_sd": subtracting the mean value of each column (first_value) and dividing by standard deviation value of each column (second_value) 
        2. "mean": subtracting the mean value of each column (first value) and dividing by 1 (second_value)
        3. "sd": subtracting 0 (first_value) and dividing by standard deviation of each column (second_value)
        4. "min_max": subtracting min value of each column (first_value) and dividing by max value of each column (second_value)
        5. "own": subtracting an user specified mean value of each column (first_value) and 
        dividing by another user specified standard deviation value of each column (second_value)
    method_args: list
        user specified arguments
        used for the last method, users need to identify their first and second value for each column as a format of list, 
        such as [first_value, second_value]
    col_index: list
        a list of column indices; Default is None
    
    Returns
    -------
    X_std: numpy array
        a standardized dataset
    """
    return X_std

def na_counter(X):
    """
    summarise the missing data in a dataset

    Parameters
    ----------
    X: numpy array, pandas dataframe
        an input dataset

    Returns
    -------
    na_dict: dictionary
        a summary dictionary (key: value)
        key = an column index of X that has missing values
        value = an tuple (NA counts, percentage of missing values in each column) 
    """
    return na_dict