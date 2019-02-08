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
    the input dataset
target_index: integer
    the column index of target variable in X
split_size: float
    the proportion of the dataset to include in the test split
seed: integer
    the random state that will make code reproducible

Returns
-------
X_train: numpy array
    the splitted features for model training
X_test: numpy array 
    the splitted features for model testing
y_train: 1d numpy array
    the splitted target for model training
y_test: 1d numpy array
    the splitted target for model testing
"""
    return X_train, y_train, X_test, y_test


def stdizer(X, col_index=None, method, method_args):
"""
standardize features

Parameters
----------
X: numpy array, pandas dataframe
    the input dataset
col_index: list
    the list of column indices; Default is None
method: string
    the method of standardization
method_args: list
    user specified arguments

Returns
-------
X_std: numpy array
    the standardized dataset
"""
    return X_std


def na_counter(X):
"""
summarise the missing data in a dataset

Parameters
----------
X: numpy array, pandas dataframe
    the input dataset

Returns
-------
na_dict: dictionary
    the summary dictionary (key = column index, value = NA count)
"""    
    return na_dict