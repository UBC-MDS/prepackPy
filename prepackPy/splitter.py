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