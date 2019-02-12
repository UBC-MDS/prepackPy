#!/usr/bin/env python

import numpy as np
import warnings
import pandas as pd

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
        # Testing the types passed in (WIP)
    
    # Transform the data
    X_stdized = np.asarray(X)
    X_stdized = X_stdized.astype(float)
    
    # Make sure only the allowed methods are passed in
    assert method in ["mean_sd","mean", "sd", "min_max", "own"], "Invalid input for the method argument"
    
    # Make sure the arguments in method_args are valid
    if method == "own":
        if len({len(i) for i in method_args}) == 2:
            raise ValueError('All lists in method_args must numeric and of length 2 (a. mean, b. standard deviation)')
        assert len(method_args) <= X_stdized.shape[1], "Too many method arguments have been entered"
    
    if (method == "mean_sd"):
        for i in range(0,X_stdized.shape[1]):
            mean = np.mean(X_stdized[:,i])
            std = np.std(X_stdized[:,i])
            X_stdized[:,i] = (X_stdized[:,i] - mean)/std
    elif (method == "mean"):
        for i in range(0,X_stdized.shape[1]):
            mean = np.mean(X_stdized[:,i])
            X_stdized[:,i] = (X_stdized[:,i] - mean)
    elif (method == "sd"):
        for i in range(0,X_stdized.shape[1]):
            std = np.std(X_stdized[:,i])
            X_stdized[:,i] = (X_stdized[:,i])/std
    elif (method == "min_max"):
        for i in range(0,X_stdized.shape[1]):
            min_val = np.min(X_stdized[:,i])
            max_val = np.max(X_stdized[:,i])
            X_stdized[:,i] = (X_stdized[:,i] - min_val)/max_val
    elif (method == "own"):
        for i in range(0,X_stdized.shape[1]):
            mean = method_args[i][0]
            std = method_args[i][1]
            X_stdized[:,i] = (X_stdized[:,i] - mean)/std
    return X_stdized
