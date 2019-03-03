#!/usr/bin/env python

import numpy as np
import pandas as pd

def stdizer(X, method="mean_sd", method_args=None):
    """
    Standardize a dataset (X) based on a specificed standardization method (method & method_args).
    
    Parameters
    ----------
    X: numpy array, pandas dataframe
        an input dataset
    method: string
        a method of standardization 
        All columns of standardized X will be calucated using the equation below:
    
        X_std[,i] = (X[,i] - first_value)/second_value
    
        allowable methods include:
        1. "mean_sd": subtracting the mean value of each column (first_value) and 
                dividing by standard deviation value of each column (second_value)
        2. "mean": subtracting the mean value of each column (first_value) and dividing by 1 (second_value)
        3. "sd": subtracting 0 (first_value) and dividing by standard deviation of each column (second_value)
        4. "min_max": subtracting min value of each column (first_value) and 
                dividing by max value of each column (second_value)
        5. "own": subtracting an user specified mean value of each column (first_value) and 
                dividing by another user specified standard deviation value of each column (second_value)
    method_args: list
        The value of this argument will be None except when method="own"
        Users need to identify the mean (first_value) and standard deviation (second_value) for each column in X, 
        This will be passed into the function as a list of lists:
                [[mean_col_1, std_col_1], [mean_col_2, std_col_2]...[mean_col_n, std_col_n]]
    
    Returns
    -------
    X_std: numpy array
        a dataset standardized based on the method argument

    Example
    -------
    >>> import numpy as np
    >>> from prepackPy import stdizer as sd
    >>> X = np.array([[-1, 0], [2, 1], [1, -2], [1, 1]])
    >>> X
    array([[-1,  0],
           [ 2,  1],
           [ 1, -2],
           [ 1,  1]])
    >>> X_std = sd.stdizer(X, method="mean_sd", method_args=None)
    >>> X_std
    array([[-1.60591014,  0.        ],
           [ 1.14707867,  0.81649658],
           [ 0.22941573, -1.63299316],
           [ 0.22941573,  0.81649658]])
    """
    # Testing that the correct types have been passed into the function
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be of numpy.ndarray or pandas.dataframe type")
    if not isinstance(method, str):
        raise TypeError("method must be a string type")
    if method_args:
        if not isinstance(method_args, list):
            raise TypeError("Method args must be of type list")
            
    # Make sure only the 5 allowed methods have been passed into the function
    if not method in ["mean_sd","mean", "sd", "min_max", "own"]:
        raise ValueError("Invalid input for the method argument")

    # Transform the dataset (X) to a numpy array
    X_stdized = np.asarray(X)
    X_stdized = X_stdized.astype(float)
    
    # Make sure the arguments in method_args are valid
    if method == "own":
        if len(method_args) > X_stdized.shape[1]:
            raise ValueError('Too many method arguments have been entered')
    if method == "own":
        if len({len(i) for i in method_args}) == 2:
            raise ValueError('All lists in method_args must numeric and of length 2 (a. mean, b. standard deviation)')
    
    # Create the standardized matrix
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
    else:
        for i in range(0,X_stdized.shape[1]):
            mean = method_args[i][0]
            std = method_args[i][1]
            X_stdized[:,i] = (X_stdized[:,i] - mean)/std
    return X_stdized