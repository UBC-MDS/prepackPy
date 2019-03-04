#!/usr/bin/env python

import numpy as np
import pandas as pd
from typing import List, Any, Dict

def na_counter(X: Any, col_index: List[int] = None) -> Dict[str, List[int]]:
    """
    This function takes in tabular data and counts missing columnwise and summarizes the
    information in a dictionary.

    Parameters
    ----------
    X: numpy array, pandas dataframe
        an input dataset
    col_index: List[int]

    Returns
    -------
    na_dict: dictionary
        a summary dictionary (key: value)
        key = a column index of X that has missing values
        value = a list (NA counts of missing values in each column)

    Example
    -------
    >>> import numpy as np
    >>> from prepackPy import na_counter as na
    >>> X = np.array([[-1, np.nan], [np.nan, np.nan], [1, np.nan], [1, 1]])

    >>> na.na_counter(X, col_index=[0,1])
    {'column': [0, 1], 'nans': [1, 3]}


    """

    # input type check
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be of numpy.ndarray or pandas.DataFrame type")

    if not isinstance(col_index, (int, list, type(None))):
        raise TypeError("col_index must be of type int or list")

    # input value check
    if X.shape[0] < 1:
        raise ValueError("X does not contain any observations")

    if isinstance(col_index, int):
        if col_index > X.shape[1] - 1 or col_index < -X.shape[1]:
            raise ValueError("The target_index value is out of bounds")

    if isinstance(col_index, list):
        if max(col_index) > X.shape[1] -1 or min(col_index) < 0:
            raise ValueError("At least one value in target_index list out of bounds")

    ### function
    columns = X.shape[1]
    na_dict: Dict[str, List[int]] = {'column':[], 'nans':[]}
    X_na = np.asarray(X)

    if isinstance(col_index, int):
        X_na = X_na[:, col_index]

        nans = np.isnan(X_na[:,]).sum()
        na_dict['column'].append(col_index)
        na_dict['nans'].append(nans)

    elif isinstance(col_index, list):
        X_na = X_na[:, col_index]
        columns = X_na.shape[1]
        for i in range(0, columns):
            nans = np.isnan(X_na[:,i]).sum()

            na_dict['column'].append(i)
            na_dict['nans'].append(nans)

    else:
        columns = X_na.shape[1]
        for i in range(0, columns):
            nans = np.isnan(X_na[:,i]).sum()

            na_dict['column'].append(i)
            na_dict['nans'].append(nans)

    return(na_dict)