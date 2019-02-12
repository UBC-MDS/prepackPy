#!/usr/bin/env python

import numpy as np
import warnings
import pandas as pd

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