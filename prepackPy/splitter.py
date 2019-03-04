#!/usr/bin/env python

import numpy as np
import pandas as pd
from typing import Any, Tuple

def splitter(X: Any, target_index: int, split_size: float, seed: int) -> Tuple[Any, Any, Any, Any]: 
    """
    split a dataset into train and test sets

    Parameters
    ----------
    X: numpy ndarray, pandas dataframe
        input dataset
    target_index: integer
        column index of target variable in X
    split_size: float
        proportion of the dataset to include in the test split
    seed: integer
        random state that will make code reproducible

    Returns
    -------
    X_train: numpy ndarray
        split features for model training
    y_train: numpy ndarray
        split target for model training
    X_test: numpy ndarray 
        split features for model testing
    y_test: numpy ndarray
        split target for model testing
        
    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from splitter import splitter
    >>> X = np.random.randint(10, size=(3, 3))
    >>> X
    array([[9, 9, 0],
           [4, 7, 3],
           [2, 7, 2]])
    >>> X_train, y_train, X_test, y_test = splitter(
    ...     X, target_index=2, split_size=0.3, seed=0)
    ...
    >>> X_train
    array([[0, 0],
           [5, 5]])
    >>> y_train
    array([4, 6])
    >>> X_test
    array([[8, 4]])
    >>> y_test
    array([1])
    """
    
    # input type check
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be of numpy.ndarray or pandas.dataframe type")
    if not isinstance(target_index, int):
        raise TypeError("target_index must be of integer type")
    if not isinstance(split_size, float):
        raise TypeError("split_size must be of float type")
    if not isinstance(seed, int):
        raise TypeError("seed must be integer type")
    
    # input value check
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must contains at least two observations and at least two columns")
    if target_index > X.shape[1]-1 or target_index < - X.shape[0]:
        raise ValueError("The absolute value of target_index cannot exceed the number of columns")
    if split_size >= 1 or split_size <= 0:
        raise ValueError("split_size must be within 0 and 1")
    if seed < 0 or seed > 2 * 32 -1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    column_index = [i for i in range(X.shape[1])]
    row_index = [i for i in range(X.shape[0])]
    label = X[:,target_index]
    column_index.pop(target_index)
    features = X[:, column_index]
    
    # set random seed
    np.random.seed(seed)
    test_index = np.random.choice(X.shape[0], int(round(X.shape[0] * split_size, 0)), replace=False)
    train_index = [x for x in row_index if x not in test_index]
    X_train: Any = features[train_index,:]
    y_train: Any = label[train_index]
    X_test: Any = features[test_index,:]
    y_test: Any = label[test_index]    
    return X_train, y_train, X_test, y_test