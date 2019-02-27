#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import splitter as sp

TOY_X_df = pd.DataFrame({"x":[1,2,3],"y":[3,4,3],"z":[5,6,3],"m":[4,3,2]})
TOY_X = np.array([[1,2,3],[3,4,3],[5,6,3],[4,3,2]])

def test_X_type():
    """
    test the type of X
    """
    with pytest.raises(TypeError):
        sp.splitter("X", target_index=1, split_size=0.2, seed=1)

def test_target_index_type():
    """
    test the type of target_index
    """
    with pytest.raises(TypeError):
        sp.splitter(TOY_X, target_index="1", split_size=0.2, seed=1)

def test_split_size_type():
    """
    test the type of split_size
    """
    with pytest.raises(TypeError):
        sp.splitter(TOY_X, target_index=1, split_size="0.2", seed=1)

def test_seed_type():
    """
    test the type of seed
    """
    with pytest.raises(TypeError):
        sp.splitter(TOY_X, target_index=1, split_size=0.2, seed="1")

def test_X_value():
    """
    test the values of X
    """
    with pytest.raises(ValueError):
        sp.splitter(np.array([1]), target_index=1, split_size=0.2, seed=1)

def test_target_index_value():
    """
    test the values of target_index
    """
    with pytest.raises(ValueError):
        sp.splitter(TOY_X, target_index=4, split_size=0.2, seed=1)

def test_split_size_value():
    """
    test the values of split_size
    """
    with pytest.raises(ValueError):
        sp.splitter(TOY_X, target_index=1, split_size=1.5, seed=1)

def test_seed_value():
    """
    test the values of seed
    """
    with pytest.raises(ValueError):
        sp.splitter(TOY_X, target_index=1, split_size=0.5, seed=-1)

def test_function_run():
    """
    test if the function can run
    """
    sp.splitter(TOY_X_df, target_index=1, split_size=0.5, seed=1)

def test_output_size():
    """
    test split proportions for each output
    """
    X_train, y_train, X_test, y_test = sp.splitter(TOY_X, target_index=1, 
                                                   split_size=0.25, seed=1)
    assert(X_train.shape[0] == 3), "Size of X train doesn't match"
    assert(y_train.shape[0] == 3), "Size of y train doesn't match"
    assert(X_test.shape[0] == 1), "Size of X test doesn't match"
    assert(y_test.shape[0] == 1), "Size of y test doesn't match"

def test_output_value():
    """
    test value for each output
    """
    expected_Xtrain = np.array([[1, 3], [3, 3], [5, 3]])
    expected_ytrain = np.array([2,4,6])
    expected_Xtest = np.array([[4,2]])
    expected_ytest = np.array([3])

    X_train, y_train, X_test, y_test = sp.splitter(TOY_X, target_index=1, 
                                                   split_size=0.25, seed=1)
    assert np.array_equal(X_train, expected_Xtrain), "Value of X train doesn't match"
    assert np.array_equal(y_train, expected_ytrain), "Value of y train doesn't match"
    assert np.array_equal(X_test, expected_Xtest), "Value of X test doesn't match"
    assert np.array_equal(y_test, expected_ytest), "Value of y test doesn't match"