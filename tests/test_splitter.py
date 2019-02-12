#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import splitter as sp

TOY_X = np.array([[1,2,3],[3,4,3],[5,6,3],[4,3,2]])

def test_input_type():
    """
    test the type of input parameters
    """
    with pytest.raises(TypeError):
        # test the type of X
        sp.splitter("X", target_index=1, split_size=0.2, seed=1)
        sp.splitter({1}, target_index=1, split_size=0.2, seed=1)
        # test the type of target_index
        sp.splitter(TOY_X, target_index="1", split_size=0.2, seed=1)
        sp.splitter(TOY_X, target_index=3.5, split_size=0.2, seed=1)
        # test the type of split_size
        sp.splitter(TOY_X, target_index=1, split_size="0.2", seed=1)
        sp.splitter(TOY_X, target_index=1, split_size=[0.5], seed=1)
        # test the type of seed
        sp.splitter(TOY_X, target_index=1, split_size=0.2, seed="1")
        sp.splitter(TOY_X, target_index=1, split_size=0.2, seed=0.5)

def test_input_value():
    """
    test the values for input  parameters
    """
    with pytest.raises(ValueError):
        # test if X contains at least two observations
        sp.splitter(np.array([1]), target_index=1, split_size=0.2, seed=1)
        sp.splitter(np.array([]), target_index=1, split_size=0.2, seed=1)
        # test the boundary of target_index
        sp.splitter(TOY_X, target_index=4, split_size=0.2, seed=1)
        sp.splitter(TOY_X, target_index=-10, split_size=0.2, seed=1)
        # test thee boundary of split_size, should be (0,1)
        sp.splitter(TOY_X, target_index=1, split_size=1.5, seed=1)
        sp.splitter(TOY_X, target_index=1, split_size=-1.5, seed=1)

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