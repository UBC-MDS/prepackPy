#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import prepackPy as pre

TOY_X = np.array([[1,2,3],[3,4,3],[5,6,3],[4,3,2]])

# test splitter(X, target_index, split_size, seed)

def test_input_type():
    """
    test the type of input parameters
    """
    with pytest.raises(TypeError):
        # test the type of X
        pre.splitter("X", target_index=1, split_size=0.2, seed=1)
        pre.splitter({1}, target_index=1, split_size=0.2, seed=1)
        # test the type of target_index
        pre.splitter(TOY_X, target_index="1", split_size=0.2, seed=1)
        pre.splitter(TOY_X, target_index=3.5, split_size=0.2, seed=1)
        # test the type of split_size
        pre.splitter(TOY_X, target_index=1, split_size="0.2", seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=[0.5], seed=1)
        # test the type of seed
        pre.splitter(TOY_X, target_index=1, split_size=0.2, seed="1")
        pre.splitter(TOY_X, target_index=1, split_size=0.2, seed=0.5)

def test_input_value():
    """
    test the values for input  parameters
    """
    with pytest.raises(ValueError):
        # test if X contains at least two observations
        pre.splitter(np.array([1]), target_index=1, split_size=0.2, seed=1)
        pre.splitter(np.array([]), target_index=1, split_size=0.2, seed=1)
        # test the boundary of target_index
        pre.splitter(TOY_X, target_index=4, split_size=0.2, seed=1)
        pre.splitter(TOY_X, target_index=-10, split_size=0.2, seed=1)
        # test thee boundary of split_size, should be (0,1)
        pre.splitter(TOY_X, target_index=1, split_size=1.5, seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=-1.5, seed=1)

def test_output_size():
    """
    test split proportions for each output
    """
    X_train, y_train, X_test, y_test = pre.splitter(TOY_X, target_index=1, split_size=0.25, seed=1)
    assert(X_train.shape[0] == 3), "size of X train doesn't match"
    assert(y_train.shape[0] == 3), "size of y train doesn't match"
    assert(X_test.shape[0] == 1), "size of X test doesn't match"
    assert(y_test.shape[0] == 1), "size of y test doesn't match"

