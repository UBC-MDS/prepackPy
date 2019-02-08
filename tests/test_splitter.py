#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import prepackPy as pre

TOY_X = np.array([[1,2,3],[3,4,3],[5,6,3],[4,3,2]])

# test splitter(X, target_index, split_size, seed)

# input type test
# test X type
def test_X_type():
    with pytest.raises(TypeError):
        pre.splitter("X", target_index=1, split_size=0.2, seed=1)
        pre.splitter({1}, target_index=1, split_size=0.2, seed=1)
        pre.splitter(3.5, target_index=1, split_size=0.2, seed=1)

# test target_index type
def test_target_index_type():
    with pytest.raises(TypeError):
        pre.splitter(TOY_X, target_index="1", split_size=0.2, seed=1)
        pre.splitter(TOY_X, target_index=3.5, split_size=0.2, seed=1)
        
# test split_size type
def test_splitsize_type():
    with pytest.raises(TypeError):
        pre.splitter(TOY_X, target_index=1, split_size="0.2", seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=5, seed=1)

# test seed type
def test_seed_type():
    with pytest.raises(TypeError):
        pre.splitter(TOY_X, target_index=1, split_size=0.2, seed="1")
        pre.splitter(TOY_X, target_index=1, split_size=0.2, seed=0.5)
        
# test at least two obsevations
def test_X_size():
    with pytest.raises(ValueError):
        pre.splitter(np.array([1]), target_index=1, split_size=0.2, seed=1)
        pre.splitter(np.array([]), target_index=1, split_size=0.2, seed=1)

# test boundary of split size
def test_split_size_bound():
    with pytest.raises(ValueError):
        pre.splitter(TOY_X, target_index=1, split_size=1.5, seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=1.0, seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=0.0, seed=1)
        pre.splitter(TOY_X, target_index=1, split_size=-1.5, seed=1)

# test split proportions for each output
def test_split_proportions_size():
    X_train, y_train, X_test, y_test = pre.splitter(TOY_X, target_index=1, split_size=0.25, seed=1)
    assert(X_train.shape[0] == 3), "size doesn't match"
    assert(y_train.shape[0] == 3), "size doesn't match"
    assert(X_test.shape[0] == 1), "size doesn't match"
    assert(y_test.shape[0] == 1), "size doesn't match"

