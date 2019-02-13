#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import na_counter as pre

toy = np.asarray([[-1., -1.], [-1., -1.], [ 1., 1.], [ 1., 1.]])
toy_na = np.asarray([[-1, np.nan], [np.nan, np.nan], [1, np.nan], [1, 1]])
toy_output = {'column':[0,1],'nans':[0,0]}
toy_na_output = {'column':[0,1],'nans':[1,3]}

# test na_counter(X, col_index)

# input type test

# test X type
def test_X_type():
    with pytest.raises(TypeError):
        pre.na_counter("X", col_index = [1,2])
        pre.na_counter({1}, col_index = [1])
        pre.na_counter(3.5)

# test col_index type
def test_col_index_type():
    with pytest.raises(TypeError):
        pre.na_counter(toy_na, col_index="1")
        pre.na_counter(toy_na, col_index=1.5)
        pre.na_counter(toy_na, col_index=(1))

# test col_index value
def test_col_index_vale():
    with pytest.raises(ValueError):
        pre.na_counter(toy, col_index=[2, 3])

# test numpy array/dataframe has at least one observation
def test_one_obs():
    with pytest.raises(ValueError):
        pre.na_counter(np.array([]), col_index=1)

# test expected outputs
def test_outputs():
    result_toy = pre.na_counter(toy, col_index=[0,1])
    assert(result_toy == toy_output), "Test array with no missing values, output incorrect."

    result_toy_na = pre.na_counter(toy_na, col_index=[0,1])
    assert(result_toy_na == toy_na_output), "Test array with missing values, output incorrect"

# test default col_index
def test_empty_columns():
    result_toy = pre.na_counter(toy, col_index=None)
    assert(result_toy == toy_output)
