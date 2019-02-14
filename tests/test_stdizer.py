#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import stdizer as std

def test_correct_stdization():
    """
    Testing each standardization method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: mean_sd method
        Test 2: mean method
        Test 3: sd method
        Test 4: min_max method
        Test 5: own method
        X = Pandas Dataframe
        Test 6: mean_sd method
        Test 7: mean method
        Test 8: sd method
        Test 9: min_max method
        Test 10: own method
    """
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.array(data)
    df_data = pd.DataFrame(data)

    mean_sd_outcome = np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0,  1.0], [1.0,  1.0]])
    mean_outcome = np.array([[-0.5, -0.5], [-0.5, -0.5], [0.5,  0.5], [0.5,  0.5]])
    sd_outcome = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    min_max_outcome = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    own_outcome = np.array([[-2.0, -0.75], [-2.0, -0.75], [-1.5, -0.5], [-1.5, -0.5]])

    assert np.array_equal(std.stdizer(np_data, method="mean_sd"), mean_sd_outcome), "Output from (stdizer(np_data, method=`mean_sd`..) is incorrect"
    assert np.array_equal(std.stdizer(np_data, method="mean"), mean_outcome), "Output from (stdizer(np_data, method=`mean`..) is incorrect"
    assert np.array_equal(std.stdizer(np_data, method="sd"), sd_outcome), "Output from (stdizer(np_data, method=`sd`..) is incorrect"
    assert np.array_equal(std.stdizer(np_data, method="min_max"), min_max_outcome), "Output from (stdizer(np_data, method=`min_max`..) is incorrect"
    assert np.array_equal(std.stdizer(np_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Output from (stdizer(np_data, method=`own`..) is incorrect"

    assert np.array_equal(std.stdizer(df_data, method="mean_sd"), mean_sd_outcome), "Output from (stdizer(df_data, method=`mean_sd`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="mean"), mean_outcome), "Output from (stdizer(df_data, method=`mean`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="sd"), sd_outcome), "Output from (stdizer(df_data, method=`sd`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="min_max"), min_max_outcome), "Output from (stdizer(df_data, method=`min_max`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Output from (stdizer(df_data, method=`own`..) is incorrect"

def test_correct_argument_types():
    """
    Testing that the input values are of valid types
        Test 1: invalid input type for X
        Test 2: invalid input type for method
        Test 3: invalid input type for method_args
    """    
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.array(data)
    with pytest.raises(TypeError):
        std.stdizer(1, method="mean_sd")
        std.stdizer(np_data, method=3)
        std.stdizer(np_data, method="mean_sd", method_args=4)

def test_correct_argument_values():
    """
    Testing that the input values are valid
        Test 1: invalid value for method 
        Test 2: invalid value for method_args 
    """ 
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.array(data)
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="not a valid method")
        std.stdizer(np_data, method="own", method_args=[[1],[1,2]])