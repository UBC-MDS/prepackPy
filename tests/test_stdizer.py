#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import stdizer as std

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
np_data = np.array(data)
df_data = pd.DataFrame(data)

def test_correct_mean_sd():
    """
    Testing the mean_sd method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: mean_sd method
        X = Pandas Dataframe
        Test 2: mean_sd method
    """
    mean_sd_outcome = np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0,  1.0], [1.0,  1.0]])

    assert np.array_equal(std.stdizer(np_data, method="mean_sd"), mean_sd_outcome), "Output from (stdizer(np_data, method=`mean_sd`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="mean_sd"), mean_sd_outcome), "Output from (stdizer(df_data, method=`mean_sd`..) is incorrect"

def test_correct_mean():
    """
    Testing the mean method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: mean method
        X = Pandas Dataframe
        Test 2: mean method
    """
    mean_outcome = np.array([[-0.5, -0.5], [-0.5, -0.5], [0.5,  0.5], [0.5,  0.5]])

    assert np.array_equal(std.stdizer(np_data, method="mean"), mean_outcome), "Output from (stdizer(np_data, method=`mean`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="mean"), mean_outcome), "Output from (stdizer(df_data, method=`mean`..) is incorrect"

def test_correct_sd():
    """
    Testing the sd method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: sd method
        X = Pandas Dataframe
        Test 2: sd method
    """
    sd_outcome = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])

    assert np.array_equal(std.stdizer(np_data, method="sd"), sd_outcome), "Output from (stdizer(np_data, method=`sd`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="sd"), sd_outcome), "Output from (stdizer(df_data, method=`sd`..) is incorrect"

def test_correct_min_max():
    """
    Testing the min_max method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: min_max method
        X = Pandas Dataframe
        Test 2: min_max method
    """
    min_max_outcome = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])

    assert np.array_equal(std.stdizer(np_data, method="min_max"), min_max_outcome), "Output from (stdizer(np_data, method=`min_max`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="min_max"), min_max_outcome), "Output from (stdizer(df_data, method=`min_max`..) is incorrect"

def test_correct_own():
    """
    Testing the own method for the correct output
    Tests will be run for both a numpy array and pandas dataframe input (X):
        X = Numpy Array
        Test 1: own method
        X = Pandas Dataframe
        Test 2: own method
    """
    own_outcome = np.array([[-2.0, -0.75], [-2.0, -0.75], [-1.5, -0.5], [-1.5, -0.5]])

    assert np.array_equal(std.stdizer(np_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Output from (stdizer(np_data, method=`own`..) is incorrect"
    assert np.array_equal(std.stdizer(df_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Output from (stdizer(df_data, method=`own`..) is incorrect"

def test_correct_argument_type_X():
    """
    Testing that the input values are of valid types for X
        Test 1: invalid input type for X        
    """    
    with pytest.raises(TypeError):
        std.stdizer(1, method="mean_sd")

def test_correct_argument_type_method():
    """
    Testing that the input values are of valid types for method
        Test 1: invalid input type for method
    """   
    with pytest.raises(TypeError):
        std.stdizer(np_data, method=1)

def test_correct_argument_type_method_args():
    """
    Testing that the input values are of valid types for method_args
        Test 1: invalid input type for method
    """   
    with pytest.raises(TypeError):
        std.stdizer(np_data, method="mean_sd", method_args=3)

def test_correct_argument_values_own1():
    """
    Testing that the input values are valid
        Test 1: invalid value for method 
    """ 
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="not a valid method")

def test_correct_argument_values_own2():
    """
    Testing that the input values are valid
        Test 2: invalid value for method_args 
    """ 
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="own", method_args=[[1],[1,2]])

def test_correct_argument_values_own3():
    """
    Testing that the input values are valid
        Test 3: too many instances of own specified mean/sd
    """ 
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="own", method_args=[[1,2],[1,2],[3,2],[8,9],[8,9],[12,12],[3,4]])