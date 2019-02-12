#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import stdizer as std

def test_correct_stdization():
    """
    Testing that the standardization methods are working correctly
    """
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    df_data = pd.DataFrame(data)

    mean_sd_outcome = np.asarray([[-1.0, -1.0], [-1.0, -1.0], [1.0,  1.0], [1.0,  1.0]])
    mean_outcome = np.asarray([[-0.5, -0.5], [-0.5, -0.5], [0.5,  0.5], [0.5,  0.5]])
    sd_outcome = np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    min_max_outcome = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    own_outcome = np.asarray([[-2.0, -0.75], [-2.0, -0.75], [-1.5, -0.5], [-1.5, -0.5]])

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
    Testing that the input values are valid types
    """    
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    with pytest.raises(TypeError):
        std.stdizer(1, method="mean_sd")
        std.stdizer(np_data, method=3)
        std.stdizer(np_data, method="mean_sd", method_args=4)

def test_correct_argument_values():
    """
    Testing that the input values are valid
    """ 
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    data_2 = [[[0, 0]]]
    np_data_2 = np.asarray(data_2)
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="not a valid method")
        std.stdizer(np_data, method="own", method_args=[[1],[1,2]])
        std.stdizer(np_data, method="own", method_args=[[1,1],[1,0]])
    with pytest.raises(ValueError):
        std.stdizer(np_data, method="not a valid method")