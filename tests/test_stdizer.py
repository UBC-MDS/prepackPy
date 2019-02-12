#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.append("../prepackPy")
import stdizer as pre

def correct_stdization():
    """
    Testing that the standardization methods are working correctly
    """
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    df_data = pd.DataFrame(data)

    mean_sd_outcome = np.array([[-1. -1.], [-1. -1.], [ 1.  1.], [ 1.  1.]])
    mean_outcome = np.array([[-0.5 -0.5], [-0.5 -0.5], [ 0.5  0.5], [ 0.5  0.5]])
    sd_outcome = np.array([[0. 0.], [0. 0.], [2. 2.], [2. 2.]])
    min_max_outcome = np.array([[-2. -0.75], [-2. -0.75], [-1.5 -0.5], [-1.5 -0.5.]])
    own_outcome = np.array([[0. 0.], [0. 0.], [1. 1.], [1. 1.]])

    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="mean_sd"), mean_sd_outcome), "Mean std, numpy"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="mean"), mean_outcome), "Mean, numpy"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="sd"), sd_outcome), "Std, numpy"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="min_max"), min_max_outcome), "Min_max, numpy"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Own, numpy"

    assert np.testing.assert_array_equal(pre.stdizer(df_data, method="mean_sd"), mean_sd_outcome), "Mean std, DataFrame"
    assert np.testing.assert_array_equal(pre.stdizer(df_data, method="mean"), mean_outcome), "Mean, DataFrame"
    assert np.testing.assert_array_equal(pre.stdizer(df_data, method="sd"), sd_outcome), "Std, DataFrame"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="min_max"), min_max_outcome), "Min_max, DataFrame"
    assert np.testing.assert_array_equal(pre.stdizer(np_data, method="own", method_args=[[4,2],[3,4]]), own_outcome), "Own, DataFrame"

def correct_argument_types():
    """
    Testing that the input values are valid types
    """    
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    with pytest.raises(TypeError):
        pre.stdizer(1, method="mean_sd")
        pre.stdizer(np_data, method=3)
        pre.stdizer(np_data, method="mean_sd", method_args=4)

def correct_argument_values():
    """
    Testing that the input values are valid
    """ 
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    np_data = np.asarray(data)
    data_2 = [[[0, 0]]]
    np_data_2 = np.asarray(data_2)
    with pytest.raises(ValueError):
        pre.stdizer(np_data, method="not a valid method")
        pre.stdizer(np_data, method="own", method_args=[[1],[1,2]])
        pre.stdizer(np_data, method="own", method_args=[[1,1],[1,0]])