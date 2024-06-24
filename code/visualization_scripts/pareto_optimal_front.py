"""
A script for visualizing the Pareto-optimal front in the space spanned by weighted mean consistency and the size term sum(N^2)/(sum(N))^2 where N is ROI size. This Pareto-optimal front is used for selecting the threshold and regularization values used in the analysis. This script is used for producing supplementary fig SI1.

Written by Onerva Korhonen based on a script by Pietro de Luca.
"""
import nibabel as nib
import numpy as np
import pickle
# TODO: does it really make sense to use plotly instead of matplotlib?
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import itertools

def get_weighted_mean_consistency(ROI_sizes, ROI_consistencies, size_exp=1):
    """
    Calculates the mean spatial consistency weighted by ROI size.

    Parameters:
    -----------
    ROI_sizes : iterable of ints
        sizes of ROIs
    ROI_consistencies : iterable of floats
        spatial consistencies of ROIs
    size_exp : int (optional, default 1)
        exponent of size used for weighting

    Returns:
    --------
    weighted_mean_consistency : float
        mean spatial consistency weighted by size
    """
    if not isinstance(ROI_sizes, np.ndarray):
        ROI_sizes = np.array(ROI_sizes)
    if not isinstance(ROI_consistencies, np.ndarray):
        ROI_consistencies = np.array(ROI_consistencies)
    weighted_mean_consistency = np.sum(ROI_sizes**size_exp * ROI_consistencies) / np.sum(ROI_sizes**size_exp)
    return weighted_mean_consistency

def get_size_term(ROI_sizes, regularization_exp, N_voxels=0):
    """
    Calculates the size term sum(N^regularization_exp)/(X)^regularization_exp, where N is ROI size, the
    summation goes over ROIs, and X is either N_voxels or sum(N).

    Parameters:
    -----------
    ROI_sizes : iterable of ints
        sizes of ROIs
    regularization_exp : int
        explonent used for calculating the size term
    N_voxels : int (optional, default 0)
        total number of voxels in the brain, used for normalization. if not given, the sum of ROI sizes
        is used instead

    Returns:
    --------
    size_term : float
        the size term
    """
    if not isinstance(ROI_sizes, np.ndarray):
        ROI_sixes = np.array(ROI_sizes)
    if N_voxels > 0:
        size_term = np.sum(ROI_sizes**regularization_exponent) / N_voxels**regularization_exponent
    else:
        size_term = np.sum(ROI_sizes**regularization_exponent) / (np.sum(ROI_sizes))**regularization_exponent
    return size_term

def read_data(data_path, dataset_name, subj_id, threshold=0, regularization=0, n_time_windows=0):
    """
    Reads consistency data from file and writes it into a pandas data frame for further processing

    Parameters:
    -----------
    data_path : str
        path where the consistency data has been saved
    dataset_name : str
        name of the dataset to be read, used for creating the data frame
    subj_id : str
        subject ID related to the data
    threshold : float (optional, default 0)
        threshold used for obtaining the data
    regularization : float (optional, default 0)
        regularization value used for obtaining the data
    n_time_windows : int (optional, default 0)
        number of time windows in the data; if not given, windows are pooled
    
    Returns:
    --------
    consistency_data_frame : pandas.DataFrame()
        a data frame containing the consistency data
    """
    f = open(save_path, 'rb')
    data = pickle.load(f)
    f.close()

    ROI_sizes = []
    ROI_consistencies = []

    if n_time_windows > 0:
        window_index = 0
        time_windows = []
        for layer in data.values():
            ROI_sizes.extend(layer['ROI_sizes'].values())
            ROI_consistencies.extend(layer['consistencies'].values())
            time_windows.extend(len(layer['ROI_sizes'].values()) * [window_index])
            window_index += 1
        consistency_data = {'ROI_sizes':ROI_sizes, 'ROI_consistencies':ROI_consistencies, 'dataset':dataset_name, 'sub_id':subj_id, 'threshold':threshold, 'regularization':regularization, 'time_window':time_windows}
        consistency_data_frame = pd.DataFrame(consistency_data)

    else:
        for layer in data.values():
            ROI_sizes.extend(layer['ROI_sizes'].values())
            ROI_consistencies.extend(layer['ROI_consistencies'].values())
        consistency_data = {'ROI_sizes':ROI_sizes, 'ROI_consistencies':ROI_consistencies, 'dataset':dataset_name, 'sub_id':subj_id, 'threshold':threshold, 'regularization':regularization}
        consistency_data_frame = pd.DataFrame(consistency_data)

    return consistency_data_frame

def construct_pareto_optimal_front(consistency_data_frame, combined_array, n_time_windows, method):
    """
    Calculates the Pareto-optimal front from given data.

    Parameters:
    -----------
    consistency_data_frame : pandas.DataFrame()
        the dataframe constructed by read_data
    combined_array : list of tuples
        a list containing information about subjects, thresholds, regularization values, etc. used for obtaining the data
    n_time_windows : int
        number of time windows in the data
    method : str
        clustering method used for obtaining the data

    Returns:
    --------
    pareto_optimal_front : pandad.DataFrame()
        the Pareto-optimal front
    """
    pareto_data = pd.DataFrame()
    for subj_id, threshold, regularization in combined_array:
        name = '{sub_id}_{method}_{thr}_{reg}'.format(sub_id=sub_id, method=method, thr=threshold, reg=regularization)
        for i in range(n_time_windows):
            ROI_sizes = consistency_data_frame[(consistency_data_frame['dataset'] == name) & (consistency_data_frame['time_window'] == i)]['ROI_sizes']
            ROI_consistencies = consistency_data_frame[(consistency_data_frame['dataset'] == name) & (consistency_data_frame['time_window'] == i)]['ROI_consistencies']
            size_term = get_size_term(ROI_sizes)
            weighted_mean_consistency = get_weighted_mean_consistency(ROI_sizes, ROI_consistencies)
            pareto_data = {'size_term':size_term, 'weighted_mean_consistency':weighted_mean_consistency, 'subj_id':subj_id, 'method':method, 'threshold':threshold, 'regularization':regularization, 'time_window':i}
            pareto_data_frame = pd.concat([pareto_data_frame, pd.DataFrame(pareto_data)], ignore_index=True)

            pareto_data_frame = pareto_data_frame.sort_values(['size_term'], ascending=True)
            pareto_optimal_front = pd.DataFrame()
            for subj_id in pareto_data_frame.subj_id.unique():
                for time_window in pareto_data_frame.time_window.unique():
                    time_window_data_frame = pareto_data_frame[(pareto_data_frame['time_window'] == time_window) & (pareto_data_frame['subj_id'] == subj_id)]
                    time_window_data_frame = time_window_data_frame.reset_index(drop=True)
                    best_row = time_window_data_frame.loc[0]
                    for _, row in time_window_data_frame.iterrows():
                        if best_row.weighted_mean_consistency < row.weighted_mean_cosnistency:
                            pareto_optimal_front = pareto_optimal_front.append(row, ignore_index=True)
                            best_row = row

    return pareto_optimal_front



       

 

