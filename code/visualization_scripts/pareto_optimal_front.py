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
    try:
        f = open(data_path, 'rb')
        data = pickle.load(f)
        f.close()
    except:
        print('file {path} not found'.format(path=data_path))

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
    pareto_optimal_front : pandas.DataFrame()
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

consistency_save_path_base = '/m/nbe/scratch/alex/private/tarmo/article_runs/maxcorr/'
pareto_optimal_front_save_path_base = 'm/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/pareto_optimization/'
figure_save_path = 'm/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/pareto_optimization/pareto_optimal_front_all_methods.pdf'

subject_ids = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']
run_number = 2
n_time_windows = 56

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
threshold_strs = ['01', '02', '03', '04', '05']
regularizations = [-10, -100, -250, -500, -1000]

craddock_data_threshold_str = '02' # the craddock files used for the main analysis are named differently than those used for the Pareto optimization, so a hack is needed for reading all the files

methods = ['ReHo_seeds_min_correlation_voxelwise_thresholding', 'craddock']
#['ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding', 'ReHo_seeds_min_correlation_voxelwise_thresholding', 'craddock']

calculate_pareto_optimal_front = True

#import pdb; pdb.set_trace()
if calculate_pareto_optimal_front:
    for method in methods:
        if method == 'ReHo_seeds_weighted_mean_cosistency_voxelwise_thresholding':
            combined_array = list(itertools.product(subject_ids, threshold_strs, regularizations))
        else:
            combined_array = list(itertools.product(subject_ids, threshold_strs, ['']))
        consistency_data_frame = pd.DataFrame()
        for subj_id, threshold, regularization in combined_array:
            if method == 'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding': 
                consistency_save_path = consistency_save_path_base + '{subj_id}/{run_number}/{method}_{threshold}_regularization{regularization}/2_layers/spatial_consistency.pkl'.format(subj_id=subj_id, run_number=run_number, method=method, threshold=threshold, regularization=regularization)
            elif method == 'ReHo_seeds_min_correlation_voxelwise_thresholding':
                consistency_save_path = consistency_save_path_base + '{subj_id}/{run_number}/{method}_{threshold}/2_layers/spatial_consistency.pkl'.format(subj_id=subj_id, run_number=run_number, method=method, threshold=threshold)
            elif method == 'craddock':
                if threshold == craddock_data_threshold_str:
                    consistency_save_path = consistency_save_path_base + '{subj_id}/{run_number}/{method}/2_layers/spatial_consistency.pkl'.format(subj_id=subj_id, run_number=run_number, method=method)
                else:
                    consistency_save_path = consistency_save_path_base + '{subj_id}/{run_number}/{method}_threshold_{threshold}/2_layers/spatial_consistency.pkl'.format(subj_id=subj_id, run_number=run_number, method=method, threshold=threshold)
            dataset_name = '{subj_id}_{method}_{threshold}_{regularization}'.format(subj_id=subj_id, method=method, threshold=threshold, regularization=regularization)
            consistency_data_frame = pd.concat([consistency_data_frame, read_data(consistency_save_path, dataset_name, subj_id, threshold, regularization, n_time_windows)])
        pareto_optimal_front = construct_pareto_optimal_front(consistency_data_frame, combined_array, n_time_windows, method)
        pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}.pkl'.format(method=method)
        pareto_optimal_front.to_pickle(pareto_optimal_front_save_path)

if not calculate_pareto_optimal_front: # assuming that the front has been calculated before and thus reading data
    pareto_optimal_front_all_methods = pd.DataFrame()
    for method in methods:
        pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}.pkl'.format(method=method)
        pareto_optimal_front = pd.read_pickle(pareto_optimal_fornt_save_path)
        pareto_optimal_front_all_methods = pd.concat([pareto_optimal_front_all_methods, pareto_optimal_front])
    
# replacing subject ids with unique numerical indices
i = 0
pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values('subj_id')
for subj_id in pareto_optimal_front_all_methods.subj_id.unique():
    pareto_optimal_front_all_methods.loc[pareto_optimal_front_all_methods['subj_id'] == subj_id, 'subj_id'] = i
    i += 1

# replacing threshold strings with numerical thresholds
pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values('threshold')
for threshold_str, threshold in zip(pareto_optimal_front_all_methods.threshold.unique(), thresholds):
    pareto_optimal_front_all_methods.loc[pareto_optimal_front_all_methods['threshold'] == threshold_str, 'threshold'] = threshold

# visualization
pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values(['time_window', 'threshold', 'regularization', 'subj_id'])
pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values(['size_term', 'weighted_mean_consistency'])
pareto_optimal_front_all_methods['threshold'] = pareto_optimal_front_all_methods['threshold'].astype(float)
pareto_optimal_front_all_methods['time_window'] = pareto_optimal_front_all_methods['time_winodw'].astype(int)
fig = px.line(pareto_optimal_front_all_methods, x='size_term', y='weighted_mean_consistency', title='pareto optimal front', color='method', markers=True, symbol_sequence=['circle', 'square', 'cross', 'star','triangle-up','x','diamond-tall','star-diamond'], line_group=pareto_optimal_front_all_methods[['time_window', 'subj_id']].astype(str).apply('-'.join, axis=1)) # TODO: consider adding range_x and range_y
fig.update_traces(marker_size=10, marker_opacity=0.3)
fig.update_layout(xaxis=dict(tick0=0, dtick=0.0005,title='size term'), yaxis=dict(title='weighted mean consistency'))
fig.write_image('figure_save_path')



       

 

