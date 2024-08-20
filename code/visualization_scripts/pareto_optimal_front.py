"""
A script for visualizing the Pareto-optimal front in the space spanned by weighted mean consistency and the size term sum(N^2)/(sum(N))^2 where N is ROI size. This Pareto-optimal front is used for selecting the threshold and regularization values used in the analysis. This script is used for producing supplementary fig SI1.

Written by Onerva Korhonen based on a script by Pietro de Luca.
"""
import nibabel as nib
import numpy as np
import pickle
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
    weighted_mean_consistency = np.sum(ROI_sizes**size_exp * ROI_consistencies) / np.float(np.sum(ROI_sizes**size_exp))
    return weighted_mean_consistency

def get_size_term(ROI_sizes, regularization_exponent=2, N_voxels=0):
    """
    Calculates the size term sum(N^regularization_exp)/(X)^regularization_exp, where N is ROI size, the
    summation goes over ROIs, and X is either N_voxels or sum(N).

    Parameters:
    -----------
    ROI_sizes : iterable of ints
        sizes of ROIs
    regularization_exp : int (optional, default 2)
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
        ROI_sizes = np.array(ROI_sizes)
    if N_voxels > 0:
        size_term = np.sum(ROI_sizes**regularization_exponent) / np.float(N_voxels**regularization_exponent)
    else:
        size_term = np.sum(ROI_sizes**regularization_exponent) / np.float(np.sum(ROI_sizes)**regularization_exponent)
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
        number of time windows in the data; if 0, windows are pooled
    
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
        consistency_data = {'ROI_sizes':ROI_sizes, 'ROI_consistencies':ROI_consistencies, 'dataset':dataset_name, 'subj_id':subj_id, 'threshold':threshold, 'regularization':regularization, 'time_window':time_windows}
        consistency_data_frame = pd.DataFrame(consistency_data)

    else:
        for layer in data.values():
            ROI_sizes.extend(layer['ROI_sizes'].values())
            ROI_consistencies.extend(layer['consistencies'].values())
        consistency_data = {'ROI_sizes':ROI_sizes, 'ROI_consistencies':ROI_consistencies, 'dataset':dataset_name, 'subj_id':subj_id, 'threshold':threshold, 'regularization':regularization}
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
        number of time windows in the data; if 0, a single Pareto-optimal front is calculated across all time windows
    method : str
        clustering method used for obtaining the data

    Returns:
    --------
    pareto_optimal_front : pandas.DataFrame()
        the Pareto-optimal front
    """
    pareto_data_frame = pd.DataFrame()
    for subj_id, threshold, regularization in combined_array:
        name = '{subj_id}_{method}_{thr}_{reg}'.format(subj_id=subj_id, method=method, thr=threshold, reg=regularization)
        if n_time_windows > 0:
            for i in range(n_time_windows):
                ROI_sizes = consistency_data_frame[(consistency_data_frame['dataset'] == name) & (consistency_data_frame['time_window'] == i)]['ROI_sizes']
                ROI_consistencies = consistency_data_frame[(consistency_data_frame['dataset'] == name) & (consistency_data_frame['time_window'] == i)]['ROI_consistencies']
                size_term = get_size_term(ROI_sizes)
                weighted_mean_consistency = get_weighted_mean_consistency(ROI_sizes, ROI_consistencies)
                if front_per_window:
                    pareto_data = {'size_term':size_term, 'weighted_mean_consistency':weighted_mean_consistency, 'subj_id':subj_id, 'method':method, 'threshold':threshold, 'regularization':regularization, 'time_window':i}
                else:
                    pareto_data = {'size_term':size_term, 'weighted_mean_consistency':weighted_mean_consistency, 'subj_id':subj_id, 'method':method, 'threshold':threshold, 'regularization':regularization, 'time_window':0} # while size term and weighted mean consistency are calculated separately for each window, windows are pooled before calculating the Pareto-optimal front
                pareto_data_frame = pd.concat([pareto_data_frame, pd.DataFrame([pareto_data])], ignore_index=True)
        else:
            ROI_sizes = consistency_data_frame[(consistency_data_frame['dataset'] == name)]['ROI_sizes']
            ROI_consistencies = consistency_data_frame[(consistency_data_frame['dataset'] == name)]['ROI_consistencies']
            size_term = get_size_term(ROI_sizes)
            weighted_mean_consistency = get_weighted_mean_consistency(ROI_sizes, ROI_consistencies)
            pareto_data = {'size_term':size_term, 'weighted_mean_consistency':weighted_mean_consistency, 'subj_id':subj_id, 'method':method, 'threshold':threshold, 'regularization':regularization, 'time_window':0}
            pareto_data_frame = pd.concat([pareto_data_frame, pd.DataFrame([pareto_data])], ignore_index=True)

    pareto_data_frame = pareto_data_frame.sort_values(['size_term'], ascending=True)
    pareto_optimal_front = pd.DataFrame()
    for subj_id in pareto_data_frame.subj_id.unique():
        for time_window in pareto_data_frame.time_window.unique():
            time_window_data_frame = pareto_data_frame[(pareto_data_frame['time_window'] == time_window) & (pareto_data_frame['subj_id'] == subj_id)]
            time_window_data_frame = time_window_data_frame.reset_index(drop=True)
            best_row = time_window_data_frame.loc[0]
            front_updated = False
            for _, row in time_window_data_frame.iterrows():
                if best_row.weighted_mean_consistency < row.weighted_mean_consistency:
                    pareto_optimal_front = pareto_optimal_front.append(row, ignore_index=True)
                    best_row = row
                    front_updated = True
            if not front_updated:
                pareto_optimal_front = pareto_optimal_front.append(best_row, ignore_index=True)

    return pareto_optimal_front

consistency_save_path_base = '/m/nbe/scratch/alex/private/tarmo/article_runs/maxcorr/'
pareto_optimal_front_save_path_base = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/pareto_optimization/pareto_optimal_front'
figure_save_path_base = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/pareto_optimization/pareto_optimal_front_all_methods'

subject_ids = ['b1k','d6i','e6x','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','v1i','v5b','y6g','z4t', 't9u', 'd3a', 'd4w', 'g3r']
run_number = 2
n_time_windows = 56

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
threshold_strs = ['01', '02', '03', '04', '05']
regularizations = [-10, -100, -250, -500, -1000]

greedy_data_threshold = 0.3
data_regularization = -100
craddock_data_threshold_str = '02' # the craddock files used for the main analysis are named differently than those used for the Pareto optimization, so a hack is needed for reading all the files
craddock_data_threshold = 0.2

methods = ['ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding', 'ReHo_seeds_min_correlation_voxelwise_thresholding', 'craddock']

front_per_window = False # if True, a separate Pareto-optimal front is calculated for each time window; otherwise one front is calculated per subject
collapse_time = False # if True, ROIs from all windows are pooled before calculating the Pareto-optimal front
if collapse_time:
    front_per_window = False # these two cannot be True at the same time
calculate_pareto_optimal_front = False
visualize = True

if collapse_time:
    n_time_windows = 0
    figure_save_path = '{base}_collapsed.pdf'.format(base=figure_save_path_base)
    full_figure_save_path = '{base_collapsed_full.pdf'.format(base=figure_save_path_base)
elif front_per_window:
    figure_save_path = '{base}_per_window.pdf'.format(base=figure_save_path_base)
    full_figure_save_path = '{base}_per_window_full.pdf'.format(base=figure_save_path_base)
else:
    figure_save_path = '{base}.pdf'.format(base=figure_save_path_base)
    full_figure_save_path = '{base}_full.pdf'.format(base=figure_save_path_base)

if calculate_pareto_optimal_front:
    pareto_optimal_front_all_methods = pd.DataFrame()
    for method in methods:
        if method == 'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding':
            combined_array = list(itertools.product(subject_ids, threshold_strs, regularizations))
        else:
            combined_array = list(itertools.product(subject_ids, threshold_strs, ['']))
        consistency_data_frame = pd.DataFrame()
        for subj_id, threshold, regularization in combined_array:
            print('reading {method}, {subj_id}, {threshold}, {regularization}'.format(method=method, subj_id=subj_id, threshold=threshold, regularization=regularization))
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
        pareto_optimal_front_all_methods = pd.concat([pareto_optimal_front_all_methods, pareto_optimal_front])
        if collapse_time:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}_collapsed.pkl'.format(method=method)
        elif front_per_window:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}_per_window.pkl'.format(method=method)
        else:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}.pkl'.format(method=method)
        pareto_optimal_front.to_pickle(pareto_optimal_front_save_path)
else: # assuming that the front has been calculated before and thus reading data
    #import pdb; pdb.set_trace()
    pareto_optimal_front_all_methods = pd.DataFrame()
    for method in methods:
        if collapse_time:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}_collapsed.pkl'.format(method=method)
        elif front_per_window:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}_per_window.pkl'.format(method=method)
        else:
            pareto_optimal_front_save_path = pareto_optimal_front_save_path_base + '_{method}.pkl'.format(method=method)
        pareto_optimal_front = pd.read_pickle(pareto_optimal_front_save_path)
        pareto_optimal_front_all_methods = pd.concat([pareto_optimal_front_all_methods, pareto_optimal_front])

if visualize:
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

    # finding points corresponding to the param values used in the main analysis
    pareto_optimal_markers_all_methods = pd.DataFrame()
    for method in methods:
        if method == 'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding':
            pareto_optimal_front_markers = pareto_optimal_front_all_methods[(pareto_optimal_front_all_methods['method'] == method) & (pareto_optimal_front_all_methods['threshold'] == greedy_data_threshold) & (pareto_optimal_front_all_methods['regularization'] == data_regularization)]
        elif method == 'ReHo_seeds_min_correlation_voxelwise_thresholding':
            pareto_optimal_front_markers = pareto_optimal_front_all_methods[(pareto_optimal_front_all_methods['method'] == method) & (pareto_optimal_front_all_methods['threshold'] == greedy_data_threshold)]
        elif method == 'craddock':
            pareto_optimal_front_markers = pareto_optimal_front_all_methods[(pareto_optimal_front_all_methods['method'] == method) & (pareto_optimal_front_all_methods['threshold'] == craddock_data_threshold)]
        pareto_optimal_markers_all_methods = pd.concat([pareto_optimal_markers_all_methods, pareto_optimal_front_markers])

    # visualization
    pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values(['time_window', 'threshold', 'regularization', 'subj_id'])
    pareto_optimal_front_all_methods = pareto_optimal_front_all_methods.sort_values(['size_term', 'weighted_mean_consistency'])
    pareto_optimal_front_all_methods['threshold'] = pareto_optimal_front_all_methods['threshold'].astype(float)
    pareto_optimal_front_all_methods['time_window'] = pareto_optimal_front_all_methods['time_window'].astype(int)
    fig = go.Figure()
    for i, method in enumerate(methods):
        pareto_optimal_front = pareto_optimal_front_all_methods[(pareto_optimal_front_all_methods['method'] == method)]
        pareto_optimal_markers = pareto_optimal_markers_all_methods[(pareto_optimal_markers_all_methods['method'] == method)]
        for j, subj_id in enumerate(pareto_optimal_front.subj_id.unique()):
            for k, time_window in enumerate(pareto_optimal_front.time_window.unique()):
                subj_time_window_pareto_optimal_front = pareto_optimal_front[(pareto_optimal_front['subj_id'] == subj_id) & (pareto_optimal_front['time_window'] == time_window)]
                subj_time_window_pareto_optimal_markers = pareto_optimal_markers[(pareto_optimal_markers['subj_id'] == subj_id) & (pareto_optimal_markers['time_window'] == time_window)]
                if j == k == 0:
                    fig.add_traces(go.Scatter(x=subj_time_window_pareto_optimal_front.size_term, y=subj_time_window_pareto_optimal_front.weighted_mean_consistency, mode='lines', line=dict(color=fig.layout['template']['layout']['colorway'][i], width=.5), legendgroup=method, name=method))
                else:
                    fig.add_traces(go.Scatter(x=subj_time_window_pareto_optimal_front.size_term, y=subj_time_window_pareto_optimal_front.weighted_mean_consistency, mode='lines', line=dict(color=fig.layout['template']['layout']['colorway'][i], width=.5), legendgroup=method, name=method, showlegend=False))
                fig.add_traces(go.Scatter(x=subj_time_window_pareto_optimal_markers.size_term, y=subj_time_window_pareto_optimal_markers.weighted_mean_consistency, mode='markers', marker_color=fig.layout['template']['layout']['colorway'][i], marker_size=7, showlegend=False))
    fig.update_traces(marker={'opacity':1})
    fig.update_traces(line={'width':1})
    fig.update_layout(xaxis=dict(tick0=0, dtick=0.01,title='size term'), yaxis=dict(tick0=0, title='weighted mean consistency'))
    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')
    #fig.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
    #fig.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(ticks='outside', tickwidth=2)
    fig.update_yaxes(ticks='outside', tickwidth=2)
    fig.update_layout(plot_bgcolor='white')
    fig.write_image(full_figure_save_path)
    fig.update_layout(xaxis=dict(tick0=0.004, dtick=0.002))
    fig.update_xaxes(range=[0.004, 0.014])
    #fig.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
    #fig.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
    #fig.update_yaxes(showline=True, linecolor='black')
    fig.write_image(figure_save_path)



       

 

