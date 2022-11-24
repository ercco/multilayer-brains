import pymnet as pn
import itertools
import numpy as np
import collections
import os
import pickle
import sys
import nibabel as nib

import subgraph_classification
import network_io
import network_construction
import pipeline
import corrs_and_mask_calculations
import clustering_by_consistency as cbc

# null_models.py: Create null model random networks and find NODE-ISOMORPHIC isomorphism classes from them

def null_model_subgraphs_from_network_file(filename,intralayer_density,interlayer_density,null_model_function,nnodes,nlayers,number_of_repeats):
    M = network_io.read_weighted_network(filename)
    M = network_construction.threshold_multilayer_network(M,intralayer_density,interlayer_density)
    all_repetitions_results = null_model_subgraphs_from_network(M,null_model_function,nnodes,nlayers,number_of_repeats)
    return all_repetitions_results

def null_model_subgraphs_from_network(M,null_model_function,nnodes,nlayers,number_of_repeats):
    all_repetitions_results = collections.defaultdict(list)
    for ii in range(number_of_repeats):
        M_null = null_model_function(M)
        results_dict = collections.defaultdict(dict)
        subgraph_classification.find_isomorphism_classes(M_null,nnodes,nlayers,filename=None,allowed_aspects=[0],aggregated_dict=results_dict)
        for compinv in results_dict:
            compinv_total_from_M_null = 0
            for layerset in results_dict[compinv]:
                compinv_total_from_M_null = compinv_total_from_M_null + results_dict[compinv][layerset]
            all_repetitions_results[compinv].append(compinv_total_from_M_null)
    return all_repetitions_results

def ER_from_net(M):
    net_layers = list(M.iter_layers())
    M_null = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    for layer in net_layers:
        M_null.add_layer(layer)
    for nl in M.iter_node_layers():
        M_null.add_node(nl[0],nl[1])
    edges_on_layers = dict()
    edges_between_layers = dict()
    for e in M.edges:
        if e[2] == e[3]:
            edges_on_layers[e[2]] = edges_on_layers.get(e[2],0) + 1
        else:
            sorted_edge = tuple(sorted([e[2],e[3]]))
            edges_between_layers[sorted_edge] = edges_between_layers.get(sorted_edge,0) + 1
    for layer in M_null.iter_layers():
        possible_edges = list(itertools.combinations(M_null.iter_nodes(layer),2))
        for index in np.random.choice(len(possible_edges),size=edges_on_layers[layer],replace=False):
            M_null[possible_edges[index][0],possible_edges[index][1],layer] = 1
    for layerpair in edges_between_layers:
        possible_edges = list(itertools.product(M_null.iter_nodes(layerpair[0]),M_null.iter_nodes(layerpair[1])))
        for index in np.random.choice(len(possible_edges),size=edges_between_layers[layerpair],replace=False):
            M_null[possible_edges[index][0],possible_edges[index][1],layerpair[0],layerpair[1]] = 1
    return M_null

def shuffle_nii(nii_path,mask_path,n_shuffles,out_path_stem):
    """
    Reads a NIFTI file, shuffles the voxel time series while retaining voxel locations, and saves the shuffled time series in 
    a new NIFTI file. The shuffling does not affect the affine of the NIFTI.
    
    Parameters:
    -----------
    nii_path: str, path to the input NIFTI file
    mask_path: str, path to a mask defining the voxel time series to shuffle (can be e.g. a gray matter or ROI mask)
    n_shuffles: int, number of NIFTI files with shuffled time series to be produced
    out_path_stem: str, path to which save the shuffled time series
    
    Returns:
    --------
    no direct output, saves the shuffled time series as NIFTI
    """
    # reading data:
    img = nib.load(nii_path) 
    affine = img.affine
    imgdata = img.get_fdata()
    n_x,n_y,n_z,n_t = imgdata.shape
    # reading mask, masking data and finding voxel coordinates
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    corrs_and_mask_calculations.gray_mask(imgdata,mask_data)
    voxel_coordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    # reading voxel time series
    n_voxels = len(voxel_coordinates)
    all_voxel_ts = np.zeros((n_voxels,n_t))
    for i,voxel in enumerate(voxel_coordinates):
        all_voxel_ts[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
    # shuffling the voxel time series, creating shuffled nii, and saving
    for i in range(n_shuffles):
        shuffled_ts = all_voxel_ts.copy()
        np.random.shuffle(shuffled_ts)
        shuffled_imgdata = np.zeros((n_x,n_y,n_z,n_t))
        for voxel,ts in zip(voxel_coordinates,shuffled_ts):
            shuffled_imgdata[voxel[0],voxel[1],voxel[2],:] = ts
        out_path = out_path_stem + '_shuffle_' + str(i) + '.nii'
        shuffled_img = nib.Nifti1Image(shuffled_imgdata,affine)     
        nib.save(shuffled_img,out_path)
        

#################### Running functions #########################################

def ER_for_file_list(file_list,nnodes,nlayers,intralayer_density,interlayer_density):
    # Assume networks are saved as layer0_layer1_layer2 named files
    # (layer numbers separated by underscores)
    # Assume all file names are different
    all_files_results = collections.defaultdict(dict) # {compinv: {layerset: [repetitions]}}
    for filename in sorted(file_list):
        sorted_layers = tuple(sorted([int(s) for s in filename.split('/')[-1].split('_')]))
        all_repetitions_results = null_model_subgraphs_from_network_file(filename,intralayer_density,interlayer_density,ER_from_net,nnodes,nlayers,50)
        for compinv in all_repetitions_results:
            all_files_results[compinv][sorted_layers] = all_repetitions_results[compinv]
    return all_files_results
