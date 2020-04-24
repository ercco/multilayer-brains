# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:02:41 2019

@author: onerva

A script for testing the multilayer construction pipeline by Tarmo Nurmi (pipeline.py)
with the consistency-optimized clustering.
"""
#import sys
#sys.path.insert(0, '/../code/')

import pipeline

nii_data_filename = '/media/onerva/KINGSTON/test-data/010/epi_STD_mask_detrend_fullreg.nii'
subj_id = '010'
run_number = 1

mask_or_template_filename = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
mask_or_template_name = 'random'

timewindow = 100
overlap = 0

# density params
intralayer_density = 0.1
interlayer_density = 0.1
density_params = {'intralayer_density':intralayer_density,'interlayer_density':interlayer_density}

# clustering method params
clustering_method = 'consistency_optimized'
consistency_target_function = 'weighted mean consistency'
nclusters = 246
n_consistency_CPUs = 5
n_consistency_iters = 5
consistency_threshold = 'maximal-voxel-wise' 
craddock_threshold = 0.5 # the correlation threshold used by Craddock et al. 2012
use_random_seeds = False
calculate_consistency_while_clustering = False
consistency_save_path = '/media/onerva/KINGSTON/test-data/010/spatial_consistency_optimized_mean_weighted_consistency_nonthresholded.pkl'
consistency_percentage_ROIs_for_thresholding = 10
clustering_method_params = {'method':clustering_method,'consistency_target_function':consistency_target_function,'consistency_threshold':consistency_threshold,'craddock_threshold':craddock_threshold,'nclusters':nclusters,'calculate_consistency':calculate_consistency_while_clustering,'consistency_save_path':consistency_save_path,'n_consistency_CPUs':n_consistency_CPUs,'n_consistency_iters':n_consistency_iters,'use_random_seeds':use_random_seeds,'centroid_template_filename':mask_or_template_filename}

# Let's look for all subgraphs of two layers, two nodes and two layers, three nodes
nlayers = 2
nnodes = [2,3]
subgraph_size_dict = {2:(2,3)} 
allowed_aspects = [0]

use_aggregated_dict = True
use_examples_dict = True

preprocess_level_folder = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/'

# TODO: check how to define save paths for networks



if True:
    pipeline.isomorphism_classes_from_file(nii_data_filename,
                                           mask_or_template_filename,
                                           timewindow,
                                           overlap,
                                           density_params,
                                           clustering_method_params,
                                           nlayers,
                                           nnodes,
                                           allowed_aspects)
else:
    pipeline.isomorphism_classes_from_nifti(nii_data_filename,subj_id,run_number,timewindow,overlap,
                                        intralayer_density,interlayer_density,subgraph_size_dict,
                                        allowed_aspects,use_aggregated_dict,use_examples_dict,clustering_method,
                                        mask_or_template_filename,mask_or_template_name,nclusters,preprocess_level_folder=preprocess_level_folder,
                                        calculate_consistency=True,n_consistency_iters=n_consistency_iters,n_consistency_CPUs=n_consistency_CPUs,
                                        use_random_seeds=use_random_seeds)

