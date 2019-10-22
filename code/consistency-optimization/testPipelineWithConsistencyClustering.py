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
timewindow = 100
overlap = 0
run_number = 1
intralayer_density = 0.1
interlayer_density = 0.1
subgraph_size_dict = {2:(2,3)} # Let's look for all subgraphs of two layers, two nodes and two layers, three nodes
allowed_aspects = [0]
use_aggregated_dict = True
use_examples_dict = True
clustering_method = 'consistency_optimized'
mask_or_template_filename = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
mask_or_template_name = 'random'
preprocess_level_folder = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/'
nClusters = 100
n_consistency_iters = 5
n_consistency_CPUs = 5
use_random_seeds = True

pipeline.isomorphism_classes_from_nifti(nii_data_filename,subj_id,run_number,timewindow,overlap,
                                        intralayer_density,interlayer_density,subgraph_size_dict,
                                        allowed_aspects,use_aggregated_dict,use_examples_dict,clustering_method,
                                        mask_or_template_filename,mask_or_template_name,nClusters,preprocess_level_folder=preprocess_level_folder,
                                        calculate_consistency=True,n_consistency_iters=n_consistency_iters,n_consistency_CPUs=n_consistency_CPUs,
                                        use_random_seeds=use_random_seeds)

