#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:13:07 2024

@author: onerva

A script for simulating data used in the schematic figure and creating the related parcellations.
The simulated data is created by selecting n_seeds random voxels, constructing spherical ROIs around
them, and simulated for each voxel of each ROI a time series by mixing the time series of the ROI centroid
and white noise. The weight of the centroid time series in the voxel time series decays following a Gaussin
function of the distance from the centroid.
"""
import numpy as np
import nibabel as nib
import pickle

from clustering_by_consistency import growSphericalROIs, growOptimizedROIs, constrainedReHoSearch
from ROIplay import writeNii

template_path = '/m/cs/scratch/networks/aokorhon/ROIplay/templates/brainnetome/BNA-MPM_thr25_4mm.nii'
underlying_parcellation_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/spherical_parcellation.nii'
underlying_voxel_labels_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/spherical_parcellation_labels.pkl'
optimized_voxel_labels_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/optimized_parcellation_labels.pkl'
optimized_parcellation_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/optimized_parcellation.nii'
ReHo_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/ReHo.nii'

simulation_length = 500
n_seeds = 246
sigma = 1
new_seeds = False # set new_seeds to True to change the location or number of seeds

def gaussian(x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi)*sigma) * np.exp(-1/2 * ((x - mu)/sigma)**2)

template_img = nib.load(template_path) # the template is used to get dimensions and to locate voxels belonging to the gray matter
template_data = template_img.get_fdata()

# defining spherical "underlying" ROIs
mask = np.transpose(np.array(np.where(template_data > 0)))

if new_seeds:
    seeds = mask[np.random.choice(mask.shape[0], n_seeds, replace=False)]
    voxel_labels, voxel_coordinates, _ = growSphericalROIs(seeds, np.expand_dims(template_data, axis=3))
    with open(underlying_voxel_labels_save_path, 'wb') as f:
        pickle.dump({'voxel_labels':voxel_labels, 'voxel_coordinates':voxel_coordinates, 'ROI_seeds':seeds}, f, -1)
else:
    f = open(underlying_voxel_labels_save_path, 'rb')
    parcellation_data = pickle.load(f)
    f.close()
    voxel_labels = parcellation_data['voxel_labels']
    voxel_coordinates = parcellation_data['voxel_coordinates']
    seeds = parcellation_data['ROI_seeds']
    
# simulating data
underlying_parcellation = -1 * np.ones((template_img.shape[0], template_img.shape[1], template_img.shape[2]))
simulated_data = np.zeros((template_img.shape[0], template_img.shape[1], template_img.shape[2], simulation_length))
for seed in seeds:
    simulated_data[seed[0], seed[1], seed[2], :] = np.random.rand(simulation_length)

ROI_indices = np.unique(voxel_labels)
for ROI in ROI_indices:
    if ROI < 0:
        continue
    else:
        seed = seeds[ROI]
        ROI_voxels = np.array(voxel_coordinates)[np.where(voxel_labels == ROI)[0], :]
        seed_ts = simulated_data[seed[0], seed[1], seed[2], :]
        for voxel in ROI_voxels:
            d = np.sqrt((voxel[0] - seed[0])**2 + (voxel[1] - seed[1])**2 + (voxel[2] - seed[2])**2)
            centroid_weight = gaussian(d, 0, sigma) / gaussian(0, 0, sigma)
            simulated_data[voxel[0], voxel[1], voxel[2], :] = centroid_weight * seed_ts + (1 - centroid_weight) * np.random.rand(simulation_length)
            underlying_parcellation[voxel[0], voxel[1], voxel[2]] = ROI
            
writeNii(underlying_parcellation, template_path, underlying_parcellation_save_path)

# obtaining "optimized ROIs" from the simulated data
cfg = {'names':'','imgdata':simulated_data,
       'threshold':'voxel-wise','targetFunction':'weighted mean consistency',
       'fTransform':False,'nROIs':246,'template':template_data,
       'percentageROIsForThresholding':0.3,
       'sizeExp':1,'nCPUs':5,
       'nReHoNeighbors':6,'percentageMinCentroidDistance':0.1,
       'ReHoMeasure':'ReHo','includeNeighborhoodsInCentroids':False,
       'returnExcludedToQueue':False,'regularization':-100,'regExp':2,'logging':False}

ROI_centroids, centroid_ReHos, ReHo_data = constrainedReHoSearch(cfg['imgdata'],cfg['template'],cfg['nROIs'],
                                                                 cfg['nReHoNeighbors'],cfg['nCPUs'],cfg['ReHoMeasure'],
                                                                 fTransform=cfg['fTransform'], returnReHoValues=False,
                                                                 returnReHoData=True)
writeNii(ReHo_data, template_path, ReHo_save_path)
cfg['ROICentroids'] = ROI_centroids
optimized_voxel_labels, optimized_voxel_coordinates = growOptimizedROIs(cfg)
with open(optimized_voxel_labels_save_path, 'wb') as f:
        pickle.dump({'voxel_labels':optimized_voxel_labels, 'voxel_coordinates':optimized_voxel_coordinates}, f, -1)

optimized_parcellation = np.zeros((template_img.shape[0], template_img.shape[1], template_img.shape[2]))
ROI_indices = np.unique(optimized_voxel_labels)
for ROI in ROI_indices:
    ROI_voxels = np.array(optimized_voxel_coordinates)[np.where(optimized_voxel_labels == ROI)[0], :]
    for voxel in ROI_voxels:
        optimized_parcellation[voxel[0], voxel[1], voxel[2]] = ROI

writeNii(optimized_parcellation, template_path, underlying_parcellation_save_path)


