#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:55:34 2023

@author: onerva

A script for creating the schematic figure (Fig. 1)
"""
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt

from surfplot import Plot
from surfplot.utils import threshold
from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps.transforms import mni152_to_fslr, mni152_to_fsaverage, fsaverage_to_fslr
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from brainspace.mesh.mesh_io import read_surface

#from brainspace.datasets import load_parcellation

load_data = True
#surface_folder = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/surf/'
#surface_type = 'pial'
template_path = '/home/onerva/projects/ROIplay/templates/brainnetome/BNA-MPM_thr25_4mm.nii'
#data_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/test_slice.nii'
#data_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/test_slice'
spherical_parcellation_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/spherical_parcellation.nii'
consistency_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/consistency.nii'
spherical_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_spherical.pdf'
brainnetome_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_brainnetome.pdf'

simulation_length = 1000
distance_threshold = 10
sigma = 5

# loading surfaces

#lh_surface_path = surface_folder + 'lh.' + surface_type
#rh_surface_path = surface_folder + 'rh.' + surface_type
#lh = read_surface(lh_surface_path, itype='fs')
#rh = read_surface(rh_surface_path, itype='fs')

#p = Plot(lh, rh)

# functions

def gaussian(x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu)/sigma)**2)

# NOTE: the following functions have been copied from clustering_by_consistency.
# Importing them is not possible since clustering_by_consistency is written in Python 2.7.
# For documentation, see clustering_by_consistency

def get_distance_matrix(ROI_centroids, voxel_coordinates):
    if len(ROI_centroids) == 1:
        nROIs = 1
    else:
        nROIs = len(ROI_centroids)
    nVoxels = len(voxel_coordinates)
    distance_matrix = np.zeros((nROIs, nVoxels))
    if nROIs == 1:
        distance_matrix[0,:] = np.sqrt(np.sum((voxel_coordinates-ROI_centroids)**2,axis=1))
    else:
        for i, centroid in enumerate(ROI_centroids):
            distance_matrix[i,:] = np.sqrt(np.sum((voxel_coordinates-centroid)**2,axis=1))
    return distance_matrix

def find_neighbors(voxel_coordinates, resolution=1, all_voxels=[], n_neighbors=6):

    x = voxel_coordinates[0]
    y = voxel_coordinates[1]
    z = voxel_coordinates[2]    
    

    neighbors = [[x+resolution,y,z],
                [x-resolution,y,z],
                [x,y+resolution,z],
                [x,y-resolution,z],
                [x,y,z+resolution],
                [x,y,z-resolution]]
    if n_neighbors == 18:
        neighbors.extend([[x+resolution,y+resolution,z],
                          [x+resolution,y-resolution,z],
                          [x-resolution,y+resolution,z],
                          [x-resolution,y-resolution,z],
                          [x+resolution,y,z+resolution],
                          [x+resolution,y,z-resolution],
                          [x-resolution,y,z+resolution],
                          [x-resolution,y,z-resolution],
                          [x,y+resolution,z+resolution],
                          [x,y+resolution,z-resolution],
                          [x,y-resolution,z+resolution],
                          [x,y-resolution,z-resolution]])
    if n_neighbors == 26:
        neighbors.extend([[x+resolution,y+resolution,z+resolution],
                          [x+resolution,y+resolution,z-resolution],
                          [x+resolution,y-resolution,z+resolution],
                          [x+resolution,y-resolution,z-resolution],
                          [x+resolution,y-resolution,z-resolution],
                          [x-resolution,y+resolution,z+resolution],
                          [x-resolution,y+resolution,z-resolution],
                          [x-resolution,y-resolution,z+resolution],
                          [x-resolution,y-resolution,z-resolution]])
                         
    if not len(all_voxels) == 0:
        accepted_neighbors = []
        for i, neighbor in enumerate(neighbors):
            if np.any((np.array(all_voxels) == neighbor).all(axis=1)):
                accepted_neighbors.append(neighbor)
        neighbors = accepted_neighbors                 
    return neighbors

def find_ROIless_voxels(voxel_coordinates, ROI_info):
    ROI_maps = ROI_info['ROIMaps']
    for i, ROI in enumerate(ROI_maps):
        if len(ROI.shape) == 1: # adding an extra dimension to enable concatenation
            ROI = np.array([ROI]) 
        if i == 0:
            in_ROI_voxels = ROI
        else:
            in_ROI_voxels = np.concatenate((in_ROI_voxels,ROI),axis=0)
    ROIless_indices = []
    ROIless_map = []
    for i, voxel in enumerate(voxel_coordinates):
        if not np.any((in_ROI_voxels == voxel).all(axis=1)): # voxel is not found in any ROI map
            ROIless_indices.append(i)
            ROIless_map.append(voxel)
    ROIless_indices = np.array(ROIless_indices)
    ROIless_map = np.array(ROIless_map)
    ROIless_voxels = {'ROIlessIndices':ROIless_indices,'ROIlessMap':ROIless_map}
    return ROIless_voxels

def find_ROIless_neighbors(ROI_index, voxel_coordinates, ROI_info):
    ROI_map = ROI_info['ROIMaps'][ROI_index]
    if len(ROI_map.shape) == 1: # adding an outermost dimension to enable proper indexing later on
        ROI_map = [ROI_map]
    for i, voxel in enumerate(ROI_map):
        if i == 0:
            ROI_neighbors = find_neighbors(voxel,all_voxels=voxel_coordinates)
        else:
            ROI_neighbors.extend(find_neighbors(voxel,all_voxels=voxel_coordinates))
    if len(ROI_neighbors) > 0:
        ROI_neighbors = np.unique(ROI_neighbors,axis=0) # removing dublicates
        ROIless_neighbors = find_ROIless_voxels(ROI_neighbors,ROI_info) 
        ROIless_map = ROIless_neighbors['ROIlessMap']
        ROIless_indices = np.zeros(ROIless_map.shape[0],dtype=int) # indices in the list of neighbors
        for i, voxel in enumerate(ROIless_map):
            ROIless_indices[i] = np.where((voxel_coordinates==voxel).all(axis=1)==1)[0][0] # finding indices in the voxelCoordinates array (indexing assumes that a voxel is present in the voxelCoordinates only once)
    else:
        ROIless_map = np.array([])
        ROIless_indices = np.array([])
    ROIless_neighbors = {'ROIlessIndices':ROIless_indices,'ROIlessMap':ROIless_map}
    return ROIless_neighbors

def add_voxel(ROI_index, voxel_index, ROI_info, voxel_coordinates):
    ROI_map = ROI_info['ROIMaps'][ROI_index]
    ROI_voxels = ROI_info['ROIVoxels'][ROI_index]
    voxel = np.array([voxel_coordinates[voxel_index]])
    if len(ROI_map.shape) == 1: # adding an outermost dimension for successful concatenation later on
            ROI_map = np.array([ROI_map])
    ROI_map = np.concatenate((ROI_map,voxel),axis=0)
    if isinstance(ROI_voxels,list):
        ROI_voxels.append(voxel_index)
    else:
        ROI_voxels = np.concatenate((ROI_voxels,np.array([voxel_index])),axis=0)
    ROI_info['ROIMaps'][ROI_index] = ROI_map
    ROI_info['ROIVoxels'][ROI_index] = ROI_voxels
    if 'ROISizes' in ROI_info.keys():
        ROI_info['ROISizes'][ROI_index] = len(ROI_voxels)
    return ROI_info

def grow_spherical_ROIs(ROI_centroids, imgdata):
    nROIs = len(ROI_centroids)
    
    voxel_coordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    voxel_labels = np.zeros(len(voxel_coordinates)) - 1
    
    distance_matrix = get_distance_matrix(ROI_centroids, voxel_coordinates) # distanceMatrix = nROIs x nVoxel
    distance_mask = np.ones(distance_matrix.shape)
        
    ROI_voxels = []
    for ROI_index, centroid in enumerate(ROI_centroids):
        centroid_index = np.where((voxel_coordinates==centroid).all(axis=1)==1)[0]
        ROI_voxels.append(np.array(centroid_index))
        voxel_labels[centroid_index] = ROI_index
        distance_mask[:,centroid_index] = 0
    
    ROI_maps = [np.array(centroid) for centroid in ROI_centroids]
    ROI_info = {'ROIMaps':ROI_maps,'ROIVoxels':ROI_voxels,'ROISizes':np.ones(nROIs,dtype=int),'ROINames':[]}
    ROI_neighbors = np.ones(nROIs) # this is a boolean array telling if a ROI still has neighbors that can be added
    
    max_radius = len(voxel_coordinates)
    
    radius = 1
    
    while np.sum(ROI_neighbors) > 0 and radius < max_radius:
        print('Growing ROIs, radius: ' + str(radius) + ', ' + str(np.sum(ROI_neighbors)) + ' ROIs able to grow')
        for ROI_index, distances in enumerate(distance_matrix):
            distances = distances * distance_mask[ROI_index,:]
            if ROI_neighbors[ROI_index] == 0:
                continue
            distance_neighbors = np.where(((0 < distances) & (distances <= radius)))[0]
            physical_neighbors = set(find_ROIless_neighbors(ROI_index,voxel_coordinates,{'ROIMaps':ROI_maps})['ROIlessIndices'])
            neighbors = [neighbor for neighbor in distance_neighbors if neighbor in physical_neighbors]
            if len(neighbors) == 0:
                ROI_neighbors[ROI_index] = 0
                continue
            for neighbor in neighbors:
                ROI_info = add_voxel(ROI_index,neighbor,ROI_info,voxel_coordinates)
                voxel_labels[neighbor] = ROI_index
                distance_mask[:,neighbor] = 0 # masking away voxels that have already been added to a ROI
        radius = radius + 1
    voxel_labels = np.array([int(label) for label in voxel_labels])
    return voxel_labels, voxel_coordinates, radius

# simulating data

template_img = nib.load(template_path) # the Brainnetome template is used to get space dimensions
template_data = template_img.get_fdata()

simulated_data = np.zeros((template_data.shape[0], template_data.shape[1], template_data.shape[2], simulation_length))
mask = np.transpose(np.array(np.where(template_data > 0)))
for voxel in mask:
    simulated_data[voxel[0], voxel[1], voxel[2], :] = np.random.rand(simulation_length)
n_seeds = len(np.unique(template_data)) - 1
seeds = mask[np.random.choice(mask.shape[0], n_seeds, replace=False)]
seed_ts = simulated_data[seeds[:,0], seeds[:,1], seeds[:,2], :]
distance_matrix = get_distance_matrix(seeds, mask)
for i, voxel in enumerate(mask):
    if not np.any(np.all(voxel==seeds, axis=1)):
        multipliers = np.zeros(n_seeds)
        for j, seed in enumerate(seeds):
            d = distance_matrix[j, i]
            if d < distance_threshold:
                multipliers[j] = gaussian(d, 0, sigma)
        simulated_data[voxel[0], voxel[1], voxel[2], :] += np.sum(multipliers * np.transpose(seed_ts), axis = 1)
        simulated_data[voxel[0], voxel[1], voxel[2], :] = simulated_data[voxel[0], voxel[1], voxel[2], :] / np.sum(simulated_data[voxel[0], voxel[1], voxel[2], :])

# creating "optimized ROIs" (= spheres around the centers)

voxel_labels, voxel_coordinates, _ = grow_spherical_ROIs(seeds, simulated_data)

# calculating consistencies per voxel (= the average correlation of each voxel to other voxel's in its ROI)
consistency = np.zeros(simulated_data.shape[0:3])
spherical_parcellation = np.zeros(simulated_data.shape[0:3])
ROI_labels = np.unique(voxel_labels)
for ROI in ROI_labels:
    if ROI >= 0:
        ROI_voxels = np.array(voxel_coordinates)[np.where(voxel_labels == ROI)[0], :]
        ROI_size = ROI_voxels.shape[0]
        ROI_voxel_ts = simulated_data[ROI_voxels[:,0], ROI_voxels[:,1], ROI_voxels[:,2], :]
        correlations = np.corrcoef(ROI_voxel_ts)
        for i, voxel in enumerate(ROI_voxels):
            consistency[voxel[0], voxel[1], voxel[2]] = (np.sum(correlations[i, :]) - 1) / (ROI_size - 1) 
            spherical_parcellation[voxel[0], voxel[1], voxel[2]] = ROI

# writing spherical ROIs and consistency to .nii

template = nib.load(template_path)
affine = template.affine
header = nib.Nifti1Header()
spherical_parcellation_image = nib.Nifti1Image(spherical_parcellation, affine, header)
nib.save(spherical_parcellation_image, spherical_parcellation_save_path)
consistency_image = nib.Nifti1Image(consistency, affine, header)
nib.save(consistency_image, consistency_save_path)

# plotting consistency data

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

consistency_gii_lh, consistency_gii_rh = mni152_to_fslr(consistency_save_path)

consistency_lh = consistency_gii_lh.agg_data()
consistency_rh = consistency_gii_rh.agg_data()

p = Plot(lh, rh)
p.add_layer({'left': consistency_lh, 'right': consistency_rh}, cmap=nilearn_cmaps['cold_hot'])

#fig = p.build()

# plotting the spherical parcellation boundaries on brain surface

spherical_gii_lh, spherical_gii_rh = mni152_to_fslr(spherical_parcellation_save_path, method='nearest')
spherical_lh = threshold(spherical_gii_lh.agg_data(), 1)
spherical_rh = threshold(spherical_gii_rh.agg_data(), 1)

p.add_layer({'left': spherical_lh, 'right': spherical_rh}, cmap='gray', as_outline=True, cbar=False)

fig = p.build()

plt.savefig(spherical_figure_save_path, format='pdf',bbox_inches='tight')

# plotting consistency data

consistency_gii_lh, consistency_gii_rh = mni152_to_fslr(consistency_save_path)

consistency_lh = consistency_gii_lh.agg_data()
consistency_rh = consistency_gii_rh.agg_data()

p = Plot(lh, rh)
p.add_layer({'left': consistency_lh, 'right': consistency_rh}, cmap=nilearn_cmaps['cold_hot'])

#fig = p.build()

# plotting the Brainnetome ROI boundaries on brain surface

template_gii_lh, template_gii_rh = mni152_to_fslr(template_path, method='nearest')
template_lh = threshold(template_gii_lh.agg_data(), 1)
template_rh = threshold(template_gii_rh.agg_data(), 1)

p.add_layer({'left': template_lh, 'right': template_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(brainnetome_figure_save_path, format='pdf',bbox_inches='tight')



