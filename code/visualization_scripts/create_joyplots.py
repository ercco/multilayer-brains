#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:03:28 2023

@author: onerva

A script for creating joyplots. NOTE: joypy requires Python 3, so this should be run with Python 3 unlike other
code of this project.
"""
import pickle
import pandas as pd
import numpy as np
import joypy
import matplotlib.pylab as plt
from databinner import binner

# path parts for reading data
consistencyInNextWindowSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/pooled_data'
jobLabels = ['random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03']
clusteringMethods = ['','','']

# path parts for saving
joyplotSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/consistency_in_next_window_joyplot'

# visualization params
nBins = 50
ignoreSingleVoxelROIs = True
nDecimalsInLabel = 2
timeLag = 1

for jobLabel, clusteringMethod in zip(jobLabels, clusteringMethods):
    if clusteringMethod == '':
        figDataSavePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '.pkl'
        joyplotSavePath = joyplotSaveStem + '_' + jobLabel + '.pdf'
    else:
        figDataSavePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pkl'
        joyplotSavePath = joyplotSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pdf'
    f = open(figDataSavePath, 'rb')
    figData = pickle.load(f)
    f.close()
    presentWindowConsistencies = figData['presentWindowConsistencies']
    nextWindowConsistencies = figData['nextWindowConsistencies']
    
    if ignoreSingleVoxelROIs:
        for c1, c2 in zip(presentWindowConsistencies, nextWindowConsistencies):
            if any([c1 == 1, c2 == 1]):
                presentWindowConsistencies.remove(c1)
                nextWindowConsistencies.remove(c2)
    
    presentWindowConsistencies = np.array(presentWindowConsistencies)
    nextWindowConsistencies = np.array(nextWindowConsistencies)
    
    bins = binner.Bins(float, min(min(presentWindowConsistencies), min(nextWindowConsistencies)), max(max(presentWindowConsistencies), max(nextWindowConsistencies)), 'lin', nBins)
    bin_limits = bins.bin_limits
   
    df = pd.DataFrame({})
    for i in range(nBins):
        btm = bin_limits[i]
        top = bin_limits[i + 1]
        label = str(np.around(btm + 0.5*(top - btm), nDecimalsInLabel))
        presentIndices = np.where((btm <= presentWindowConsistencies) & (presentWindowConsistencies < top))
        nextValues = nextWindowConsistencies[presentIndices]
        if i == nBins - 1:
            lastLimitIndices = np.where(presentWindowConsistencies == bin_limits[-1])
            lastLimitValues = nextWindowConsistencies[lastLimitIndices]
            nextValues = list(nextValues)
            nextValues.extend(list(lastLimitValues))
            nextValues = np.array(nextValues)
        if len(nextValues) == 0:
            continue # there are no next window consistency values in this bin
        df = pd.concat([df, pd.DataFrame({label: nextValues})], axis=1)
    fig, ax = joypy.joyplot(df, kind='counts', title='%s, consistency %x windows after the present' %(jobLabel, timeLag))
    plt.savefig(joyplotSavePath,format='pdf',bbox_inches='tight')
