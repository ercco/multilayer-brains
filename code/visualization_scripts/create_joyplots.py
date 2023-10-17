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

# path parts for reading data
consistencyInNextWindowSaveStem = '/home/onerva/projects/multilayer-meta/pooled_data'
jobLabels = ['craddock']
clusteringMethods = ['']

# path parts for saving
joyplotSaveStem = '/home/onerva/projects/multilayer-meta/consistency_in_next_window_joyplot'

# visualization params
nBins = 50
ignoreSingleVoxelROIs = True

for jobLabel, clusteringMethod in zip(jobLabels, clusteringMethods):
    if clusteringMethod == '':
        figDataSavePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '.pdf'
        joyplotSavePath = joyplotSaveStem + '_' + jobLabel + '.pdf'
    else:
        figDataSavePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pdf'
        joyplotSavePath = joyplotSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pdf'
    f = open(figDataSavePath, 'rb')
    figData = pickle.load(f)
    f.close()
    presentWindowConsistencies = figData['presentWindowConsistencies']
    nextWindowConsistencies = figData['nextWindowConsistencies']
    
    if ignoreSingleVoxelROIs:
        for c1, c2 in zip(presentWindowConsistencies, nextWindowConsistencies):
            if any(c1 == 1, c2 == 1):
                presentWindowConsistencies.remove(c1)
                nextWindowConsistencies.remove(c2)
    
    presentWindowConsistencies = np.array(presentWindowConsistencies)
    nextWindowConsistencies = np.array(nextWindowConsistencies)
    
    bins = np.arange(min(min(presentWindowConsistencies), min(nextWindowConsistencies)), max(max(presentWindowConsistencies), max(nextWindowConsistencies)))
    
    for i in range(nBins):
        btm = bins[i]
        top = bins[i + 1]
        label = str(btm + 0.5*(top - btm))
        presentIndices = np.where((btm <= presentWindowConsistencies) & (presentWindowConsistencies < top))
        nextValues = nextWindowConsistencies[presentIndices]
        if i == nBins - 1:
            lastLimitIndices = np.where(presentWindowConsistencies == bins[-1])
            lastLimitValues = nextWindowConsistencies[lastLimitIndices]
            nextValues = list(nextValues)
            nextValues.extend(list(lastLimitValues))
            nextValues = np.array(nextValues)
        if i == 0:
            df = pd.DataFrame({label: nextValues})
        else:
            df = pd.concat([df, {label: nextValues}])
    
    fig, ax = joypy.joyplot()
    plt.savefig(joyplotSavePath,format='pdf',bbox_inches='tight')