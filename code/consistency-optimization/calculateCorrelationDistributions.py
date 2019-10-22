#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:10:32 2019

@author: onerva

A script for calculating the distribution of voxel-voxel correlations inside and
between ROIs for multiple parcellations.
"""
import numpy as np
import matplotlib.pylab as plt
from scipy import io
import cPickle as pickle

from scipy.stats import binned_statistic

import os.path
import sys
if os.path.exists('/home/onerva/dippa/local'):
    sys.path.insert(0,'/home/onerva/dippa/local')
elif os.path.exists('home/onerva/projects/dippa/local'):
    sys.path.insert(0,'/home/onerva/projects/dippa/local')
else:
    sys.path.insert(0,'/scratch/cs/networks/aokorhon/dippa/code')     

import clustering_by_consistency as cbc
import onion_parameters as params

subjects = ['/scratch/cs/networks/aokorhon/multilayer/010/']
originalROIInfoFile = '/scratch/cs/networks/aokorhon/multilayer/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
optimizedROIInfoFile = 'optimized-rois-test-for-Tarmo-mean-weighted-consistency'
allVoxelTsFileName = '/roi_voxel_ts_all_rois4mm_FWHM0.mat'
figureSavePath = '/scracth/cs/networks/aokorhon/multilayer/outcome/correlation-distributions-weighted-mean-consistency.pdf'
nBins = 100

originalColor = params.originalColor
originalAlpha = params.originalAlpha
optimizedColor = params.optimizedColor
optimizedAlpha = params.optimizedAlpha
inROILs = params.inROILs
betweenROILs = params.betweenROILs

def getDistribution(data, nBins):
    """
    Calculates the PDF of the given data
    
    Parameters:
    -----------
    data: a container of data points, e.g. list or np.array
    nBins: int, number of bins used to calculate the distribution
    
    Returns:
    --------
    pdf: np.array, PDF of the data
    binCenters: np.array, points where pdf has been calculated
    """
    count, binEdges, _ = binned_statistic(data, data, statistic='count', bins=nBins)
    pdf = count/float(np.sum(count*(binEdges[1]-binEdges[0])))
    binCenters = 0.5*(binEdges[:-1]+binEdges[1:])
    
    return pdf, binCenters

def getInAndBwROIMasks(inROIIndices):
    """
    Returns the masks for picking correlations within and between rois from
    a voxel-voxel correlation matrix where voxels are ordered by their ROI identity
    
    Parameters:
    -----------
    inROIIndices: list of arrays; indices of voxels belonging to each ROI
    
    Returns:
    --------
    withinROIMask, betweenROIMask: nROIs x nROIs arrays where elements a) inside a ROI
                                   or b) between ROIs are 1, others 0
    """
    inROIxIndices = []
    inROIyIndices = []
    nVoxels = 0
    offset = 0
    for ROI in inROIIndices:
        s = len(ROI)
        nVoxels = nVoxels + s
        template = np.zeros((s, s))
        triu = np.triu_indices_from(template, 1)
        inROIxIndices.extend(triu[0] + offset)
        inROIyIndices.extend(triu[1] + offset)
        offset = offset + s
    withinROIMask = np.zeros((nVoxels,nVoxels))
    withinROIMask[inROIxIndices, inROIyIndices] = 1
    betweenROIMask = np.ones((nVoxels,nVoxels))
    betweenROIMask = np.triu(betweenROIMask - withinROIMask - np.eye(nVoxels))
    return (withinROIMask, betweenROIMask)

# Let's first handle the original ROI maps (that can be done outside of the loops)
print('set-up ready, starting...')
# Reading ROI maps and voxel coordinates
_,_,originalVoxelCoordinates,originalROIMaps = cbc.readROICentroids(originalROIInfoFile,readVoxels=True)

# Finding location of ROI's voxels in the voxel time series list (the list is in the same order as originalVoxelCoordinates)
originalROIIndices = []
for ROIMap in originalROIMaps:
    indices = np.zeros(len(ROIMap),dtype=int)
    for i, voxel in enumerate(ROIMap):
        indices[i] = np.where((originalVoxelCoordinates == voxel).all(axis=1)==1)[0][0]   
    originalROIIndices.append(indices)
originalROIIndices = originalROIIndices
# Correlations need to be calculated for each subject separately
originalInROICorrelations = []
optimizedInROICorrelations = []
originalBetweenROICorrelations = []
optimizedBetweenROICorrelations = []
for subject in subjects:
    # reading voxel ts:
    allVoxelTsPath = subject + allVoxelTsFileName
    allVoxelTs = io.loadmat(allVoxelTsPath)['roi_voxel_data'][0]['roi_voxel_ts'][0]
    allVoxelCorrelations = np.corrcoef(allVoxelTs)
    print('correlation matrix ready!')
    # let's calculate correlations for original ROIs first:
    withinROIMask, betweenROIMask = getInAndBwROIMasks(originalROIIndices)
    withinROICorrs = allVoxelCorrelations[np.where(withinROIMask > 0)]
    betweenROICorrs = allVoxelCorrelations[np.where(betweenROIMask > 0)]
    originalInROICorrelations.extend(withinROICorrs)
    originalBetweenROICorrelations.extend(betweenROICorrs)
    print('original ROIs done!')    

    # next, let's calculate correlations for the optimized ROIs
    # reading the optimized ROI info
    optimizedROIInfoPath = subject + optimizedROIInfoFile + '.pkl'
    tempPath = subject + optimizedROIInfoFile + '_temp.pkl'
    f = open(optimizedROIInfoPath, "rb")
    ROIInfo = pickle.load(f)
    f.close()
    # some manipuation, depending on the format of optimized ROI info...
    if not 'ROIMaps' in ROIInfo.keys():
        ROIInfo = cbc.voxelLabelsToROIInfo(ROIInfo['ROILabels'],ROIInfo['voxelCoordinates'],constructROIMaps=True)
        with open(tempPath, 'wb') as f:
            pickle.dump(ROIInfo, f, -1)
    else:
        tempPath = optimizedROIInfoPath
    
    _,_,_,optimizedROIMaps = cbc.readROICentroids(tempPath,readVoxels=True)
    # Finding location of ROI's voxels in the voxel time series list (the list is in the same order as originalVoxelCoordinates)
    optimizedROIIndices = []
    for ROIMap in optimizedROIMaps:
        if len(ROIMap.shape) == 1:
            indices = np.where((originalVoxelCoordinates == ROIMap).all(axis=1)==1)[0]
        else:
            indices = np.zeros(len(ROIMap),dtype=int)
            for j, voxel in enumerate(ROIMap):
                #print 'something'
                indices[j] = np.where((originalVoxelCoordinates == voxel).all(axis=1)==1)[0][0]
        optimizedROIIndices.append(indices)
    optimizedWithinROIMask, optimizedBetweenROIMask = getInAndBwROIMasks(optimizedROIIndices)
    print(allVoxelCorrelations.shape,optimizedWithinROIMask.shape)
    # reordering the correlation array by voxels' ROI identity (mask creation functions assume this ordering)
    sortedAllVoxelCorrelations = allVoxelCorrelations[np.concatenate(optimizedROIIndices),:][:,np.concatenate(optimizedROIIndices)]
    print(sortedAllVoxelCorrelations.shape)
    optimizedWithinROICorrs = sortedAllVoxelCorrelations[np.where(optimizedWithinROIMask > 0)]
    optimizedBetweenROICorrs = sortedAllVoxelCorrelations[np.where(optimizedBetweenROIMask > 0)]
    optimizedInROICorrelations.extend(optimizedWithinROICorrs)
    optimizedBetweenROICorrelations.extend(optimizedBetweenROICorrs)
    print('optimized ROIs done!')

# Calculating and plotting distributions
print('calculating distributions...')
originalWithinROIDist,originalWithinBinCenters = getDistribution(originalInROICorrelations,nBins)
originalBetweenROIDist,originalBetweenBinCenters = getDistribution(originalBetweenROICorrelations,nBins)
print(len(optimizedInROICorrelations))
print(len(optimizedBetweenROICorrelations))
optimizedWithinROIDist,optimizedWithinBinCenters = getDistribution(optimizedInROICorrelations,nBins)
optimizedBetweenROIDist,optimizedBetweenBinCenters = getDistribution(optimizedBetweenROICorrelations,nBins)

data = {'originalInROI':(originalWithinROIDist,originalWithinBinCenters), 'originalBetweenROI':(originalBetweenROIDist,originalBetweenBinCenters),'optimizedInROI':(optimizedWithinROIDist,optimizedWithinBinCenters), 'optimizedBetweenROI':(optimizedBetweenROIDist,optimizedBetweenBinCenters)}

with open('/scratch/cs/networks/aokorhon/multilayer/outcome/correlation-distributions-weighted-mean-consistency.pkl', 'wb') as f:
    pickle.dump(data, f, -1)


#fig = plt.figure()
#ax = fig.add_subplot(111)

#plt.plot(originalWithinBinCenters,originalWithinROIDist,color=originalColor,alpha=originalAlpha,ls=inROILs,label='original, inside ROIs')
#plt.plot(originalBetweenBinCenters,originalBetweenROIDist,color=originalColor,alpha=originalAlpha,ls=betweenROILs,label='original, between ROIs')
#plt.plot(optimizedWithinBinCenters,optimizedWithinROIDist,color=optimizedColor,alpha=optimizedAlpha,ls=inROILs,label='optimized, inside ROIs')
#plt.plot(optimizedBetweenBinCenters,optimizedBetweenROIDist,color=optimizedColor,alpha=optimizedAlpha,ls=betweenROILs,label='optimized, between ROIs')

#ax.set_xlabel('Pearson correlation coefficient')
#ax.set_ylabel('PDF')
#ax.legend()

#plt.tight_layout()
#plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')

    

