# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:31:45 2018

@author: onerva

A script for validating the consistency-based optimization of ROIs by comparing
the consistency of optimized ROIs to the consistency of ROIs of the Brainnetome
parcellation.

In future, this script should calculate the consistency distributions form pooled
data of all subjects.
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
else:
    sys.path.insert(0,'/home/onerva/projects/dippa/local')     

import clustering_by_consistency as cbc
import onion_parameters as params

subjects = ['/media/onerva/KINGSTON/test-data/010/']
originalROIInfoFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
optimizedROIInfoFile = 'optimized-rois-test-for-Tarmo-mean-weighted-consistency-thresholded-voxelwise'
allVoxelTsFileName = '/roi_voxel_ts_all_rois4mm_FWHM0.mat'
originalSpatialConsistencySavePath = '/media/onerva/KINGSTON/test-data/spatial-consistency-original.pkl'
optimizedSpatialConsistencySaveName = '/spatial-consistency-optimized-test-for-Tarmo-mean-weighted-consistency-thresholded-voxelwise.pkl'
originalCorrelationSavePath = '/media/onerva/KINGSTON/test-data/correlation-to-centroid-original.pkl'
optimizedCorrelationSaveName = 'correlation-to-centroid-spatial-consistency-optimized-test-for-Tarmo-mean-weighted-consistency-thresholded-voxelwise.pkl'
figureSavePath = '/media/onerva/KINGSTON/test-data/outcome/spatial-consisistency-validation-mean-weighted-consistency-thresholded-voxelwise.pdf'
sizeSavePath = '/media/onerva/KINGSTON/test-data/outcome/spatial-consisistency-validation-mean-weighted-consistency-thresholded-voxelwise-sizes.pdf'

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
    pdf = count/float(np.sum(count)*(binEdges[1]-binEdges[0]))
    binCenters = 0.5*(binEdges[:-1]+binEdges[1:])
    
    return pdf, binCenters

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

sizeFig = plt.figure(2)
sizeAx = sizeFig.add_subplot(111)

originalCentroids,_,originalVoxelCoordinates,originalROIMaps = cbc.readROICentroids(originalROIInfoFile,readVoxels=True,fixCentroids=True)
centroidIndices = np.zeros(len(originalCentroids),dtype=int)
for i, originalCentroid in enumerate(originalCentroids):
    centroidIndices[i] = np.where((originalVoxelCoordinates==originalCentroid).all(axis=1)==1)[0][0]

for i, subject in enumerate(subjects):
    optimizedROIInfoPath = subject + optimizedROIInfoFile + '.pkl'
    tempPath = subject + optimizedROIInfoFile + '_temp.pkl'
    allVoxelTsPath = subject + allVoxelTsFileName
    
    allVoxelTs = io.loadmat(allVoxelTsPath)['roi_voxel_data'][0]['roi_voxel_ts'][0]
    f = open(optimizedROIInfoPath, "rb")
    ROIInfo = pickle.load(f)
    f.close()
    
    if not 'ROIMaps' in ROIInfo.keys():
        ROIInfo = cbc.voxelLabelsToROIInfo(ROIInfo['ROILabels'],ROIInfo['voxelCoordinates'],constructROIMaps=True)
        with open(tempPath, 'wb') as f:
            pickle.dump(ROIInfo, f, -1)
    else:
        tempPath = optimizedROIInfoPath
    
    
    _,_,_,ROIMaps = cbc.readROICentroids(tempPath,readVoxels=True)
    sizes = [len(ROIMap) for ROIMap in ROIMaps]
    
    ROIIndices = []
    for ROIMap in ROIMaps:
        #indices = np.zeros(len(ROIMap),dtype=int)
        if len(ROIMap.shape) == 1:
            indices = np.where((originalVoxelCoordinates == ROIMap).all(axis=1)==1)[0]
        else:
            indices = np.zeros(len(ROIMap),dtype=int)
            for j, voxel in enumerate(ROIMap):
                #print 'something'
                indices[j] = np.where((originalVoxelCoordinates == voxel).all(axis=1)==1)[0][0]
        ROIIndices.append(indices)
        
    spatialConsistencies = cbc.calculateSpatialConsistencyInParallel(ROIIndices,allVoxelTs)
    spatialConsistencyData = {'spatialConsistencies':spatialConsistencies,'type':'optimized'}
    savePath = subject + optimizedSpatialConsistencySaveName
    with open(savePath, 'wb') as f:
        pickle.dump(spatialConsistencyData, f, -1)
    print('Mean optimized consistency: ' + str(np.mean(spatialConsistencies)))
    print('Number of voxels in optimized parcellation: ' + str(sum(sizes)))
    consistencyDistribution,consistencyBinCenters = getDistribution(spatialConsistencies,params.nConsistencyBins)
    sizeDist, sizeBinCenters = getDistribution(sizes,params.nSizeBins)
    
#    correlationsToCentroid = functions.calculateCorrelationsToCentroidInParallel(ROIIndices,allVoxelTs,centroidIndices)
#    correlationData = {'correlationsToCentroid':correlationsToCentroid,'type':'optimized'}
#    savePath = subject + optimizedCorrelationSaveName
#    with open(savePath, 'wb') as f:
#        pickle.dump(correlationData, f, -1)
#    correlationDistribution,correlationBinCenters = cbc.getDistribution(correlationsToCentroid,params.nConsistencyBins)
    
    if i == 0:
        ax1.plot(consistencyBinCenters,consistencyDistribution,color=params.optimizedColor,alpha=params.optimizedAlpha,label='Optimized ROIs')
        sizeAx.plot(sizeBinCenters,sizeDist,color=params.optimizedColor,alpha=params.optimizedAlpha,label='Optimzed ROIs')
#        ax2.plot(correlationBinCenters,correlationDistribution,color=params.optimizedColor,alpha=params.optimizedAlpha,label='Optimized ROIs')
    else:
        ax1.plot(consistencyBinCenters,consistencyDistribution,color=params.optimizedColor,alpha=params.optimizedAlpha)
        sizeAx.plot(sizeBinCenters,sizeDist,color=params.optimizedColor,alpha=params.optimizedAlpha)
#        ax2.plot(correlationBinCenters,correlationDistribution,color=params.optimizedColor,alpha=params.optimizedAlpha)   
#_,_,voxelCoordinates,ROIMaps = functions.readROICentroids(originalROIInfoPath,readVoxels=True)

ROIIndices = []
for ROIMap in originalROIMaps:
    indices = np.zeros(len(ROIMap),dtype=int)
    for i, voxel in enumerate(ROIMap):
        indices[i] = np.where((originalVoxelCoordinates == voxel).all(axis=1)==1)[0][0]
    ROIIndices.append(indices)
  
spatialConsistencies = cbc.calculateSpatialConsistencyInParallel(ROIIndices,allVoxelTs)
spatialConsistencyData = {'spaitalConsistencies':spatialConsistencies,'type':'original Brainnetome'}
with open(originalSpatialConsistencySavePath, 'wb') as f:
        pickle.dump(spatialConsistencyData, f, -1)
print('Mean original consistency: ' + str(np.mean(spatialConsistencies)))
consistencyDistribution,consistencyBinCenters = getDistribution(spatialConsistencies,params.nConsistencyBins)
sizes = [len(ROIMap) for ROIMap in originalROIMaps]
print('Number of voxels in original parcellation: ' + str(sum(sizes)))
sizeDist,sizeBinCenters = getDistribution(sizes,params.nSizeBins)

#correlationsToCentroid = cbc.calculateCorrelationsToCentroidInParallel(ROIIndices,allVoxelTs,centroidIndices)
#correlationData = {'correlationsToCentroid':correlationsToCentroid,'type':'original Brainnetome'}
#savePath = subject + optimizedCorrelationSaveName
#with open(savePath, 'wb') as f:
#    pickle.dump(correlationData, f, -1)
#correlationDistribution,correlationBinCenters = cbc.getDistribution(correlationsToCentroid,params.nConsistencyBins)

ax1.plot(consistencyBinCenters,consistencyDistribution,color=params.originalColor,alpha=params.originalAlpha,label='Original ROIs')
#ax2.plot(correlationBinCenters,correlationDistribution,color=params.originalColor,alpha=params.originalAlpha,label='Original ROIs')
sizeAx.plot(sizeBinCenters,sizeDist,color=params.originalColor,alpha=params.originalAlpha,label='Original ROIs')

plt.figure(1)
ax1.set_xlabel('Spatial consistency')
ax1.set_ylabel('PDF')
ax1.legend()

#ax2.set_xlabel('Correllation to ROI centroid')
#ax2.set_ylabel('PDF')
#ax2.legend()

plt.tight_layout()
plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')

plt.figure(2)
sizeAx.set_xlabel('ROI size')
sizeAx.set_ylabel('PDF')
sizeAx.legend()

plt.tight_layout()
plt.savefig(sizeSavePath,formt='pdf',bbox_inches='tight')
    
