# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:51:14 2019

@author: onerva

This is a script for calculating the mean correlation between voxel and its
neighborhood of varying sizes (= other voxels within a given distance from the
central voxel).
"""
import numpy as np
import matplotlib.pylab as plt
from scipy import io
import cPickle as pickle

from scipy.stats import pearsonr

import os.path
import sys
if os.path.exists('/home/onerva/consistency-as-onion'):
    sys.path.insert(0,'/home/onerva/consistency-as-onion')
else:
    sys.path.insert(0,'/home/onerva/projects/consistency-as-onion')  

#these will be imported from consistency-as-onion
import functions
import onion_parameters as params

visualizeOnly = False
correlationSavePath = '/media/onerva/KINGSTON/test-data/outcome/neighborhood-correlation-vs-radius.pkl'
figureSavePath = '/media/onerva/KINGSTON/test-data/outcome/neighborhood-correlation-vs-radius.pdf'

if not visualizeOnly:
    subjectFolders = ['/media/onerva/KINGSTON/test-data/010/',
                      '/media/onerva/KINGSTON/test-data/011/']
    nSubjects = len(subjectFolders)
    roiInfoFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
    resolution = 4
    distanceMatrixPath = params.distanceMatrixPath
    
    radiusRange = np.arange(1,10,2)
    nRadia = len(radiusRange)

    _, _, voxelCoords, _ = functions.readROICentroids(roiInfoFile,readVoxels=True) # this returns the coordinates of all voxels belonging to ROIs
    nROIs = len(voxelCoords)
    
    correlations = np.zeros((nSubjects,nRadia,nROIs))
    allVoxelTs = []
    
    for i, subject in enumerate(subjectFolders): # reading the data here and not inside the loop to save time
        voxelTsFilePath = subject + params.ROIVoxelTsFileName
        allVoxelTs.append(io.loadmat(voxelTsFilePath)['roi_voxel_data'][0]['roi_voxel_ts'][0])
        
    for j, radius in enumerate(radiusRange):
        if j == 0: # calculating the distance matrix only once
            roiInfo = functions.defineSphericalROIs(voxelCoords,voxelCoords,radius,resolution=resolution,save=False)
        else:
            roiInfo = functions.defineSphericalROIs(voxelCoords,voxelCoords,radius,resolution=resolution,save=False)
        voxelIndices = roiInfo['ROIVoxels']
        for i, subject in enumerate(allVoxelTs):
            for k, ROI in enumerate(voxelIndices):
                print('Calculating radius' + str(radius) + ', subject ' + str(i) + ', voxel ' + str(k))
                centroidTs = subject[k,:]
                neighborTss = subject[ROI,:]
                correlation = np.mean([pearsonr(centroidTs,neighborTs)[0] for neighborTs in neighborTss])
                correlations[i,j,k] = correlation
                
    correlationData = {'radia':radiusRange,'subjects':subjectFolders,'correlations':correlations}   
    with open(correlationSavePath, 'wb') as f:
        pickle.dump(correlationData, f, -1)
        
else:
    f = open(correlationSavePath, "rb")
    correlationData = pickle.load(f)
    f.close()
    correlations = correlationData['correlations']
    radiusRange = correlationData['radia']
    
meanCorrelations = np.mean(correlations,axis=(0,2)) # average & std over subjects and ROIs
stdCorrelations = np.std(correlations,axis=(0,2))
    
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(radiusRange,meanConsistencies,ls='',marker='.',color='k')
plt.errorbar(radiusRange,meanCorrelations,yerr=stdCorrelations,ls='',marker='.',color='k',ecolor='k')
plt.plot([radiusRange[0],radiusRange[-1]],[0,0],ls='--',color='k',alpha=0.7)
ax.set_xlabel('Neighborhood radius (in voxels)')
ax.set_ylabel('Mean neighborhood correlations')
plt.tight_layout()
plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')

    
    


