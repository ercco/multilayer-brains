#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:36:19 2019

@author: onerva

This is as script for comparing the original, non-optimized Brainnetome parcellation
with a parcellation grown around random centroids. To compare the parcellations,
variation of information (VI) is used
"""
import numpy as np
import nibabel as nib
from scipy import io
import matplotlib.pylab as plt

import clustering_by_consistency as cbc

def calculateIntersection(a,b):
    """
    Calculates the size of the intersection of two lists a and b. Code originally from
    https://www.geeksforgeeks.org/python-intersection-two-lists/
    
    Parameters:
    -----------
    a, b: lists
    
    Returns:
    --------
    Ni: int, size of the intersection of a, b (i.e. the number of elements present in 
                  both a and b)
    """
    temp = set(a) 
    Ni = len([value for value in b if value in temp]) 
    return Ni

def calculateVI(A, B):
    """
    Calculates the variation of information (VI) between two parcellations A and
    B as
    VI = H(A) + H(B) - MI(A,B)
    where H = entropy, MI = mutual information.
    
    Parameters:
    -----------
    A, B: lists of lists, the two parcellations to compare. Each of the parcellations
          should be presented as a list of lists of elements belonging to each parcel
          (ROI, module, cluster).
          
    Returns:
    --------
    VI: float, the variation of information between A and B
    """
    #import pdb; pdb.set_trace()
    assert sum([len(ROI) for ROI in A]) == sum([len(ROI) for ROI in B]), 'Parcellations A and B contain different numbers of nodes!!!'
    N = float(sum([len(ROI) for ROI in A]))
    HA = -1 * sum([len(ROI)/N*np.log2(len(ROI)/N) for ROI in A])
    HB = -1 * sum([len(ROI)/N*np.log2(len(ROI)/N) for ROI in B])
    MI = 0
    for ROIA in A:
        for ROIB in B:
            Ni = calculateIntersection(ROIA,ROIB)
            if not Ni == 0:
                MI = MI + Ni/N*np.log2(Ni*N/(len(ROIA)*len(ROIB)))
    VI = HA + HB - 2*MI
    return VI
            
testSubjectFolder = '/media/onerva/KINGSTON/test-data/010/'
ROIVoxelTsFileName = 'roi_voxel_ts_all_rois4mm_FWHM0.mat'
ROIVoxelTsInfoFileName = 'roi_voxel_ts_all_rois4mm_WHM0_info.mat'
allVoxelTsPath = testSubjectFolder + ROIVoxelTsFileName
ROIInfoFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
imgdataPath = testSubjectFolder + 'epi_STD_mask_detrend_fullreg.nii'

figureSavePath = '/media/onerva/KINGSTON/test-data/outcome/random-vs-brainnetome.pdf'

equalSized = False
nIterations = 10
nROIs = 246

allVoxelTs = io.loadmat(allVoxelTsPath)['roi_voxel_data'][0]['roi_voxel_ts'][0]
imgdata = nib.load(imgdataPath).get_data()

groupMaskPath = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
groupMask = nib.load(groupMaskPath).get_data()
nTime = imgdata.shape[3]
for i in range(nTime):
    imgdata[:,:,:,i] = imgdata[:,:,:,i] * groupMask
    
_,_,originalVoxelCoordinates,originalROIMaps = cbc.readROICentroids(ROIInfoFile,readVoxels=True,fixCentroids=True)
originalROIs = []
for ROIMap in originalROIMaps:
    ROI = []
    for voxel in ROIMap:
        voxelIndex = np.where((originalVoxelCoordinates == voxel).all(axis=1)==1)[0][0]
        if len(cbc.findNeighbors(voxel,allVoxels=originalVoxelCoordinates)) > 0:
            ROI.append(voxelIndex)
    originalROIs.append(ROI)
    
VIs = np.zeros(nIterations)
    
for i in range(nIterations):
    ROILabels, voxelCoordinates, radius = cbc.growSphericalROIs('random', imgdata, nROIs=nROIs, template=groupMask, equalSized=equalSized)
    #import pdb; pdb.set_trace()
    randomROIs = [list(np.where(ROILabels == j)[0]) for j in range(nROIs)]
    VIs[i] = calculateVI(originalROIs,randomROIs)
    
maxVI = np.log2(sum([len(randomROI) for randomROI in randomROIs]))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(VIs,color='b',label='data')
plt.plot([1,nIterations],[maxVI,maxVI],color='r',label='max VI (log_2(N))')

ax.set_xlabel('Iteration')
ax.set_ylabel('VI(Brainnetome,random)')
ax.legend()

plt.tight_layout()
plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')


    
    

