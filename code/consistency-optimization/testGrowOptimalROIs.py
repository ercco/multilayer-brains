# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:17:59 2018

@author: onerva

A script for testing local/clustering-by-consistency
"""
import numpy as np
from scipy import io
import cPickle as pickle
import matplotlib.pylab as plt

import os.path
import sys
#if os.path.exists('/home/onerva/dippa/local'):
#    sys.path.insert(0,'/home/onerva/dippa/local')
#else:
#    sys.path.insert(0,'/home/onerva/projects/dippa/local')   

#import functions
#import onion_parameters as params
import nibabel as nib

import clustering_by_consistency as cbc

testSubjectFolders = ['/media/onerva/KINGSTON/test-data/010/',
                      '/media/onerva/KINGSTON/test-data/011/']
ROIVoxelTsFileName = 'roi_voxel_ts_all_rois4mm_FWHM0.mat'
ROIVoxelTsInfoFileName = 'roi_voxel_ts_all_rois4mm_WHM0_info.mat'
allVoxelTsPath = testSubjectFolders[0] + ROIVoxelTsFileName
ROIInfoFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
imgdataPath = testSubjectFolders[0] + 'epi_STD_mask_detrend_fullreg.nii'

niiSavePath = testSubjectFolders[0] + '/optimized-rois-test-for-Tarmo-mean-weighted-consistency-thresholded-voxel-neighborhood.nii'
pickleSavePath = testSubjectFolders[0] + '/optimized-rois-test-for-Tarmo-mean-weighted-consistency-thresholded-voxel-neighborhood.pkl'

ROICentroids,_,voxelCoordinates,_ = cbc.readROICentroids(ROIInfoFile,readVoxels=True,fixCentroids=True)
allVoxelTs = io.loadmat(allVoxelTsPath)['roi_voxel_data'][0]['roi_voxel_ts'][0]
imgdata = nib.load(imgdataPath).get_data()

groupMaskPath = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
groupMask = nib.load(groupMaskPath).get_data()
nTime = imgdata.shape[3]
for i in range(nTime):
    imgdata[:,:,:,i] = imgdata[:,:,:,i] * groupMask

cfg = {}
cfg['ROICentroids'] = ROICentroids
cfg['voxelCoordinates'] = voxelCoordinates
cfg['names'] = ''
cfg['allVoxelTs'] = allVoxelTs
cfg['imgdata'] = imgdata

cfg['threshold'] = 'voxel-neighborhood'
cfg['targetFunction'] = 'weighted mean consistency'

ROILabels, voxelCoordinates,_ = cbc.growOptimizedROIs(cfg)

ROIInfo = {'ROILabels':ROILabels,'voxelCoordinates':voxelCoordinates}

with open(pickleSavePath, 'wb') as f:
        pickle.dump(ROIInfo, f, -1)

# TODO: check if the following is needed in future
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(selectedMeasures)
#ax.set_xlabel('Iteration step')
#ax.set_ylabel('Maximal similarity index')
#plt.tight_layout()
#plt.savefig('/media/onerva/KINGSTON/test-data/outcome/maximal-measure-spatialConsistency-for-Tarmo.pdf',format='pdf',bbox_inches='tight')
#
#
#cbc.createNii(ROIInfo, niiSavePath, imgSize=[45,54,45], affine=np.eye(4))
#with open(pickleSavePath, 'wb') as f:
#        pickle.dump(ROIInfo, f, -1)

