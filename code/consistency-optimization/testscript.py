#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:56:24 2019

@author: onerva
"""
import cProfile

import clustering_by_consistency as cbc
import nibabel as nib
import numpy as np

template = nib.load('/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii').get_fdata()
ROICentroids = cbc.getRandomCentroids(100,template)

ROIMaps = [np.array(centroid) for centroid in ROICentroids]

imgdata = nib.load('/media/onerva/KINGSTON/test-data/010/epi_STD_mask_detrend_fullreg.nii').get_fdata()
voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))

#pr = cProfile.Profile()
#pr.enable()
#    
#pr.disable()
#pr.print_stats(sort='time')
    



