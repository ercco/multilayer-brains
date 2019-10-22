#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:05:14 2019

@author: onerva

This is a test script for the optimizeParcellationByFlipping function.
"""
import nibabel as nib
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import binned_statistic

import clustering_by_consistency as cbc
import network_construction as nc

templateFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
template = nib.load(templateFile)
templateimg = template.get_fdata()

imgfile = '/media/onerva/KINGSTON/test-data/010/epi_STD_mask_detrend_fullreg.nii'
img = nib.load(imgfile)
imgdata = img.get_fdata()

targetFunctions = ['weighted mean consistency']#,'mean consistency']

nConsBins = 50
nSizeBins = 50

origColor = 'k'
colors = ['b']#,'r']
origLineStyle = '-'
lineStyles = ['--']#,'-.']
origLabel = 'Brainnetome'

targetFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/flipping/flipping-sophisticated-target-function.pdf'
consistencyFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/flipping/flipping-sophisticated-consistency-dist.pdf'
sizeFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/flipping/flipping-sophisticated-size-dist.pdf'
weightedConsistencyDistSavePath = '/media/onerva/KINGSTON/test-data/outcome/flipping/flipping-sophisticated-weighted-consistency-dist.pdf'
weightedConsistencySavePath = '/media/onerva/KINGSTON/test-data/outcome/flipping/flipping-sophisticated-mean-weighted-consistency.pdf'

visualize = True

voxelsInClusters = nc.get_voxels_in_clusters(templateimg)
origROIInfo, origVoxelCoordinates, origVoxelLabels = cbc.voxelsInClustersToROIInfo(voxelsInClusters)

origROILabels = voxelsInClusters.keys()
if -1 in origROILabels:
    origROILabels.remove(-1)
    
nVoxels = origVoxelCoordinates.shape[0]
nROIs = len(origROILabels)
nTime = imgdata.shape[-1]
allVoxelTs = np.zeros((nVoxels,nTime))
for i,voxel in enumerate(origVoxelCoordinates):
    allVoxelTs[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
 
if visualize:    
    origSizes = [len(voxelsInClusters[label]) for label in origROILabels]
    origVoxelIndices = origROIInfo['ROIVoxels']
    origConsistencies = cbc.calculateSpatialConsistencyInParallel(origVoxelIndices,allVoxelTs)
    origWeightedConsistencies = [origConsistency*origSize for origConsistency,origSize in zip(origConsistencies,origSizes)]
    origConsistencyDist, origConsBinEdges,_ = binned_statistic(origConsistencies,origConsistencies,statistic='count',bins=nConsBins)
    origConsBinWidth = origConsBinEdges[1] - origConsBinEdges[0]
    origConsBinCenters = origConsBinEdges[1:] - origConsBinWidth/2
    origConsistencyDist = origConsistencyDist/float(np.sum(origConsistencyDist))
    origWeightedConsistencyDist, origWeightedConsistencyBinEdges, _ = binned_statistic(origWeightedConsistencies,origWeightedConsistencies,statistic='count',bins=nConsBins)
    origWeightedConsBinWidth = origWeightedConsistencyBinEdges[1] - origWeightedConsistencyBinEdges[0]
    origWeightedConsBinCenters = origWeightedConsistencyBinEdges[1:] - origWeightedConsBinWidth/2
    origWeightedConsistencyDist = origWeightedConsistencyDist/float(np.sum(origWeightedConsistencyDist))
    origSizeDist, origSizeBinEdges,_ = binned_statistic(origSizes,origSizes,statistic='count',bins=nSizeBins)
    origSizeBinWidth = (origSizeBinEdges[1] - origSizeBinEdges[0])
    origSizeBinCenters = origSizeBinEdges[1:] - origSizeBinWidth/2
    origSizeDistribution = origSizeDist/float(np.sum(origSizeDist))
    
    targetFig = plt.figure(1)
    targetAx = targetFig.add_subplot(111)
    
    consistencyFig = plt.figure(2)
    consistencyAx = consistencyFig.add_subplot(111)
    consistencyAx.plot(origConsBinCenters,origConsistencyDist,ls=origLineStyle,color=origColor,label=origLabel)
    sizeFig = plt.figure(3)
    sizeAx = sizeFig.add_subplot(111)
    sizeAx.plot(origSizeBinCenters,origSizeDist,ls=origLineStyle,color=origColor,label=origLabel)
    weightedConsistencyFig = plt.figure(4)
    weightedConsistencyAx = weightedConsistencyFig.add_subplot(111)
    weightedConsistencyAx.plot(origWeightedConsBinCenters,origWeightedConsistencyDist,ls=origLineStyle,color=origColor,label=origLabel)

for targetFunction, color, ls in zip(targetFunctions,colors,lineStyles):
    cfg = {'inputType':'voxelsInClusters',
           'voxelsInClusters':voxelsInClusters,
           'imgdata':imgdata,
           'targetFunction':targetFunction,
           'fTransform': False}
    
    voxelLabels,voxelCoordinates,targets, meanWeightedConsistencies = cbc.optimizeParcellationByFlipping(cfg)
    ROILabels = list(np.unique(voxelLabels))
    if -1 in ROILabels:
        ROILabels.remove(-1)
    voxelIndices = []
    for ROILabel in ROILabels:
        voxelIndices.append(np.where(np.array(voxelLabels)==ROILabel)[0])
    
    if visualize:
        targetAx.plot(targets,label=targetFunction)
        nFlips = len(np.unique(targets)) - 1 # excluding the initial target value, each unique target value indicates a flip
        print(targetFunction + ': ' + str(nFlips) + ' voxels flipped to neighboring ROI')
        
        sizes = [len(indices) for indices in voxelIndices]
        consistencies = cbc.calculateSpatialConsistencyInParallel(voxelIndices,allVoxelTs)
        weightedConsistencies = [consistency*size for consistency,size in zip(consistencies,sizes)]
        consistencyDist, consBinEdges,_ = binned_statistic(consistencies,consistencies,statistic='count',bins=nConsBins)
        consBinWidth = (consBinEdges[1] - consBinEdges[0])
        consBinCenters = consBinEdges[1:] - consBinWidth/2
        consistencyDist = consistencyDist/float(np.sum(consistencyDist))
        weightedConsistencyDist, weightedConsistencyBinEdges, _ = binned_statistic(weightedConsistencies,weightedConsistencies,statistic='count',bins=nConsBins)
        weightedConsBinWidth = weightedConsistencyBinEdges[1] - weightedConsistencyBinEdges[0]
        weightedConsBinCenters = weightedConsistencyBinEdges[1:] - weightedConsBinWidth/2
        weightedConsistencyDist = weightedConsistencyDist/float(np.sum(weightedConsistencyDist))
        sizeDist, sizeBinEdges,_ = binned_statistic(sizes,sizes,statistic='count',bins=nSizeBins)
        sizeBinWidth = (sizeBinEdges[1] - sizeBinEdges[0])
        sizeBinCenters = sizeBinEdges[1:] - sizeBinWidth/2
        sizeDistribution = sizeDist/float(np.sum(sizeDist))
        
        consistencyAx.plot(consBinCenters,consistencyDist,ls=ls,color=color,label=targetFunction)
        sizeAx.plot(sizeBinCenters,sizeDist,ls=ls,color=color,label=targetFunction)
        weightedConsistencyAx.plot(weightedConsBinCenters,weightedConsistencyDist,ls=ls,color=color,label=targetFunction)
        
        wFig = plt.figure(5)
        wAx = wFig.add_subplot(111)
        wAx.plot(meanWeightedConsistencies)

#import pdb; pdb.set_trace()

if visualize:    
    plt.figure(1)    
    targetAx.set_xlabel('iteration step')
    targetAx.set_ylabel('target function value')
    targetAx.legend()
    plt.tight_layout()
    plt.savefig(targetFigureSavePath,format='pdf',bbox_inches='tight')
    
    plt.figure(2)
    consistencyAx.set_xlabel('consistency')
    consistencyAx.set_ylabel('pdf')
    consistencyAx.legend()
    plt.tight_layout()
    plt.savefig(consistencyFigureSavePath,format='pdf',bbox_inches='tight')
    
    plt.figure(3)
    sizeAx.set_xlabel('ROI size')
    sizeAx.set_ylabel('pdf')
    sizeAx.legend()
    plt.tight_layout()
    plt.savefig(sizeFigureSavePath,format='pdf',bbox_inches='tight')
    
    plt.figure(4)
    weightedConsistencyAx.set_xlabel('weighted consistency (consistency*size)')
    weightedConsistencyAx.set_ylabel('pdf')
    weightedConsistencyAx.legend()
    plt.tight_layout()
    plt.savefig(weightedConsistencyDistSavePath,format='pdf',bbox_inches='tight')
    
    plt.figure(5)
    wAx.set_xlabel('iteration step')
    wAx.set_ylabel('mean weighted consistency')
    plt.tight_layout()
    plt.savefig(weightedConsistencySavePath,format='pdf',bbox_inches='tight')

