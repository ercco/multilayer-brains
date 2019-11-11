#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:03:53 2019

@author: onerva

A script for calculating and visualizing spatial consistency based on earlier
calculated ROIs.
"""
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
import pickle

import clustering_by_consistency as cbc

from scipy.stats import binned_statistic

# path parts
subjectFolder = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline'
subjects = ['010/1']
niiDataFileNames = ['/media/onerva/KINGSTON/test-data/010/epi_STD_mask_detrend_fullreg.nii'] # This should have the same length as the subjects list
runNumbers = ['1','2']
clusteringMethods = ['consistency_optimized','template_clustering']
templateNames = [['brainnetome','random'],['brainnetome']] # this is a methods x templates structure
# TODO: check if the looping over template names is needed (and otherwise fix the code to correspond to Tarmo's final folder structure)
netIdentificators = [[['net_100_0_2019-02-19T16.26.24/2_layers'],['net_100_0_2019-03-22T18.40.27/2_layers']],[['net_100_0_2019-02-19T13.40.36/2_layers']]] # this should be a clustering methods x templates x subjects list (they may have different network identificators); is there any way to do this automatically?
nLayers = 2
allFileNames = ['0_1']

readableFileNameIndices = range(0,len(allFileNames),nLayers) # same layer is saved in multiple files; therefore we read only every nLayer-th file
fileNames = []
for index in readableFileNameIndices:
    fileNames.append(allFileNames[index])

# time window parameters
timewindow = 100 # This is the time window length used to construct the ROIs
overlap = 0 # This is the overlap between consequent time windows

# parameters for calculating consistency
consistencyType='pearson c'
fTransform=False
nCPUs=5
templateFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
calculateConsistencies = False # set to False to read and visualize earlier calculated consistency

# parameters for reading consistencies calculated during clustering
calculatedDuringClustering = False
excludeSizes = True
consistencySaveNames = ['spatial_consistency_optimized_craddock','spatial_consistency_optimized_random_balls','spatial-consistency-optimized-test-for-Tarmo-mean-weighted-consistency-thresholded-voxelwise','spatial-consistency-optimized-test-for-Tarmo-weighted-mean-consistency']
timeWindows = ['0']

templateimg = nib.load(templateFile)
template = templateimg.get_fdata()

# visualization parameters
nConsBins = 50
consFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/consistency_distributions_all_test.pdf'

nSizeBins = 50
sizeFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/size_distributions_all_test.pdf'
    
#import pdb; pdb.set_trace()

pooledConsistencies = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames] # this is a clustering methods x template names x ROIs list (ROIs pooled over subjects and layers)
ROISizes = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames] 

if calculateConsistencies:
    for i, (clusteringMethod, templateNamesPerMethod, netIdentificatorsPerMethod) in enumerate(zip(clusteringMethods,templateNames,netIdentificators)):
        for j, (templateName, netIdentificatorsPerTemplate) in enumerate(zip(templateNamesPerMethod,netIdentificatorsPerMethod)):
            layersetwiseNetworkSavefolders = [subjectFolder + '/' + subject + '/' + clusteringMethod + '/' + templateName + '/' + netIdentificator for subject,netIdentificator in zip(subjects,netIdentificatorsPerTemplate)]
            savePath = subjectFolder + '/spatialConsistency_' + clusteringMethod + '_' + templateName + '.pkl'
            spatialConsistencyData = cbc.calculateSpatialConsistencyPostHoc(niiDataFileNames,templateFile,layersetwiseNetworkSavefolders,
                                                                            allFileNames,nLayers,timewindow,overlap,consistencyType,
                                                                            fTransform,nCPUs,savePath)
            ROISizes[i][j].extend(spatialConsistencyData['roi_sizes'])
            pooledConsistencies[i][j].extend(spatialConsistencyData['spatial_consistencies'])
        
elif calculatedDuringClustering:
    for subject in subjects:
        consistencySaveNames = [subjectFolder + subject + '/' + consistencySaveName for consistencySaveName in consistencySaveNames]
        for i, (clusteringMethod, tempalateNamesPerMethod, consistencySaveName) in enumerate(zip(clusteringMethods,templateNames,consistencySaveNames)):
            for j, templateName in enumerate(templateNamesPerMethod):
                for timeWindow in timeWindows:
                    consistencyPath = consistencySaveName + '_' + timeWindow + '.pkl'
                    f = open(consistencyPath, 'r')
                    data = pickle.load(f)
                    f.close()
                    if 'consistencies' in data.keys():
                        pooledConsistencies[i][j].extend(data['consistencies'])
                    else:
                        pooledConsistencies[i][j].extend(data['spatialConsistencies'])                        
else:
    for i, (clusteringMethod, templateNamesPerMethod, netIdentificatorsPerMethod) in enumerate(zip(clusteringMethods,templateNames,netIdentificators)):
        for j, (templateName, netIdentificatorsPerTemplate) in enumerate(zip(templateNamesPerMethod,netIdentificatorsPerMethod)):
            savePath = subjectFolder + '/spatialConsistency_' + clusteringMethod + '_' + templateName + '.pkl'
            f = open(savePath,'r')
            spatialConsistencyData = pickle.load(f)
            f.close()
            ROISizes[i][j].extend(spatialConsistencyData['roi_sizes'])
            pooledConsistencies[i][j].extend(spatialConsistencyData['spatial_consistencies'])

# calculating and visualizing consistency distributions
consFig = plt.figure(1)
consAx = consFig.add_subplot(111)

if not excludeSizes:
    sizeFig = plt.figure(2)
    sizeAx = sizeFig.add_subplot(111)

for pooledConsistencyPerMethod, sizesPerMethod, clusteringMethod, templateNamesPerMethod in zip(pooledConsistencies, ROISizes, clusteringMethods, templateNames):
    for pooledConsistency, sizes, templateName in zip(pooledConsistencyPerMethod, sizesPerMethod, templateNamesPerMethod):
        consDistribution,consBinEdges,_ = binned_statistic(pooledConsistency,pooledConsistency,statistic='count',bins=nConsBins)
        consBinWidth = (consBinEdges[1] - consBinEdges[0])
        consBinCenters = consBinEdges[1:] - consBinWidth/2
        consDistribution = consDistribution/float(np.sum(consDistribution))
        consAx.plot(consBinCenters,consDistribution,label=clusteringMethod + ', ' + templateName)
        
        if not excludeSizes:
            sizeDistribution,sizeBinEdges,_ = binned_statistic(ROISizes,sizes,statistic='count',bins=nSizeBins)
            sizeBinWidth = (sizeBinEdges[1] - sizeBinEdges[0])
            sizeBinCenters = sizeBinEdges[1:] - sizeBinWidth/2
            sizeDistribution = sizeDistribution/float(np.sum(sizeDistribution))
            sizeAx.plot(sizeBinCenters,sizeDistribution,label=clusteringMethod + ', ' + templateName)

plt.figure(1)
        
consAx.set_xlabel('Consistency')
consAx.set_ylabel('PDF')
consAx.legend()

plt.tight_layout()
plt.savefig(consFigureSavePath,format='pdf',bbox_inches='tight')

if not excludeSizes:

    plt.figure(2)
    
    sizeAx.set_xlabel('ROI size (number of voxels)')
    sizeAx.set_ylabel('PDF')
    sizeAx.legend()
    
    plt.tight_layout()
    plt.savefig(sizeFigureSavePath,format='pdf',bbox_inches='tight')

