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
import network_construction
import corrs_and_mask_calculations

from scipy.stats import binned_statistic

# path parts
subjectFolder = '/media/onerva/KINGSTON/test-data/'
subjects = ['010']
niiDataFileNames = ['/media/onerva/KINGSTON/test-data/010/epi_STD_mask_detrend_fullreg.nii'] # This should have the same length as the subjects list
runNumber = '1'
clusteringMethods = ['craddock','random_balls','consistency_optimized_thresholded','consistency_optimized_non-thresholded','template']
templateNames = [['craddock'],['random'],['brainnetome'],['brainnetome']]  # this is a methods x templates structure
# TODO: check if the looping over template names is needed (and otherwise fix the code to correspond to Tarmo's final folder structure)
netIdentificators = [[['net_100_0_2019-02-19T16.26.24','net_100_0_2019-03-22T18.40.27'],['net_100_0_2019-02-19T13.40.36']]] # this should be a subjects x clustering methods x templates list (they may have different network identificators); is there any way to do this automatically?
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
template = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
consistencySavePath = subjectFolder + '/' + 'spatialConsistency_' + runNumber + '.pkl'
calculateConsistencies = False # set to False to read and visualize earlier calculated consistency

# parameters for reading consistencies calculated during clustering
calculatedDuringClustering = True
excludeSizes = True
consistencySaveNames = ['spatial_consistency_optimized_craddock','spatial_consistency_optimized_random_balls','spatial-consistency-optimized-test-for-Tarmo-mean-weighted-consistency-thresholded-voxelwise','spatial-consistency-optimized-test-for-Tarmo-weighted-mean-consistency']
timeWindows = ['0']

templateimg = nib.load(template)
template = templateimg.get_fdata()

# visualization parameters
nConsBins = 50
consFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/consistency_distributions_all.pdf'

nSizeBins = 50
sizeFigureSavePath = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/size_distributions_all.pdf'
    
#import pdb; pdb.set_trace()
startTimes,endTimes = network_construction.get_start_and_end_times(nLayers,timewindow,overlap) # end and start points of time windows

pooledConsistencies = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames] # this is a clustering methods x template names x ROIs list (ROIs pooled over subjects and layers)
ROISizes = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames] 

if calculateConsistencies:
    # calculating consistencies
    for subject, netIdentificatorsPerSubject, niiDataFileName in zip(subjects, netIdentificators, niiDataFileNames):
        img = nib.load(niiDataFileName) # reading data; later on, this will be used to calculate consistencies
        imgdata = img.get_fdata()
        corrs_and_mask_calculations.gray_mask(imgdata,template)
        nTime = imgdata.shape[-1]
        
        x,y,z = np.where(imgdata[:,:,:,0] != 0) # finding voxel coordinates (these may vary between subjects: it's possible that subject's EPI doesn't include all voxels of the template)
        nVoxels = len(x)
        voxelCoordinates = np.concatenate((x,y,z)).reshape(3,nVoxels).T
        
        allVoxelTs = np.zeros((nVoxels,nTime)) # reading voxel time series
        for i, voxel in enumerate(voxelCoordinates):
            allVoxelTs[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
        
        for i, (clusteringMethod, templateNamesPerMethod, netIdentificatorsPerMethod) in enumerate(zip(clusteringMethods, templateNames, netIdentificatorsPerSubject)):
            for j, (templateName, netIdentificator) in enumerate(zip(templateNamesPerMethod, netIdentificatorsPerMethod)):
                for fileName in fileNames:
                    netPath = subjectFolder + '/' + subject + '/' + runNumber + '/' + clusteringMethod + '/' + templateName + '/' + netIdentificator + '/' + str(nLayers) + '_layers' + '/' + fileName
                    voxelIndices = cbc.readVoxelIndices(netPath, voxelCoordinates, layers='all')
                    for voxelIndicesPerLayer, startTime, endTime in zip(voxelIndices, startTimes, endTimes):
                        ROISizes[i][j].extend([len(voxelInds) for voxelInds in voxelIndicesPerLayer])
                        pooledConsistencies[i][j].extend(cbc.calculateSpatialConsistencyInParallel(voxelIndicesPerLayer,allVoxelTs[:,startTime:endTime],consistencyType,fTransform,nCPUs))
    
    # saving consistencies for further use
    metaData = {'subjectFolder':subjectFolder,'subjects':subjects,'niiDataFileNames':niiDataFileNames,'runNumber':runNumber,'clusteringMethods':clusteringMethods,'templateNames':templateNames,'netIdentificators':netIdentificators,'nLayers':nLayers,'fileNames':fileNames,'timewindow':timewindow,'overlap':overlap,'consistencyType':consistencyType,'fTransform':fTransform}
    data = {'metaData':metaData,'consistencies':pooledConsistencies,'ROI sizes':ROISizes}
    with open(consistencySavePath, 'wb') as f:
        pickle.dump(data, f, -1)
        
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
    # TODO: remove the following lines, they are a hack!!!
#    templateNames.append(['brainnetome'])
#    f = open('/media/onerva/KINGSTON/test-data/spatial-consistency-original.pkl','r')
#    data = pickle.load(f)
#    f.close()
#    pooledConsistencies.append([data['spatialConsistencies']])
#    ROISizes.append([[]])
                         
else:
    f = open(consistencySavePath,'r')
    data = pickle.load(f)
    f.close()
    pooledConsistencies = data['consistencies']

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

