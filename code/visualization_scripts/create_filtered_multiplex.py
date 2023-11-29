# A script for defining the nodes for a "filtered multiplex network". Nodes of the network are ROIs from a static parcellation (e.g. Brainnetome), but not all nodes are present on all layers. In each time window (on each layer), ROIs are added in the order of decreasing spatial consistency till the included ROIs cover the same number of voxels as the ROIs of an optimized reference parcellation. 

import numpy as np
import pickle
import nibabel as nib

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]
nLayers = 2

# path parts for reading data
consistencySaveStem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
filteredJobLabels = ['template_brainnetome'] # The job labels specify the job submitted to Triton; there may be several jobs saved under each label
referenceJobLabel = 'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100'
# NOTE: before running the script, check that consistencySaveStem, jobLabel, clusteringMethods, and savePath (specified further below) match your data
blacklistedROIs = []#np.arange(211,247)
blacklistWholeROIs = True # if True, all ROIs with blacklisted voxels are removed; if False, blacklisted voxels are removed but rest of the ROI kept, which affects size distribution
windowLength = 80
windowOverlap = 0
if len(blacklistedROIs) > 0:
    iniDataFolder = '/scratch/nbe/alex/private/janne/preprocessed_ini_data/'

# path parths for saving
pooledDataSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/pooled_spatial_consistency_data_filtered_multiplex.pkl'

if __name__=='__main__':
    pooledConsistencies = [[] for i in range(len(filteredJobLabels)*2 + 1)] # the list will contain the original multiplex, filtered multiplex and reference
    pooledROISizes = [[] for i in range(len(filteredJobLabels)*2 + 1)]
    nRefROIs = []
    nRefSingles = []
    meanNROIs = []
    stdNROIs = []
    meanNSingles = []
    stdNSingles = []
    meanSizes = []
    stdSizes = []
    totRefSizes = [[] for i in range(len(filteredJobLabels))]

    for i, filteredJobLabel in zip(np.arange(0, len(filteredJobLabels)*2, 2), filteredJobLabels): 
        nROIs = []
        nSingles = []
        origNROIs = []
        origNSingles = []
        for subjId in subjectIds:
            for runNumber in runNumbers:
                referencePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + referenceJobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
                ref = open(referencePath, 'r')
                refConsistencyData = pickle.load(ref)
                ref.close()
 
                filteredPath = savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + filteredJobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
                f = open(savePath,'r')
                filteredConsistencyData = pickle.load(f)
                f.close()

                for windowIndex in refConsistencyData:
                    refLayer = refConsistencyData[windowIndex]
                    layer = filteredConsistencyData[windowIndex]
                    
                    if len(blacklistedROIs) > 0:
                        refConsistencies = []
                        consistencies = []
                        refROISizes = []
                        ROISizes = []

                        dataMaskFileName = iniDataFolder+'ROI_parcellations/'+subjId+'/run'+str(runNumber)+'/'+subjId+'_final_BN_Atlas_246_1mm.nii'
                        maskImg = nib.load(dataMaskFileName)
                        maskData = maskImg.get_fdata()
                        dataFileName = iniDataFolder+subjId+'/run'+str(runNumber)+'/detrended_maxCorr5comp.nii'
                        img = nib.load(dataFileName)
                        data = img.get_fdata()
                        nWindows = nc.get_number_of_layers(data.shape, windowLength, windowOverlap)
                        startTimes, endTimes = nc.get_start_and_end_times(nWindows, windowLength, windowOverlap)
                        for ROI in refLayer['ROI_sizes']:
                            ROIVoxelMaskIndices = np.rint([maskData[voxel] for voxel in ROI])
                            blacklistedIndices = np.array([ROIVoxelMaskIndex in blacklistedROIs for ROIVoxelMaskIndex in ROIVoxelMaskIndices])
                            if np.all(blacklistedIndices):
                                continue
                            elif np.any(blacklistedIndices):
                                if blacklistWholeROIs:
                                    continue
                                else:
                                    refROISizes.append(refLayer['ROI_sizes'][ROI] - np.sum(blacklistedIndices))
                                    ROI = [ROI[index] for index in np.where(1 - blacklistedIndices)[0]]
                                    allVoxelTs = np.array([data[voxel[0], voxel[1], voxel[2], startTimes[windowIndex]:endTimes[windowIndex]] for voxel in ROI])
                                    voxelIndices = np.arange(0,len(ROI))
                                    consistencyType = layer['consistency_type']
                                    if 'pearson c' in consistencyType: # a hack to fix mistyped consistency type
                                        consistencyType = 'pearson c'
                                    refConsistencies.append(cbc.calculateSpatialConsistency(({'allVoxelTs':allVoxelTs, 'consistencyType':consistencyType, 'ftransform':layer['ftransform']},voxelIndices)))
                            else:
                                refROISizes.append(layer['ROI_sizes'][ROI])
                                refConsistencies.append(layer['consistencies'][ROI])

                        for ROI in layer['ROI_sizes']:
                            ROIVoxelMaskIndices = np.rint([maskData[voxel] for voxel in ROI])
                            blacklistedIndices = np.array([ROIVoxelMaskIndex in blacklistedROIs for ROIVoxelMaskIndex in ROIVoxelMaskIndices])
                            if np.all(blacklistedIndices):
                                continue
                            elif np.any(blacklistedIndices):
                                if blacklistWholeROIs:
                                    continue
                                else:
                                    ROISizes.append(layer['ROI_sizes'][ROI] - np.sum(blacklistedIndices))
                                    ROI = [ROI[index] for index in np.where(1 - blacklistedIndices)[0]]
                                    allVoxelTs = np.array([data[voxel[0], voxel[1], voxel[2], startTimes[windowIndex]:endTimes[windowIndex]] for voxel in ROI])
                                    voxelIndices = np.arange(0,len(ROI))
                                    consistencyType = layer['consistency_type']
                                    if 'pearson c' in consistencyType: # a hack to fix mistyped consistency type
                                        consistencyType = 'pearson c'
                                    consistencies.append(cbc.calculateSpatialConsistency(({'allVoxelTs':allVoxelTs, 'consistencyType':consistencyType, 'ftransform':layer['ftransform']},voxelIndices)))
                            else:
                                ROISizes.append(layer['ROI_sizes'][ROI])
                                consistencies.append(layer['consistencies'][ROI])
                    else:
                        refConsistencies = refLayer['consistencies'].values()
                        refROISizes = refLayer['ROI_sizes'].values()
                        consistencies = layer['consistencies'].values()
                        consistencyIndices = np.argsort(consistencies)
                        consistencies = np.array(consistencies)[consistencyIndices[::-1]]
                        ROISizes = np.array(layer['ROI_sizes'].values())[consistencyIndices[::-1]]

                    refNVoxels = np.sum(refROISizes)
                    totRefSizes[i].append(refNVoxels)
                    nVoxels = 0
                    index = 0
                    pooledConsistencies[i+1].extend(consistencies)
                    pooledROISizes[i+1].extend(ROISizes)
                    origNROIs.append(len(ROISizes))
                    origNSingles.append(np.sum(np.array(ROISizes)==1))
                    while nVoxels <= refNVoxels:
                        pooledConsistencies[i].append(consistencies[index])
                        pooledROISizes[i].append(ROISizes[index])
                        nVoxels += ROISizes[index]
                        index += 1
                    nROIs.append(index-1)
                    nSingles.append(np.sum(np.array(ROISizes[0:index]) == 1))
                    if i == len(filteredJobLabels)*2 - 2:
                        pooledConsistencies[-1].extend(refConsistencies) # the last element of the lists contains the consistency and ROI size of the reference parcellation
                        pooledROISizes[-1].extend(refROISizes)
                        nRefROIs.append(len(refROISizes))
                        nRefSingles.append(np.sum(np.array(refROISizes) == 1))
        meanNROIs.append(np.mean(nROIs))
        stdNROIs.append(np.std(nROIs))
        meanNSingles.append(np.mean(nSingles))
        stdNSingles.append(np.std(nSingles))
        sizes = np.array(pooledROISizes[i])
        meanSizes.append(sizes[sizes>1].mean())
        stdSizes.append(sizes[sizes>1].std())
        meanNROIs.append(np.mean(origNROIs))
        stdNROIs.append(np.std(origNROIs))
        meanNSingles.append(np.mean(origNSingles))
        stdNSingles.append(np.std(origNSingles))
        sizes = np.array(pooledROISizes[i+1])
        meanSizes.append(sizes[sizes>1].mean())
        stdSizes.append(sizes[sizes>1].std())        
        if i == len(filteredJobLabels)*2 - 2:
           meanNROIs.append(np.mean(nRefROIs))
           stdNROIs.append(np.std(nRefROIs))
           meanNSingles.append(np.mean(nRefSingles))
           stdNSingles.append(np.std(nRefSingles))
           sizes = np.array(pooledROISizes[-1])
           meanSizes.append(sizes[sizes>1].mean())
           stdSizes.append(sizes[sizes>1].std())

    f = open(pooledDataSavePath, 'wb')
    pooledData = {'pooledConsistencies':pooledConsistencies,'ROISizes':pooledROISizes,'meanNROIs':meanNROIs,'stdNROIs':stdNROIs,'meanNSingles':meanNSingles,'stdNSingles':stdNSingles,'meanSizes':meanSizes,'stdSizes':stdSizes,'nRefVoxels':totRefSizes}
    pickle.dump(pooledData, f)
    f.close()

    


