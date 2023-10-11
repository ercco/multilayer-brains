"""
A script for calculating distributions of earlier-calculated spatial consistency and for visualizing them.
"""
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
import pickle
import os
import sys
from databinner import binner

import clustering_by_consistency as cbc
import network_construction as nc

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]
nLayers = 2

# path parts for reading data
consistencySaveStem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
jobLabels = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that consistencySaveStem, jobLabel, clusteringMethods, and savePath (specified further below) match your data
blacklistedROIs = range(211,247)
blacklistWholeROIs = True # if True, all ROIs with blacklisted voxels are removed; if False, blacklisted voxels are removed but rest of the ROI kept, which affects size distribution
windowLength = 80
windowOverlap = 0
if len(blacklistedROIs) > 0:
    iniDataFolder = '/scratch/nbe/alex/private/janne/preprocessed_ini_data/'

# path parths for saving
pooledDataSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/pooled_spatial_consistency_data_for_fig.pkl'
consFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/consistency_distributions.pdf'
sizeFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/size_distributions.pdf'
consSizeFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/size_vs_consistency.pdf'

# distribution and visualization parameters
nConsBins = 50
sizeBinFactor = 1.2
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
excludeSizes = False
excludeSingleVoxels = True

import pdb; pdb.set_trace()
if os.path.isfile(pooledDataSavePath):
        f = open(pooledDataSavePath, 'rb')
        pooledData = pickle.load(f)
        f.close()
        pooledConsistencies = pooledData['pooledConsistencies']
        ROISizes = pooledData['ROISizes']
        meanNROIs = pooledData['meanNROIs']
        stdNROIs = pooledData['stdNROIs']
        meanNSingles = pooledData['meanNSingles']
        stdNSingles = pooledData['stdNSingles']
        meanSizes = pooledData['meanSizes']
        stdSizes = pooledData['stdSizes']
        if len(blacklistedROIs) > 0:
            nBlacklisted = pooledData['nBlacklisted']
            nModified = pooledData['nModified']
else:
    # reading data
    pooledConsistencies = [[] for clusteringMethod in clusteringMethods] # this is a clustering methods x ROIs list (ROIs pooled over subjects, layers, and runs)
    ROISizes = [[] for clusteringMethod in clusteringMethods]

    meanNROIs = []
    stdNROIs = []
    meanNSingles = []
    stdNSingles = []
    meanSizes = []
    stdSizes = []
    if len(blacklistedROIs) > 0:
        nBlacklisted = []
        nModified = []

    for i, (jobLabel,clusteringMethod) in enumerate(zip(jobLabels,clusteringMethods)):
        nROIs = []
        nSingles = []
        blacklistCounter = 0
        modifiedCounter = 0
        for subjId in subjectIds:
            for runNumber in runNumbers:
                if clusteringMethod == '':
                    savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
                else:
                    savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency_' + clusteringMethod + '.pkl'
                f = open(savePath,'r')
                spatialConsistencyData = pickle.load(f)
                f.close()
                for windowIndex in spatialConsistencyData:
                    layer = spatialConsistencyData[windowIndex]
                    if len(blacklistedROIs) > 0:
                        import nibabel as nib
                        dataMaskFileName = iniDataFolder+'ROI_parcellations/'+subjId+'/run'+str(runNumber)+'/'+subjId+'_final_BN_Atlas_246_1mm.nii'
                        maskImg = nib.load(dataMaskFileName)
                        maskData = maskImg.get_fdata()
                        dataFileName = iniDataFolder+subjId+'/run'+str(runNumber)+'/detrended_maxCorr5comp.nii'
                        img = nib.load(dataFileName)
                        data = img.get_fdata()
                        nWindows = nc.get_number_of_layers(data.shape, windowLength, windowOverlap)
                        startTimes, endTimes = nc.get_start_and_end_times(nWindows, windowLength, windowOverlap)
                        for ROI in layer['ROI_sizes']:
                            ROIVoxelMaskIndices = np.rint([maskData[voxel] for voxel in ROI])
                            blacklistedIndices = np.array([ROIVoxelMaskIndex in blacklistedROIs for ROIVoxelMaskIndex in ROIVoxelMaskIndices])
                            if np.all(blacklistedIndices):
                                blacklistCounter += 1
                                continue
                            elif np.any(blacklistedIndices):
                                if blacklistWholeROIs:
                                    blacklistCounter += 1
                                    continue
                                else:
                                    modifiedCounter += 1
                                    ROISizes[i].append(layer['ROI_sizes'][ROI] - np.sum(blacklistedIndices))
                                    ROI = [ROI[index] for index in np.where(1 - blacklistedIndices)[0]]
                                    allVoxelTs = np.array([data[voxel[0], voxel[1], voxel[2], startTimes[windowIndex]:endTimes[windowIndex]] for voxel in ROI])
                                    voxelIndices = np.arange(0,len(ROI))
                                    consistencyType = layer['consistency_type']
                                    if 'pearson c' in consistencyType: # a hack to fix mistyped consistency type
                                        consistencyType = 'pearson c'
                                    pooledConsistencies[i].append(cbc.calculateSpatialConsistency(({'allVoxelTs':allVoxelTs, 'consistencyType':consistencyType, 'ftransform':layer['ftransform']},voxelIndices)))
                            else:
                                ROISizes[i].append(layer['ROI_sizes'][ROI])
                                pooledConsistencies[i].append(layer['consistencies'][ROI])
                    else:
                        ROISizesInLayer = layer['ROI_sizes'].values()
                        ROISizes[i].extend(ROISizesInLayer)
                        nROIs.append(len(ROISizesInLayer))
                        nSingles.append(np.sum(np.array(ROISizesInLayer) == 1))
                        pooledConsistencies[i].extend(layer['consistencies'].values())
        meanNROIs.append(np.mean(nROIs))
        stdNROIs.append(np.std(nROIs))
        meanNSingles.append(np.mean(nSingles))
        stdNSingles.append(np.std(nSingles))
        sizes = np.array(ROISizes[i])
        meanSizes.append(sizes[sizes>1].mean())
        stdSizes.append(sizes[sizes>1].std())
        if len(blacklistedROIs) > 0:
            nBlacklisted.append(blacklistCounter/(len(subjectIds)*len(runNumbers)*nWindows))
            nModified.append(modifiedCounter/(len(subjectIds)*len(runNumbers)*nWindows))

    f = open(pooledDataSavePath, 'wb')
    pooledData = {'pooledConsistencies':pooledConsistencies,'ROISizes':ROISizes,'meanNROIs':meanNROIs,'stdNROIs':stdNROIs,'meanNSingles':meanNSingles,'stdNSingles':stdNSingles,'meanSizes':meanSizes,'stdSizes':stdSizes}
    if len(blacklistedROIs) > 0:
        pooledData['nBlacklisted'] = nBlacklisted
        pooledData['nModified'] = nModified
    pickle.dump(pooledData, f)
    f.close()

print 'Clustering methods:', jobLabels
print 'Mean N ROIs:', meanNROIs
print 'STD N ROIs:', stdNROIs
print 'Mean N singles:', meanNSingles
print 'STD N singles:', stdNSingles
print 'Mean ROI size:', meanSizes
print 'STD ROI size:', stdSizes
if len(blacklistedROIs) > 0:
    print 'On average N ROIs removed by blacklisting:', nBlacklisted
    print 'On average N ROIs modified by blacklisting:', nModified

# calculating and visualizing consistency distributions
consFig = plt.figure(1)
consAx = consFig.add_subplot(111)
consBins = np.linspace(-1,1,nConsBins+1)
consBinCenters = (consBins[1::] + consBins[:-1])/2.

if not excludeSizes:
    sizeFig = plt.figure(2)
    sizeAx = sizeFig.add_subplot(111)
    sizeAx.set_xscale('log')
    minSizes = []
    maxSizes = []
    for sizes in ROISizes:
        sortedSizes = np.sort(np.unique(sizes))
        if excludeSingleVoxels and sortedSizes[0]==1:
            minSize = sortedSizes[1] # the smallest unique size is 1
        else:
            minSize = sortedSizes[0]
        maxSize = sortedSizes[-1]
        minSizes.append(minSize)
        maxSizes.append(maxSize)
    minSize = np.min(minSizes)
    maxSize = np.max(maxSizes)
    sizeBins = binner.Bins(float, minSize, maxSize, 'linlog', sizeBinFactor)  
    sizeBinEdges = np.array(sizeBins.bin_limits)
    sizeBinCenters = (sizeBinEdges[1:] + sizeBinEdges[:-1])/2.

    consSizeFig = plt.figure(3)
    consSizeAx = consSizeFig.add_subplot(111)
    consSizeAx.set_xscale('log')

for pooledConsistency, sizes, jobLabel, clusteringMethod, color, alpha in zip(pooledConsistencies, ROISizes, jobLabels, clusteringMethods, colors, alphas):
    if excludeSingleVoxels:
        sizes = np.array(sizes)
        pooledConsistency = np.array(pooledConsistency)
        nTotal = float(len(pooledConsistency))
        mask = np.where(sizes > 1)
        pooledConsistency = pooledConsistency[mask]
        sizes = sizes[mask]
        fracExcluded = 1 - len(mask[0])/nTotal

    consDistribution, consBinEdges = np.histogram(pooledConsistency,bins=consBins,density=True)
    if clusteringMethod == '':
        label = jobLabel
    else:
        label = clusteringMethod
    if excludeSingleVoxels:
        label = label + ', fraction 1-voxel ROIs  %f'  %(fracExcluded)

    consAx.plot(consBinCenters,consDistribution,color=color,alpha=alpha,label=label)

    if not excludeSizes:
        sizeDistribution = sizeBins.bin_count_divide(sizes)/len(sizes)
        sizeAx.plot(sizeBinCenters,sizeDistribution,color=color,alpha=alpha,label=label)
       
        consPerSizeData = np.array([[size, consistency] for consistency, size in zip(pooledConsistency, sizes)])
        consPerSize = sizeBins.bin_average(consPerSizeData)
        consSizeAx.plot(sizeBinCenters,consPerSize,color=color,alpha=alpha,label=label)

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
    #sizeAx.legend()

    plt.tight_layout()
    plt.savefig(sizeFigureSavePath,format='pdf',bbox_inches='tight')

    plt.figure(3)

    consSizeAx.set_xlabel('ROI size (number of voxels)')
    consSizeAx.set_ylabel('Consistency')
    
    plt.tight_layout()
    plt.savefig(consSizeFigureSavePath,format='pdf',bbox_inches='tight')

