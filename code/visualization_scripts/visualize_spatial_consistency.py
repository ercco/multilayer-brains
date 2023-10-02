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

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]
nLayers = 2

# path parts for reading data
consistencySaveStem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
jobLabels = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that consistencySaveStem, jobLabel, clusteringMethods, and savePath (specified further below) match your data

# path parths for saving
consFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/consistency_distributions.pdf'
sizeFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/size_distributions.pdf'
consSizeFigureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/size_vs_consistency.pdf'

# distribution and visualization parameters
nConsBins = 50
sizeBinFactor = 1.5
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
excludeSizes = False
excludeSingleVoxels = True

# reading data
pooledConsistencies = [[] for clusteringMethod in clusteringMethods] # this is a clustering methods x ROIs list (ROIs pooled over subjects, layers, and runs)
ROISizes = [[] for clusteringMethod in clusteringMethods]

meanNROIs = []
stdNROIs = []
meanNSingles = []
stdNSingles = []
meanSizes = []
stdSizes = []

for i, (jobLabel,clusteringMethod) in enumerate(zip(jobLabels,clusteringMethods)):
    nROIs = []
    nSingles = []
    for subjId in subjectIds:
        for runNumber in runNumbers:
            if clusteringMethod == '':
                savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
            else:
                savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency_' + clusteringMethod + '.pkl'
            f = open(savePath,'r')
            spatialConsistencyData = pickle.load(f)
            f.close()
            for layer in spatialConsistencyData.values():
                ROISizesInLayer = layer['ROI_sizes']
                ROISizes[i].extend(ROISizesInLayer)
                nROIs.append(len(ROISizesInLayer))
                nSingles.append(np.sum(np.array(ROISizesInLayer) == 1))
                pooledConsistencies[i].extend(layer['consistencies'])
    meanNROIs.append(np.mean(nROIs))
    stdNROIs.append(np.std(nROIs))
    meanNSingles.append(np.mean(nSingles))
    stdNSingles.append(np.std(nSingles))
    sizes = np.array(ROISizes[i])
    meanSizes.append(sizes[sizes>1].mean())
    stdSizes.append(sizes[sizes>1].std())

print 'Clustering methods:', jobLabels
print 'Mean N ROIs:', meanNROIs
print 'STD N ROIs:', stdNROIs
print 'Mean N singles:', meanNSingles
print 'STD N singles:', stdNSingles
print 'Mean ROI size:', meanSizes
print 'STD ROI size:', stdSizes

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

