"""
A script for calculating p-values (through Student's T-test for means of independent samples)
of earlier-calculated and pooled spatial consistency (and ROI size) data.
"""
import numpy as np
import pickle
from scipy.stats import ttest_ind

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r',
              'i2p','i7c','m3s','m8f','n5n','n5s',
              'n6z','o9e','p5n','p9u','q4c','r9j',
              't1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]
nLayers = 2

pooledDataSavePaths = ['/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/pooled_spatial_consistency_data_for_fig_max_size.pkl',
                       '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/pooled_spatial_consistency_data_random_seeds.pkl']
jobLabels = [['ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100'], 
             ['random_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100', 'constrained_random_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100']]
# there may be multiple job labels per pooled data file
referenceJobLabel = 'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100' # label of the job, against which the p-values will be calculated

allPooledConsistencies = []
allROISizes = []
allJobLabels = [label for labelsPerPooledData in jobLabels for label in labelsPerPooledData]
referenceIndex = allJobLabels.index(referenceLabel)

assert any(referenceJobLabel in jobLabelsPerPooledData for jobLabelsInPooledData in jobLabels), 'the reference job label is not included in the pooled data, cannot calculate p-values'

for pooledDataSavePath in pooledDataSavePaths, jobLabels:
    f = open(pooledDataSavePath, 'rb')
    pooledData = pickle.load(f)
    f.close()
    allPooledConsistencies.append(pooledData['pooledConsistencies'])
    allROISizes.append(pooledData['ROISizes'])

for pooledConsistencies, ROISizes, jobLabel in zip(allPooledConsistencies, allROISizes, allJobLabels):
    meanConsistency = np.mean(pooledConsistencies)
    meanROISize = np.mean(ROISizes)
    print 'Clustering method:', jobLabel
    print 'Mean consistency:', meanConsistency
    print 'Mean ROI size', meanROISize
    if not jobLabel == referenceJobLabel:
        _, consistencyP = ttest_ind(allPooledConsistencies[referenceIndex], pooledConsistencies)
        _, sixeP = ttest_ind(allROISizes[referenceIndex], ROISizes)
        print 'Consistency p-value:' consistencyP
        print 'Size p-value': sizeP
