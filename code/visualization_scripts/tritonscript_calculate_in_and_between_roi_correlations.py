"""
A script for calculating the voxel-voxel correlations inside and between ROIs. This script is meant for being run on Triton, in parallel over subjects. After running this, run visualize_in_and_between_roi_correlations.py to visualize the correlation distributions.
"""
import numpy as np
from scipy.stats import binned_statistic
import os.path
import sys
import itertools

import clustering_by_consistency as cbc

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]

combinedArray = list(itertools.product(subjectIds,runNumbers)) # len=225

# Path parts for reading data
subjectFolder = '/m/nbe/scratch/alex/private/tarmo/article_runs/maxcorr/'
niiDataFileStem = '/m/nbe/scratch/alex/private/janne/preprocessed_ini_data/'
niiDataFileName = 'detrended_maxCorr5comp.nii'
clusteringMethods = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03']
netIdentificator = '2_layers/net'
nLayers = 2

# Path parts for saving
correlationSaveFolder = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/in_and_between_roi_correlations/means/'

# Time window parameters
timewindow = 80 # This is the time window length used to construct the ROIs
overlap = 0 # This is the overlap between consequent time windows

# Distribution parameters
nBins = 100

# Needed for calculating distributions
def getDistribution(data, nBins):
    """
    Calculates the PDF of the given data
    
    Parameters:
    -----------
    data: a container of data points, e.g. list or np.array
    nBins: int, number of bins used to calculate the distribution
    
    Returns:
    --------
    pdf: np.array, PDF of the data
    binCenters: np.array, points where pdf has been calculated
    """
    count, binEdges, _ = binned_statistic(data, data, statistic='count', bins=nBins)
    pdf = count/float(np.sum(count*(binEdges[1]-binEdges[0])))
    binCenters = 0.5*(binEdges[:-1]+binEdges[1:])

    return pdf, binCenters

if __name__ == '__main__':
    index = int(sys.argv[1])
    subjId,runNumber = combinedArray[index]
    for clusteringMethod in clusteringMethods:
        print 'calculating correlations... ' + clusteringMethod + ', ' + str(runNumber) 
        niiDataFilePath = niiDataFileStem + subjId + '/run' + str(runNumber) + '/' + niiDataFileName
        layersetwiseNetworkSavefolder = subjectFolder + '/' + subjId + '/' + str(runNumber) + '/' + clusteringMethod + '/' + netIdentificator
        allFileNames = os.listdir(layersetwiseNetworkSavefolder)
        # let's pick the network files to be read
        # same layer is saved in multiple files; therefore we read only every nLayer-th file
        allFileNames = [str(n) + '_' + str(n+1) for n in range(0,len(allFileNames),nLayers)]
        savePath = correlationSaveFolder + '/in-between-correlations_' + clusteringMethod + '_subject_' + subjId + '_run_' + str(runNumber) + '.pkl'
        correlationData = cbc.calculateCorrelationsInAndBetweenROIs([niiDataFilePath],[layersetwiseNetworkSavefolder],
                                                                    allFileNames,nLayers,timewindow,overlap,savePath,nBins=nBins,normalizeDistributions=False)

