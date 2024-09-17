"""
A script for reading the earlier-calculated distributions of voxel-voxel correlations inside and between ROIs from files and for visualizing them.
"""

import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

runNumbers = [2,3,4,5,6,7,8,9,10]

# path parts for reading data
correlationSaveFolder = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/in_and_between_roi_correlations/means/'
clusteringMethods = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03']

# path parts for saving
figureSavePath = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/correlation_distributions_in_and_between_rois.pdf'

# visualization parameters:
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
inROILs = '-'
betweenROILs = '--'

visualize = False

inROIDistributions = [[] for clusteringMethod in clusteringMethods]
betweenROIDistributions= [[] for clusteringMethod in clusteringMethods]

for i, clusteringMethod in enumerate(clusteringMethods):
    meanInROICorrelations = []
    meanBetweenROICorrelations = []
    initializeDistributions = True
    for subjectId in subjectIds:
        for runNumber in runNumbers:
            savePath = correlationSaveFolder + '/in-between-correlations_' + clusteringMethod + '_subject_' + subjectId + '_run_' + str(runNumber) + '.pkl'
            f = open(savePath,'r')
            correlationData = pickle.load(f)
            f.close()
            if initializeDistributions:
                inROIDistributions[i] = correlationData['inROIDistribution']
                betweenROIDistributions[i] = correlationData['betweenROIDistribution']
                initializeDistributions = False
            else:
                inROIDistributions[i] = inROIDistributions[i] + correlationData['inROIDistribution']
                betweenROIDistributions[i] = betweenROIDistributions[i] + correlationData['betweenROIDistribution']
            meanInROICorrelations.append(correlationData['meanWithinROICorrelation'])
            meanBetweenROICorrelations.append(correlationData['meanBetweenROICorrelation'])
    if i == 0:
        binCenters = correlationData['binCenters'] # same bins are used for all clustering methods so the centers are read only once
    inROIDistributions[i] = inROIDistributions[i]/float(np.sum(inROIDistributions[i]*np.abs(binCenters[0]-binCenters[1])))        
    betweenROIDistributions[i] = betweenROIDistributions[i]/float(np.sum(betweenROIDistributions[i]*np.abs(binCenters[0]-binCenters[1]))) 
    meanInROICorrelation = np.sum(meanInROICorrelations) / len(meanInROICorrelations)
    meanBetweenROICorrelation = np.sum(meanBetweenROICorrelations) / len(meanBetweenROICorrelations)
    print 'Mean in-ROI correlation, {method}: {value}'.format(method=clusteringMethod, value=str(meanInROICorrelation))
    print 'Mean between-ROI correlation, {method}: {value}'.format(method=clusteringMethod, value=str(meanBetweenROICorrelation))

if visualize:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for inROIDistribution, betweenROIDistribution, clusteringMethod, color, alpha in zip(inROIDistributions, betweenROIDistributions, clusteringMethods, colors, alphas):
        plt.plot(binCenters,inROIDistribution,color=color,alpha=alpha,ls=inROILs,label=clusteringMethod+', inside ROI')
        plt.plot(binCenters,betweenROIDistribution,color=color,alpha=alpha,ls=betweenROILs,label=clusteringMethod+ ', between ROIs')

    ax.set_xlabel('Pearson correlation coefficient')
    ax.set_ylabel('PDF')
    ax.legend()

    plt.tight_layout()
    plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')            
