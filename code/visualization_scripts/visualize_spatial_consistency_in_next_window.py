"""
A script for visualizing spatial consistency calculated in the window where ROIs were defined (t) and calculated in window t + timelag using the same ROI definitions
"""
import nibabel as nib
import pickle
import os.path
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import binned_statistic_2d

import clustering_by_consistency as cbc

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']
runNumbers = [2,3,4,5,6,7,8,9,10]

# path parts for reading data
consistencyInNextWindowSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/consistency_in_next_window'
jobLabels = ['craddock']#,'ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that data paths, jobLabels, clusteringMethods, and savePath (specified further below) match your data

# path parts for saving
consistencyInNextWindowFigDataSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/pooled_data'
consistencyInNextWindowFigureSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/consistency_in_next_window'

# visualization parameters
timelag = 1
nBins = 50
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
excludeSingleVoxelROIs = True

for jobLabel, clusteringMethod, color, alpha in zip(jobLabels, clusteringMethods, colors, alphas):
    if clusteringMethod == '':
        figDataSavePath = consistencyInNextWindowFigDataSaveStem + '_' + jobLabel + '.pkl'
    else:
        figDataSavePath = consistencyInNextWindowFigDataSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pkl'
    if os.path.isfile(figDataSavePath):
        f = open(figDataSavePath, 'rb')
        figData = pickle.load(f)
        f.close()
        presentWindowConsistencies = figData['presentWindowConsistencies']
        nextWindowConsistencies = figData['nextWindowConsistencies']

    else:
        presentWindowConsistencies = []
        nextWindowConsistencies = []
    
        for subjId in subjectIds:
            for runNumber in runNumbers:
                print('Running %s, %s, %x') % (jobLabel, subjId, runNumber) 
                if clusteringMethod == '':
                    savePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_subject_' + subjId + '_run_' + str(runNumber) + '.pkl'
                else:
                    savePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_subject_' + subjId + '_run_' + str(runNumber) + '.pkl'
                f = open(savePath, 'rb')
                nextWindowConsistencyData = pickle.load(f)
                f.close()
                for ROI in nextWindowConsistencyData.keys():
                    if isinstance(nextWindowConsistencyData[ROI], list):
                        for c in nextWindowConsistencyData[ROI]:
                            presentWindowConsistencies.append(c[0])
                            nextWindowConsistencies.append(c[1])
                    else:
                        presentWindowConsistencies.append(nextWindowConsistencyData[ROI][0])
                        nextWindowConsistencies.append(nextWindowConsistencyData[ROI][1])

        f = open(figDataSavePath, 'wb')
        pickle.dump({'presentWindowConsistencies':presentWindowConsistencies, 'nextWindowConsistencies':nextWindowConsistencies},f)
        f.close()
    
    if excludeSingleVoxelROIs:
        for c1, c2 in zip(presentWindowConsistencies, nextWindowConsistencies):
            if c1 == 1 or c2 == 1:
                presentWindowConsistencies.remove(c1)
                nextWindowConsistencies.remove(c2)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot(presentWindowConsistencies, nextWindowConsistencies, color=color, alpha=alpha)
    #ax.set_xlabel('Consistency in present window')
    #ax.set_ylabel('Consistency %x windows after the present' %timelag)
    if clusteringMethod == '':
        title = jobLabel
        figureSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '.pdf'
        imshowSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_binned.pdf'
        distributionSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_ydist.pdf'
    else:
        title = jobLabel + ', ' + clusteringMethod
        figureSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pdf'
        imshowSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_binned.pdf'
        distributionSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_ydist.pdf'
    #ax.set_title(title)
    #plt.tight_layout()
    #plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')
    bins = np.linspace(min(min(presentWindowConsistencies), min(nextWindowConsistencies)), max(max(nextWindowConsistencies),max(presentWindowConsistencies)), nBins + 1)
    ret = binned_statistic_2d(presentWindowConsistencies, nextWindowConsistencies, presentWindowConsistencies, statistic='count', bins=bins)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.pcolor(ret.x_edge, ret.y_edge, np.transpose(ret.statistic))
    ax.set_xlabel('Consistency in present window')
    ax.set_ylabel('Consistency in the window present + %x' %timelag)
    ax.set_title(title)
    plt.colorbar(label='Count')
    diagonal = [max(ret.x_edge[0],ret.y_edge[0]), min(ret.x_edge[-1],ret.y_edge[-1])]
    #diagonal = [bins[0], bins[-1]] 
    plt.plot(diagonal, diagonal, color='r')
    plt.tight_layout()
    plt.savefig(imshowSavePath,format='pdf',bbox_inches='tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dists = np.zeros(ret.statistic.shape)
    for i in range(nBins):
        y = list(np.array(nextWindowConsistencies)[np.where((bins[i] <= np.array(presentWindowConsistencies)) & (np.array(presentWindowConsistencies) < bins[i + 1]))])
        if i == nBins - 1:
            y2 = np.array(nextWindowConsistencies)[np.where(np.array(presentWindowConsistencies) == bins[i + 1])]
            y.extend(y2)
        d, _ = np.histogram(y, bins=bins, density=True)
        if np.all(np.isnan(d)):
            d = np.zeros(len(d))
        dists[:, i] = d
    plt.pcolor(ret.x_edge, ret.y_edge, dists)
    ax.set_xlabel('Consistency in present window')
    ax.set_ylabel('Consistency in the window present + %x' %timelag)
    ax.set_title(title)
    plt.colorbar(label = 'PDF(y)')
    plt.plot(diagonal, diagonal, color='r')
    plt.tight_layout()
    plt.savefig(distributionSavePath, format='pdf', bbox_inches='tight')
