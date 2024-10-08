"""
A script for visualizing spatial consistency calculated in the window where ROIs were defined (t) and calculated in window t + timelag using the same ROI definitions
"""
import nibabel as nib
import pickle
import os.path
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

import clustering_by_consistency as cbc

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']
runNumbers = [2,3,4,5,6,7,8,9,10]

# path parts for reading data
consistencyInNextWindowSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/consistency_in_next_window_no_single_voxels'
jobLabels = ['template_brainnetome', 'random_balls', 'craddock','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that data paths, jobLabels, clusteringMethods, and savePath (specified further below) match your data

# path parts for saving
consistencyInNextWindowFigDataSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/pooled_data_no_single_voxels'
consistencyInNextWindowFigureSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/consistency_in_next_window_no_single_voxels'

# visualization parameters
timelag = 1
nBins = 50
nZoomBins = 20
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
excludeSingleVoxelROIs = True
cmap = 'viridis'
visualize = True

presentWindowPercentiles = []
nextWindowPercentiles = []

binnedInfo = []
distInfo = []
pooledPresent = []
pooledNext = []

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
        presentWindowConsistencies = np.array(presentWindowConsistencies)
        nextWindowConsistencies = np.array(nextWindowConsistencies)
        mask = np.where(np.logical_and(presentWindowConsistencies != 1, nextWindowConsistencies != 1))
        presentWindowConsistencies = presentWindowConsistencies[mask]
        nextWindowConsistencies = nextWindowConsistencies[mask]        
 
    absoluteDifference = np.array(presentWindowConsistencies) - np.array(nextWindowConsistencies)
    meanAbsoluteDifference = np.mean(absoluteDifference)
    stdAbsoluteDifference = np.std(absoluteDifference)
    relativeDifference = (np.array(presentWindowConsistencies) - np.array(nextWindowConsistencies)) / np.array(presentWindowConsistencies)
    meanRelativeDifference = np.mean(relativeDifference)
    stdRelativeDifference = np.std(relativeDifference)
    relativeSummary = np.mean(absoluteDifference) / np.mean(presentWindowConsistencies)
    pooledPresent.append(presentWindowConsistencies)
    pooledNext.append(nextWindowConsistencies)
    presentWindowPercentiles.append(np.percentile(presentWindowConsistencies, [5, 95]))
    nextWindowPercentiles.append(np.percentile(nextWindowConsistencies, [5, 95]))

    print('%s, mean absolute present - next: %.10f') % (jobLabel, meanAbsoluteDifference)
    print('%s, std absolute present - next: %.10f') % (jobLabel, stdAbsoluteDifference)
    print('%s, mean relative present - next: %.10f') % (jobLabel, meanRelativeDifference)
    print('%s, std relative present - next: %.10f') % (jobLabel, stdRelativeDifference)
    print('%s, relative summary measure: %.10f') % (jobLabel, relativeSummary)

    if visualize:
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
        bins = np.linspace(min(min(presentWindowConsistencies), min(nextWindowConsistencies)), max(max(nextWindowConsistencies),max(presentWindowConsistencies)), nBins + 1)
        binCenters = bins[:-1] + (bins[1::] - bins[:-1])/2
        ret = binned_statistic_2d(presentWindowConsistencies, nextWindowConsistencies, presentWindowConsistencies, statistic='count', bins=bins)
        averageStat = binned_statistic(presentWindowConsistencies, nextWindowConsistencies, statistic='mean', bins=bins).statistic
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ret.statistic[np.where(ret.statistic == np.amin(ret.statistic))] = 'nan'
        binnedInfo.append((ret.x_edge, ret.y_edge, binCenters, ret.statistic, averageStat))
        plt.pcolor(ret.x_edge, ret.y_edge, np.transpose(ret.statistic), cmap=cmap)
        plt.plot(binCenters, averageStat, color='c')
        ax.set_xlabel('Consistency in present window')
        ax.set_ylabel('Consistency in the window present + %x' %timelag)
        ax.set_title(title)
        plt.colorbar(label='Count')
        diagonal = [max(ret.x_edge[0],ret.y_edge[0]), min(ret.x_edge[-1],ret.y_edge[-1])]
        plt.plot(diagonal, diagonal, color='r')
        plt.tight_layout()
        plt.savefig(imshowSavePath,format='pdf',bbox_inches='tight')
        plt.close()
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dists = np.zeros(ret.statistic.shape)
        averageStat = np.zeros(ret.statistic.shape[0])
        for i in range(nBins):
            y = list(nextWindowConsistencies[np.where((bins[i] <= np.array(presentWindowConsistencies)) & (np.array(presentWindowConsistencies) < bins[i + 1]))])
            if i == nBins - 1:
                y2 = np.array(nextWindowConsistencies)[np.where(np.array(presentWindowConsistencies) == bins[i + 1])]
                y.extend(y2)
            d, _ = np.histogram(y, bins=bins, density=True)
            m = np.nanmean(y)
            if np.all(np.isnan(d)):
                d = np.zeros(len(d))
                m = 0
            dists[:, i] = d
            averageStat[i] = m
        dists[np.where(dists == np.amin(dists))] = 'nan'
        distInfo.append((ret.x_edge, ret.y_edge, binCenters, dists, averageStat))
        plt.pcolor(ret.x_edge, ret.y_edge, dists, cmap=cmap)
        plt.plot(binCenters, averageStat, color='c')
        ax.set_xlabel('Consistency in present window')
        ax.set_ylabel('Consistency in the window present + %x' %timelag)
        ax.set_title(title)
        plt.colorbar(label = 'PDF(y)')
        plt.plot(diagonal, diagonal, color='r')
        plt.tight_layout()
        plt.savefig(distributionSavePath, format='pdf', bbox_inches='tight')
        plt.close()

if visualize:
    x_min = np.amin(np.array(presentWindowPercentiles)[:,0])
    x_max = np.amax(np.array(presentWindowPercentiles)[:,1])
    y_min = np.amin(np.array(nextWindowPercentiles)[:,0])
    y_max = np.amax(np.array(nextWindowPercentiles)[:,1])

    zoom_min = min(x_min, y_min)
    zoom_max = max(x_max, y_max)

    print '5% and 95% percentiles in x direction:'
    print presentWindowPercentiles
    print '5% and 95% percentiles in y direction:'
    print nextWindowPercentiles

    for presentWindowConsistencies, nextWindowConsistencies, ret, dist, clusteringMethod, jobLabel in zip(pooledPresent, pooledNext, binnedInfo, distInfo, clusteringMethods, jobLabels):
        if clusteringMethod == '':
            zoomBinnedSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_zoomed_binned.pdf'
            zoomSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_zoomed.pdf'
            zoomDistSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_zoomed_ydist.pdf'
            title = jobLabel
        else:
            zoomBinnedSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_zoomed_binned.pdf'
            zoomSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_zoomed.pdf'
            zoomDistSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_zoomed_ydist.pdf'
            title = jobLabel + ', ' + clusteringMethod
        xmask = np.where((zoom_min < dist[0]) & (dist[0] < zoom_max))
        ymask = np.where((zoom_min < dist[1]) & (dist[1] < zoom_max))
        x = dist[0][xmask]
        y = dist[1][ymask]
        binCenterMask = np.where((zoom_min < dist[2]) & (dist[2] < zoom_max))
        binCenters = dist[2][binCenterMask]

        zoomedStatistic = ret[3][xmask].T[ymask]
        zoomedStatisticAverageStat = ret[4][binCenterMask]
        zoomedDist = dist[3][ymask].T[xmask].T # in dist[3], rows correspond to next window, columns to present window
        zoomedDistAverageStat = dist[4][binCenterMask]
        if np.any(binCenters > max(x)) or np.any(binCenters < min(x)):
            zoomedStatisticAverageStat = zoomedStatisticAverageStat[np.where((binCenters <= max(x)) & (binCenters >= min(x)))]
            zoomedDistAverageStat = zoomedDistAverageStat[np.where((binCenters <= max(x)) & (binCenters >= min(x)))]
            binCenters = binCenters[np.where((binCenters <= max(x)) & (binCenters >= min(x)))]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(x, y, zoomedStatistic, cmap=cmap)
        plt.plot(binCenters, zoomedStatisticAverageStat, color='c')
        ax.set_xlabel('Consistency in present window')
        ax.set_ylabel('Consistency in the window present + %x' %timelag)
        ax.set_title(title)
        plt.colorbar(label='Count')
        plt.plot([min(x), max(x)], [min(y), max(y)], color='r')
        plt.tight_layout()
        plt.savefig(zoomBinnedSavePath, format='pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(x, y, zoomedDist, cmap=cmap)
        plt.plot(binCenters, zoomedDistAverageStat, color='c')
        ax.set_xlabel('Consistency in present window')
        ax.set_ylabel('Consistency in the window present + %x' %timelag)
        ax.set_title(title)
        plt.colorbar(label='PDF(y)')
        plt.plot([min(x), max(x)], [min(y), max(y)], color='r')
        plt.tight_layout()
        plt.savefig(zoomSavePath, format='pdf', bbox_inches='tight')
        plt.close()

        xBins = np.linspace(x_min, x_max, nZoomBins + 1)
        yBins = np.linspace(y_min, y_max, nZoomBins + 1)

        mask = np.where((x_min < presentWindowConsistencies) & (presentWindowConsistencies < x_max) & (y_min < nextWindowConsistencies) & (nextWindowConsistencies < y_max))
        presentWindowConsistencies = presentWindowConsistencies[mask]
        nextWindowConsistencies = nextWindowConsistencies[mask]

        dists = np.zeros((nZoomBins, nZoomBins))
        for i in range(nZoomBins):
            y = list(nextWindowConsistencies[np.where((xBins[i] <= np.array(presentWindowConsistencies)) & (np.array(presentWindowConsistencies) < xBins[i + 1]))])
            if i == nZoomBins - 1:
                y2 = np.array(nextWindowConsistencies)[np.where(np.array(presentWindowConsistencies) == xBins[i + 1])]
                y.extend(y2)
            d, _ = np.histogram(y, bins=yBins, density=True)
            if np.all(np.isnan(d)):
                d = np.zeros(len(d))
            dists[:, i] = d

        dists[np.where(dists == np.amin(dists))] = 'nan'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(xBins, yBins, dists, cmap=cmap)
        ax.set_xlabel('Consistency in present window')
        ax.set_ylabel('Consistency in the window present + %x' %timelag)
        ax.set_title(title)
        plt.colorbar(label='PDF(y)')
        plt.plot([x_min, x_max], [y_min, y_max], color='r')
        plt.tight_layout()
        plt.savefig(zoomDistSavePath, formt='pdf', bbox_inches='tight')
        plt.close()

