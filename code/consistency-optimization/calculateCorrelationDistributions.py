#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:10:32 2019

@author: onerva

A script for calculating the distribution of voxel-voxel correlations inside and
between ROIs for multiple parcellations.
"""
import numpy as np
import matplotlib.pylab as plt
from scipy import io
import cPickle as pickle

from scipy.stats import binned_statistic

import os.path
import sys
if os.path.exists('/home/onerva/dippa/local'):
    sys.path.insert(0,'/home/onerva/dippa/local')
elif os.path.exists('home/onerva/projects/dippa/local'):
    sys.path.insert(0,'/home/onerva/projects/dippa/local')
else:
    sys.path.insert(0,'/scratch/cs/networks/aokorhon/dippa/code')     

import clustering_by_consistency as cbc
import onion_parameters as params

subjects = ['/scratch/cs/networks/aokorhon/multilayer/010/']
originalROIInfoFile = '/scratch/cs/networks/aokorhon/multilayer/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
optimizedROIInfoFile = 'optimized-rois-test-for-Tarmo-mean-weighted-consistency'
allVoxelTsFileName = '/roi_voxel_ts_all_rois4mm_FWHM0.mat'
figureSavePath = '/scracth/cs/networks/aokorhon/multilayer/outcome/correlation-distributions-weighted-mean-consistency_NEWTEST.pdf'
nBins = 100

returnCorrelations = False

originalColor = params.originalColor
originalAlpha = params.originalAlpha
optimizedColor = params.optimizedColor
optimizedAlpha = params.optimizedAlpha
inROILs = params.inROILs
betweenROILs = params.betweenROILs

import pdb; pdb.set_trace() 

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
    
    
# path parts
subjectFolder = '/m/cs/scratch/networks/aokorhon/multilayer/'
subjects = ['010/1']
niiDataFileNames = ['/m/cs/scratch/networks/aokorhon/multilayer/010/epi_STD_mask_detrend_fullreg.nii'] # This should have the same length as the subjects list
runNumbers = ['1','2']
clusteringMethods = ['consistency_optimized','template_clustering']
templateNames = [['brainnetome','random'],['brainnetome']] # this is a methods x templates structure
netIdentificators = [[['net_100_0_2019-02-19T16.26.24/2_layers'],['net_100_0_2019-03-22T18.40.27/2_layers']],[['net_100_0_2019-02-19T13.40.36/2_layers']]] # this should be a clustering methods x templates x subjects list (they may have different network identificators); is there any way to do this automatically?
nLayers = 2
allFileNames = ['0_1']

# time window parameters
timewindow = 100 # This is the time window length used to construct the ROIs
overlap = 0 # This is the overlap between consequent time windows

# visualization parameters:
colors = ['r','k']
alphas = [0.9,0.5]
inROILs = '-'
betweenROILs = '--'

calculateCorrelations = True

inROICorrelations = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames] # this is a clustering methods x template names x ROIs list (ROIs pooled over subjects and layers)
betweenROICorrelations = [[[] for templateName in templateNamesPerMethod] for templateNamesPerMethod in templateNames]

if calculateCorrelations:
    for i, (clusteringMethod, templateNamesPerMethod, netIdentificatorsPerMethod) in enumerate(zip(clusteringMethods,templateNames,netIdentificators)):
        for j, (templateName, netIdentificatorsPerTemplate) in enumerate(zip(templateNamesPerMethod,netIdentificatorsPerMethod)):
            print 'calculating correlations... ' + clusteringMethod + ', ' + templateName
            layersetwiseNetworkSavefolders = [subjectFolder + '/' + subject + '/' + clusteringMethod + '/' + templateName + '/' + netIdentificator for subject,netIdentificator in zip(subjects,netIdentificatorsPerTemplate)]
            #savePath = subjectFolder + '/in-between-correlations_' + clusteringMethod + '_' + templateName + '.pkl'
            savePath = None
            correlationData = cbc.calculateCorrelationsInAndBetweenROIs(niiDataFileNames,layersetwiseNetworkSavefolders,
                                                                            allFileNames,nLayers,timewindow,overlap,savePath,
                                                                            nBins=nBins,returnCorrelations=returnCorrelations)
            inROICorrelations[i][j].extend(correlationData['inROICorrelations'])
            betweenROICorrelations[i][j].extend(correlationData['betweenROICorrelations'])
            
else:
    for i, (clusteringMethod, templateNamesPerMethod, netIdentificatorsPerMethod) in enumerate(zip(clusteringMethods,templateNames,netIdentificators)):
        for j, (templateName, netIdentificatorsPerTemplate) in enumerate(zip(templateNamesPerMethod,netIdentificatorsPerMethod)):
            savePath = subjectFolder + '/in-between-correlations_' + clusteringMethod + '_' + templateName + '.pkl'
            f = open(savePath,'r')
            correlationData = pickle.load(f)
            f.close()
            inROICorrelations[i][j].extend(correlationData['inROICorrelations'])
            betweenROICorrelations[i][j].extend(correlationData['betweenROICorrelations'])
            
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#for inROICorrelationsPerMethod, betweenROICorrelationsPerMethod, clusteringMethod, templateNamesPerMethod, color, alpha in zip(inROICorrelations, betweenROICorrelations, clusteringMethods, templateNames, colors, alphas):
#    for inROICorrelation, betweenROICorrelation, templateName in zip(inROICorrelationsPerMethod, betweenROICorrelationsPerMethod, templateNamesPerMethod):
#        inDistribution,inBinCenters = getDistribution(inROICorrelation,nBins)
#        betweenDistribution,betweenBinCenters = getDistribution(betweenROICorrelation,nBins)
#        plt.plot(inBinCenters,inDistribution,color=color,alpha=alpha,ls=inROILs,label=clusteringMethod+', ' + templateName + ', inside ROI')
#        plt.plot(betweenBinCenters,betweenDistribution,color=color,alpha=alpha,ls=betweenROILs,label=clusteringMethod+', ' + templateName + ', between ROIs')
#
#ax.set_xlabel('Pearson correlation coefficient')
#ax.set_ylabel('PDF')
#ax.legend()
#
#plt.tight_layout()
#plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')

    

