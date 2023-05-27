# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:27:24 2019

@author: onerva

A script for visualizing the consistency of ROIs obtained with different clustering
strategies (template, sklearn, consistency optimization). This is a pooling and
visualization script; before running this, the consistencies should be calculated
elsewhere (e.g. with testPipelineWithConsistencyClustering).

For now, let's assume that the network construction has been done once per clustering
method for each subject so we don't need to care about the network identificators.

At the moment, consistency is pooled across subjects and windows to compare 
clustering methods. 

"""
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import binned_statistic

preprocessLevelFolder = '/media/onerva/KINGSTON/test-data/outcome/test-pipeline/'
subjects = ['010']
runNumber = '1'
clusteringMethods = ['template_clustering','consistency_optimized','craddock','random_balls']#,'sklearn']
template = 'brainnetome'
nClusters = 100 # this is needed for reading the consistencies of the sklearn ROIs
windowLength = 100
overlap = 0
nWindows = 2
consistencyFile = 'spatial-consistency.pkl'

nBins = 50

pooledConsistencies = [[] for clusteringMethod in clusteringMethods]

for subject in subjects:
    for i, clusteringMethod in enumerate(clusteringMethods):
        networkLevelFolder = preprocessLevelFolder + '/' + subject + '/' + runNumber + '/' + clusteringMethod + '/' + template
        for folder in os.listdir(networkLevelFolder):
            if 'net_' in folder:
                break
        for windowNo in range(nWindows):
            consistencyPath = networkLevelFolder + '/' + folder + '/' + str(windowNo) + consistencyFile
            f = open(consistencyPath, "rb")
            consistencyData = pickle.load(f)
            f.close()
            consistency = consistencyData['consistencies']
            pooledConsistencies[i].extend(consistency)
            
fig = plt.figure()
ax = fig.add_subplot(111)

for pooledConsistency,clusteringMethod in zip(pooledConsistencies,clusteringMethods):
    distribution,binEdges,_ = binned_statistic(pooledConsistency,pooledConsistency,statistic='count',bins=nBins)
    binWidth = (binEdges[1] - binEdges[0])
    binCenters = binEdges[1:] - binWidth/2
    distribution = distribution/float(np.sum(distribution))
    plt.plot(binCenters,distribution,label=clusteringMethod)
ax.set_xlabel('Consistency')
ax.set_ylabel('PDF')
ax.legend()
plt.show()
    
            
        

