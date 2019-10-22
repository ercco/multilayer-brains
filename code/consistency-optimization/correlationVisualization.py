#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:11:19 2019

@author: onerva

A small script for visualizing voxel-voxel correlation distributions inside and between ROIs,
assuming that the distributions have already been calculated and saved as pkl.
"""
import cPickle as pickle
import matplotlib.pylab as plt

import onion_parameters as params

dataPath = '/media/onerva/KINGSTON/test-data/outcome/correlation-distributions-weighted-mean-consistency.pkl'
figureSavePath = '/media/onerva/KINGSTON/test-data/outcome/correlation-distributions-weighted-mean-consistency.pdf'

originalColor = params.originalColor
originalAlpha = params.originalAlpha
optimizedColor = params.optimizedColor
optimizedAlpha = params.optimizedAlpha
inROILs = params.inROILs
betweenROILs = params.betweenROILs

f = open(dataPath, "rb")
data = pickle.load(f)
f.close()

originalInROIDist, originalInROIBinCenters = data['originalInROI']
originalBetweenROIDist, originalBetweenROIBinCenters = data['originalBetweenROI']
optimizedInROIDist, optimizedInROIBinCenters = data['optimizedInROI']
optimizedBetweenROIDist, optimizedBetweenROIBinCenters = data['optimizedBetweenROI']

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(originalInROIBinCenters,originalInROIDist,color=originalColor,alpha=originalAlpha,ls=inROILs,label='original, inside ROIs')
plt.plot(originalBetweenROIBinCenters,originalBetweenROIDist,color=originalColor,alpha=originalAlpha,ls=betweenROILs,label='original, between ROIs')
plt.plot(optimizedInROIBinCenters,optimizedInROIDist,color=optimizedColor,alpha=optimizedAlpha,ls=inROILs,label='optimized, inside ROIs')
plt.plot(optimizedBetweenROIBinCenters,optimizedBetweenROIDist,color=optimizedColor,alpha=optimizedAlpha,ls=betweenROILs,label='optimized, between ROIs')

ax.set_xlabel('Pearson correlation coefficient')
ax.set_ylabel('PDF')
ax.legend()

plt.tight_layout()
plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')



