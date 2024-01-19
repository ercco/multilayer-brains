#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:55:34 2023

@author: onerva

A script for creating the schematic figure (Fig. 1)
"""
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt

from surfplot import Plot
from surfplot.utils import threshold
from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps.transforms import mni152_to_fslr, mni152_to_fsaverage

#from brainspace.datasets import load_parcellation

load_data = True
template_path = '/m/cs/scratch/networks/aokorhon/ROIplay/templates/brainnetome/BNA-MPM_thr25_4mm.nii'
data_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/test_slice.nii'
data_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/test_slice'
figure_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/schematic_fig/schematic_fig.pdf'

# plotting the Brainnetome ROI boundaries on brain surface

template_img = nib.load(template_path)
template_data = template_img.get_fdata()

template_gii_lh, template_gii_rh = mni152_to_fslr(template_path, method='nearest')
template_lh = threshold(template_gii_lh.agg_data(), 3)
template_rh = threshold(template_gii_rh.agg_data(), 3)

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
sulc_lh, sulc_rh = surfaces['sulc']

p = Plot(lh, rh)
p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r', cbar=False)
p.add_layer({'left': template_lh, 'right': template_rh}, cmap='gray', as_outline=True, cbar=False)

# plotting data

if load_data:
    data_img = nib.load(data_path)
    data = data_img.get_fdata()

    data_lh, data_rh = mni152_to_fslr(data_path)
    np.save(data_save_path + '_lh', data_lh)
    np.save(data_save_path + '_rh', data_rh)
    
else:
    data_lh = np.load(data_save_path + '_lh')
    data_rh = np.load(data_save_path + '_rh')
    
data_lh = threshold(data_lh.agg_data(), 3)
data_rh = threshold(data_rh.agg_data(), 3)

p.add_layer({'left': data_lh, 'right': data_rh})

fig = p.build()
plt.savefig(figure_save_path, format='pdf',bbox_inches='tight')

