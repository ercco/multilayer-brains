#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:55:34 2023

@author: onerva

A script for visualizing the outputs of schematic_simulation.py and to create the schematic figure (Fig. 1)
"""
import matplotlib.pylab as plt
import numpy as np

from surfplot import Plot
from surfplot.utils import threshold
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import mni152_to_fslr
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

template_path = '/home/onerva/projects/ROIplay/templates/brainnetome/BNA-MPM_thr25_4mm_nosub.nii'
underlying_parcellation_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/spherical_parcellation_sigma5_v1.nii'
underlying_voxel_labels_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/spherical_parcellation_labels_sigma5_v1.pkl'
optimized_voxel_labels_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/optimized_parcellation_labels_sigma5_v1.pkl'
optimized_parcellation_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/optimized_parcellation_sigma5_v1.nii'
ReHo_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/ReHo_sigma5_v1.nii'

underlying_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_underlying_sigma5_v1.svg'
optimized_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_optimized_sigma5_v1.svg'
brainnetome_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_brainnetome_sigma5_v1.svg'

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

# visualize ReHo + underlying ROIs

ReHo_gii_lh, ReHo_gii_rh = mni152_to_fslr(ReHo_save_path)

ReHo_lh = ReHo_gii_lh.agg_data()
ReHo_rh = ReHo_gii_rh.agg_data()

sizes_l = [np.sum(np.where(ReHo_lh == ReHo)) for ReHo in np.unique(ReHo_lh)]
medial_wall_value_l = np.unique(ReHo_lh)[np.where(sizes_l == max(sizes_l))]
medial_wall_indices_l = np.where(ReHo_lh == medial_wall_value_l)
ReHo_lh[medial_wall_indices_l] = np.nan

sizes_r = [np.sum(np.where(ReHo_rh == ReHo)) for ReHo in np.unique(ReHo_rh)]
medial_wall_value_r = np.unique(ReHo_rh)[np.where(sizes_r == max(sizes_r))]
medial_wall_indices_r = np.where(ReHo_rh == medial_wall_value_r)
ReHo_rh[medial_wall_indices_r] = np.nan

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red'])

#underlying_gii_lh, underlying_gii_rh = mni152_to_fslr(underlying_parcellation_save_path, method='nearest')
#underlying_lh = threshold(underlying_gii_lh.agg_data(), 0)
#underlying_rh = threshold(underlying_gii_rh.agg_data(), 0)

#underlying_lh[medial_wall_indices_l] = np.nan
#underlying_rh[medial_wall_indices_r] = np.nan

#p.add_layer({'left': underlying_lh, 'right': underlying_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(underlying_figure_save_path, format='svg',bbox_inches='tight')

# visualize ReHo + optimized ROIs

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red'])

optimized_gii_lh, optimized_gii_rh = mni152_to_fslr(optimized_parcellation_save_path, method='nearest')
optimized_lh = threshold(optimized_gii_lh.agg_data(), 0)
optimized_rh = threshold(optimized_gii_rh.agg_data(), 0)

optimized_lh[medial_wall_indices_l] = np.nan
optimized_rh[medial_wall_indices_r] = np.nan

p.add_layer({'left': optimized_lh, 'right': optimized_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(optimized_figure_save_path, format='svg',bbox_inches='tight')

# visualize ReHo + Brainnetome ROIs

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red'])

template_gii_lh, template_gii_rh = mni152_to_fslr(template_path, method='nearest')

template_lh = threshold(template_gii_lh.agg_data(), 1)
template_rh = threshold(template_gii_rh.agg_data(), 1)

template_lh[medial_wall_indices_l] = np.nan 
template_rh[medial_wall_indices_r] = np.nan

p.add_layer({'left': template_lh, 'right': template_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(brainnetome_figure_save_path, format='svg',bbox_inches='tight')
