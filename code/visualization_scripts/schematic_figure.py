#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:55:34 2023

@author: onerva

A script for visualizing the outputs of schematic_simulation.py and to create the schematic figure (Fig. 1)
"""
import matplotlib.pylab as plt

from surfplot import Plot
from surfplot.utils import threshold
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import mni152_to_fslr
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

template_path = '/home/onerva/projects/ROIplay/templates/brainnetome/BNA-MPM_thr25_4mm.nii'
underlying_parcellation_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/spherical_parcellation_sigma5.nii'
underlying_voxel_labels_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/spherical_parcellation_labels_sigma5.pkl'
optimized_voxel_labels_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/optimized_parcellation_labels_sigma5.pkl'
optimized_parcellation_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/optimized_parcellation_sigma5.nii'
ReHo_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/ReHo_sigma5.nii'

underlying_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_underlying_sigma5.pdf'
optimized_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_optimized_sigma5.pdf'
brainnetome_figure_save_path = '/home/onerva/projects/multilayer-meta/article_figs/schematic_fig/schematic_fig_brainnetome_sigma5.pdf'

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

# visualize ReHo + underlying ROIs

ReHo_gii_lh, ReHo_gii_rh = mni152_to_fslr(ReHo_save_path)

ReHo_lh = ReHo_gii_lh.agg_data()
ReHo_rh = ReHo_gii_rh.agg_data()

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red_r'])

underlying_gii_lh, underlying_gii_rh = mni152_to_fslr(underlying_parcellation_save_path, method='nearest')
underlying_lh = threshold(underlying_gii_lh.agg_data(), 0)
underlying_rh = threshold(underlying_gii_rh.agg_data(), 0)

p.add_layer({'left': underlying_lh, 'right': underlying_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(underlying_figure_save_path, format='pdf',bbox_inches='tight')

# visualize ReHo + optimized ROIs

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red_r'])

optimized_gii_lh, optimized_gii_rh = mni152_to_fslr(optimized_parcellation_save_path, method='nearest')
optimized_lh = threshold(optimized_gii_lh.agg_data(), 0)
optimized_rh = threshold(optimized_gii_rh.agg_data(), 0)

p.add_layer({'left': optimized_lh, 'right': optimized_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(optimized_figure_save_path, format='pdf',bbox_inches='tight')

# visualize ReHo + Brainnetome ROIs

p = Plot(lh, rh)
p.add_layer({'left': ReHo_lh, 'right': ReHo_rh}, cmap=nilearn_cmaps['black_red_r'])

template_gii_lh, template_gii_rh = mni152_to_fslr(template_path, method='nearest')
template_lh = threshold(template_gii_lh.agg_data(), 1)
template_rh = threshold(template_gii_rh.agg_data(), 1)

p.add_layer({'left': template_lh, 'right': template_rh}, cmap='gray', as_outline=True, cbar=False)
fig = p.build()

plt.savefig(brainnetome_figure_save_path, format='pdf',bbox_inches='tight')
