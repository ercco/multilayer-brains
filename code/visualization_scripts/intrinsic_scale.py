#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:04:18 2024

@author: onervak

A script for calculating the intrinsic scale (as per Murray et al. 2014, Nature Neuroscience 17) for each ROI based on its layerwise overlap with its initial configuration.

NOTE: this script should be run with Python 3, not with Python 2.7
"""
import matplotlib.pylab as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

data_path_base = '/home/onervak/projects/multilayer-brains/article_figs/trajectories/' #'/m/nbe/scratch/alex/private/tarmo/trajectories/'
clustering_methods = ['test'] #['ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100', 'random', 'craddock']
reference_layer = 0

n_bins = 100
color = 'k'
ls = '-'

def exponential_decay(lag, A, intrinsic_scale, B):
    """
    Exponential decay function: K(lag) = A*(exp(-lag/intrinsic_scale) + B)

    Parameters
    ----------
    lag : float or array of floats
        the independent variable
    A : float
        a normalization constant
    intrinsic_scale : float
        a parameter related to the intrinsic scale
    B : float
        a normalization constant

    Returns
    -------
    K : float or array of floats
        the exponential decay values corresponding to lag
    """
    K = A * (np.exp(-lag/intrinsic_scale) + B)
    return K

for clustering_method in clustering_methods:
    data_path = data_path_base + '/' + clustering_method + '/' + str(reference_layer) + '.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    intrinsic_scales = []
    A = []
    B = []
    for subject in list(data.keys())[0:1]:
        for run in list(data[subject].keys())[0:1]:
            for roi in data[subject][run].keys():
                roi_trajectory = data[subject][run][roi]
                lag, autocorrelation, _, _ = plt.acorr(roi_trajectory, maxlags=None)
                lag = lag[len(roi_trajectory)::]
                autocorrelation = autocorrelation[len(roi_trajectory)::]
                fit = curve_fit(exponential_decay, lag, autocorrelation, p0=[25, 250, -1], maxfev=10000) # TODO: try to figure out an initial guess so the number of evaluations could be decreased
                intrinsic_scales.append(fit[0][1])
                A.append(fit[0][0])
                B.append(fit[0][2])
                print('fit for ' + subject + ', run ' + str(run))
                
    #import pdb; pdb.set_trace()
    is_pdf, is_bin_edges, _ = binned_statistic(intrinsic_scales, intrinsic_scales, statistic='count', bins=n_bins)
    is_pdf = is_pdf / float(np.sum(is_pdf * np.abs(is_bin_edges[0] - is_bin_edges[1])))
    is_bin_centers = 0.5 * (is_bin_edges[:-1] + is_bin_edges[1:]) 
    
    a_pdf, a_bin_edges, _ = binned_statistic(A, intrinsic_scales, statistic='count', bins=n_bins)
    a_pdf = a_pdf / float(np.sum(a_pdf * np.abs(a_bin_edges[0] - a_bin_edges[1])))
    a_bin_centers = 0.5 * (a_bin_edges[:-1] + a_bin_edges[1:])
    b_pdf, b_bin_edges, _ = binned_statistic(B, intrinsic_scales, statistic='count', bins=n_bins)
    b_pdf = b_pdf / float(np.sum(b_pdf * np.abs(b_bin_edges[0] - b_bin_edges[1])))
    b_bin_centers = 0.5 * (b_bin_edges[:-1] + b_bin_edges[1:])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(is_bin_centers, is_pdf, color=color, ls=ls)
    ax.set_xlabel('Intrinsic scale')
    ax.set_ylabel('PDF')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(a_bin_centers, a_pdf, color=color, ls=ls)
    ax.set_xlabel('A')
    ax.set_ylabel('PDF')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(b_bin_centers, b_pdf, color=color, ls=ls)
    ax.set_xlabel('B')
    ax.set_ylabel('PDF')
    
    
    

    
                
                






