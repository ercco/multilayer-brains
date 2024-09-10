"""
A script for visualizing the temporal changes in spatial consistency.
"""
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
import pickle
import os
import sys

subj_ids = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']

run_numbers = [2,3,4,5,6,7,8,9,10]
nLayers = 2

# path parts for reading data
consistency_save_stem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
job_labels = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03','craddock'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject

# path parths for saving
pooled_data_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/pooled_spatial_consistency_dynamics.pkl'
figure_save_path = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/article_figs/consistency_dynamics.pdf'

# visualization parameters
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]
std_alpha = 0.05
exclude_single_voxels = True

visualize = True

if os.path.isfile(pooled_data_save_path):
    f = open(pooled_data_save_path, 'rb')
    pooled_data = pickle.load(f)
    f.close()
    mean_consistencies = pooled_data['mean_consistencies']
    std_consistencies = pooled_data['std_consistencies']

else: # reading data
    pooled_consistencies = [[] for job_label in job_labels] # this will be a clustering method x (time_window x run) x ROI list, pooled across subjects
    pooled_sizes = [[] for job_label in job_labels]
    mean_consistencies = [] # this will be a clustering method x (time_window x run) list, averaged across ROIs and subjects
    std_consistencies = []

    for i, job_label in enumerate(job_labels):
        for subj_id in subj_ids:
            layer_index = 0
            for run_number in run_numbers:
                save_path = consistency_save_stem + '/' + subj_id + '/' + str(run_number) + '/' + job_label + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
                f = open(save_path,'r')
                spatial_consistency_data = pickle.load(f)
                f.close()
                for window_index in spatial_consistency_data:
                    if len(pooled_consistencies[i]) <= layer_index:
                        pooled_consistencies[i].append(spatial_consistency_data[window_index]['consistencies'].values())
                        pooled_sizes[i].append(spatial_consistency_data[window_index]['ROI_sizes'].values())
                    else:
                        pooled_consistencies[i][layer_index].extend(spatial_consistency_data[window_index]['consistencies'].values())
                        pooled_sizes[i][layer_index].extend(spatial_consistency_data[window_index]['ROI_sizes'].values())
                    layer_index += 1
    
    for consistencies, sizes, job_label, color, alpha in zip(pooled_consistencies, pooled_sizes, job_labels, colors, alphas):
        mean_consistency = np.zeros(len(consistencies))
        std_consistency = np.zeros(len(consistencies))
        for i, (consistency, size) in enumerate(zip(consistencies, sizes)):
            consistency = np.array(consistency)
            size = np.array(size)
            if exclude_single_voxels:
                mask = np.where(size > 1)
                consistency = consistency[mask]
            mean_consistency[i] = np.mean(consistency)
            std_consistency[i] = np.std(consistency)
        mean_consistencies.append(mean_consistency)
        std_consistencies.append(std_consistency)

    f = open(pooled_data_save_path, 'wb')
    pooled_data = {'mean_consistencies':mean_consistencies,'std_consistencies':std_consistencies}
    pickle.dump(pooled_data, f)
    f.close()

if visualize:
    # calculating and visualizing temporal changes in consistency
    fig = plt.figure(1)
    mean_ax = fig.add_subplot(211)
    std_ax = fig.add_subplot(212)

    for mean_consistency, std_consistency, job_label, color, alpha in zip(mean_consistencies, std_consistencies, job_labels, colors, alphas):
        x = np.arange(mean_consistency.shape[0])
        mean_ax.plot(x, mean_consistency, color=color, alpha=alpha, label=job_label)
        std_ax.plot(x, std_consistency, color=color, alpha=alpha, label=job_label)

    mean_ax.set_xlabel('Time window')
    mean_ax.set_ylabel('Mean spatial consistency')
    mean_ax.legend()

    std_ax.set_xlabel('Time window')
    std_ax.set_ylabel('Std of spatial consistency')

    plt.tight_layout()
    plt.savefig(figure_save_path, format='pdf',bbox_inches='tight')



