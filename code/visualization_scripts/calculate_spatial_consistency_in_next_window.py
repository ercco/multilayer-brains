"""
A script for visualizing spatial consistency calculated in the window where ROIs were defined (t) and calculated in window t + timelag using the same ROI definitions
"""
import os
import sys
import itertools
import nibabel as nib
import pickle

import clustering_by_consistency as cbc

subjectIds = ['d4w','b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']
runNumbers = [2,3,4,5,6,7,8,9,10]

combinedArray = list(itertools.product(subjectIds,runNumbers))

# path parts for reading data
niiDataFileStem = '/m/nbe/scratch/alex/private/janne/preprocessed_ini_data/'
niiDataFileName = '/detrended_maxCorr5comp.nii'
consistencySaveStem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
netIdentificator = '2_layers/net'
nLayers = 2
jobLabels = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that data paths, jobLabels, clusteringMethods, and savePath (specified further below) match your data

# path parts for saving
consistencyInNextWindowSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/spatial_consistency/next_window/consistency_in_next_window_no_single_voxels'

# time window parameters
timewindow = 80 # This is the time window length used to construct the ROIs
overlap = 0 # This is the overlap between consequent time windows

# parameters for calculating consistency
timelag = 1
nCPUs = 5
excludeSingleVoxels = True

# visualization parameters

if __name__ == '__main__':
    index = int(sys.argv[1])
    subjId,runNumber = combinedArray[index]
    for jobLabel, clusteringMethod in zip(jobLabels, clusteringMethods):
        presentWindowConsistencies = []
        nextWindowConsistencies = []
        if clusteringMethod == '':
            savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
        else:
            savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency_' + clusteringMethod + '.pkl'
        f = open(savePath,'r')
        spatialConsistencyData = pickle.load(f)
        f.close()
        filename = niiDataFileStem +subjId+ '/run' + str(runNumber) + niiDataFileName
        img = nib.load(filename)
        imgdata = img.get_fdata()
        if 'pearson c' in spatialConsistencyData[0]['consistency_type']:
            spatialConsistencyData[0]['consistency_type'] = 'pearson c' # this is a hack: before 2023-10-02, the consistency type was mistyped in calculation phase
        nextWindowConsistencyData = cbc.calculateSpatialConsistencyInNextWindow(spatialConsistencyData, imgdata, timewindow, overlap, timelag, nCPUs, excludeSingleVoxels)
        if clusteringMethod == '':
            savePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_subject_' + subjId + '_run_' + str(runNumber) + '.pkl'
        else:
            savePath = consistencyInNextWindowSaveStem + '_' + jobLabel + '_' + clusteringMethod + '_subject_' + subjId + '_run_' + str(runNumber) + '.pkl'
        with open(savePath, 'wb') as f:
            pickle.dump(nextWindowConsistencyData, f, -1)


