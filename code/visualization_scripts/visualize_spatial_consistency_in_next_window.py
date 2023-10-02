"""
A script for visualizing spatial consistency calculated in the window where ROIs were defined (t) and calculated in window t + timelag using the same ROI definitions
"""
import nibabel as nib
import pickle
import matplotlib.pylab as plt

import clustering_by_consistency as cbc

subjectIds = ['b1k','d3a','d4w','d6i','e6x','g3r','i2p','i7c','m3s','m8f','n5n','n5s','n6z','o9e','p5n','p9u','q4c','r9j','t1u','t9n','t9u','v1i','v5b','y6g','z4t']
runNumbers = [2,3,4,5,6,7,8,9,10]

# path parts for reading data
niiDataFileStem = '/m/nbe/scratch/alex/private/janne/preprocessed_ini_data/'
niiDataFileName = '/detrended_maxCorr5comp.nii'
consistencySaveStem = '/scratch/nbe/alex/private/tarmo/article_runs/maxcorr'
netIdentificator = '2_layers/net'
nLayers = 2
#jobLabels = ['template_brainnetome','craddock','random_balls','ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100','ReHo_seeds_min_correlation_voxelwise_thresholding_03'] # This label specifies the job submitted to Triton; there may be several jobs saved under each subject
jobLabels = ['craddock', 'random_balls']
clusteringMethods = ['','','','','']
# NOTE: before running the script, check that data paths, jobLabels, clusteringMethods, and savePath (specified further below) match your data

# path parts for saving
consistencyInNextWindowFigureSaveStem = '/m/cs/scratch/networks/aokorhon/multilayer/outcome/consistency_in_next_window'

# time window parameters
timewindow = 80 # This is the time window length used to construct the ROIs
overlap = 0 # This is the overlap between consequent time windows

# parameters for calculating consistency
timelag = 1
nCPUs = 5

# visualization parameters
colors = ['r','k','b','g','c']
alphas = [0.9,0.5,0.9,0.9,0.9]

for jobLabel, clusteringMethod, color, alpha in zip(jobLabels, clusteringMethods, colors, alphas):
    presentWindowConsistencies = []
    nextWindowConsistencies = []
    for subjId in subjectIds:
        for runNumber in runNumbers:
            filename = niiDataFileStem +subjId+ '/run' + str(runNumber) + niiDataFileName
            img = nib.load(filename)
            imgdata = img.get_fdata()
            if clusteringMethod == '':
                savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency.pkl'
            else:
                savePath = consistencySaveStem + '/' + subjId + '/' + str(runNumber) + '/' + jobLabel + '/' + str(nLayers) + '_layers' + '/spatial_consistency_' + clusteringMethod + '.pkl'
            f = open(savePath,'r')
            spatialConsistencyData = pickle.load(f)
            f.close()
            if 'pearson c' in spatialConsistencyData[0]['consistency_type']:
                spatialConsistencyData[0]['consistency_type'] = 'pearson c' # this is a hack: before 2023-10-02, the consistency type was mistyped in calculation phase
            nextWindowConsistencyData = cbc.calculateSpatialConsistencyInNextWindow(spatialConsistencyData, imgdata, timewindow, overlap, timelag, nCPUs)
            for ROI in nextWindowConsistencyData.keys():
                presentWindowConsistencies.append(nextWindowConsistencyData[ROI][0])
                nextWindowConsistencies.append(nextWindowConsistencyData[ROI][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(presentWindowConsistencies, nextWindowConsistencies, color=color, alpha=alpha)
    ax.set_xlabel('Consistency in present window')
    ax.set_ylabel('Consistency %x windows after the present' %timelag)
    ax.set_title(jobLabel + ', ' + clusteringMethod)
    plt.tight_layout() 
    if clusteringMethod == '':
        title = jobLabel
        figureSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '.pdf'
    else:
        title = job_label + ', ' + clusteringMethod
        figureSavePath = consistencyInNextWindowFigureSaveStem + '_' + jobLabel + '_' + clusteringMethod + '.pdf'
    plt.savefig(figureSavePath,format='pdf',bbox_inches='tight')


