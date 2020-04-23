# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:31:06 2019

@author: onerva

Functions for clustering voxels by maximizing the spatial consistency (= mean Pearson
correlation coefficient between the time series of voxels in the cluster). See
functionwise documentation for further details.

"""
import numpy as np
import pickle
import nibabel as nib
import decimal
import random
import heapq

import network_io, network_construction

from scipy import io, matrix
from scipy.stats import binned_statistic
from scipy.stats.stats import pearsonr
from scipy.sparse import csc_matrix
from concurrent.futures import ProcessPoolExecutor as Pool

# Data io

#TODO: check how ROI coordinates, centroids, etc. are handled in Tarmo's code
def readROICentroids(ROIInfoFile, readVoxels=False, fixCentroids=False, resolution=4):
    """
    Reads ROI data, in particular the coordinates of ROI centroids. An option
    for reading coordinates of all voxels exist. 
    
    Parameters:
    -----------
    ROIInfoFile: str, path to the file that contains information about ROIs
    readVoxels: bool, will voxel coordinates be read? (default=False)
    fixCentroids: bool, if fixCentroids=True, the function will return as centroid 
                  the one of ROI's voxels that is closest to the actual centroid.
    resolution: int, the physical distance (in mm) between two points in the voxel space
                (used for transforming centroids into the MNI space if the ROIInfoFile
                doesn't contain the MNI coordinates; default: 4).
        
    Returns:
    --------
    ROICentroids: nROIs x 3 np.array, coordinates (in voxels) of the centroids of the ROIs
    ROIMNICentroids: nROIs x 3 np.array, coordinates (in mm) of the centroids of the ROIs
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    ROIMaps: list of ROISize x 3 np.arrays, coordinates (in voxels) of voxels belonging to each ROI
    """
    if ROIInfoFile[-4::] == '.mat':
        infoData = io.loadmat(ROIInfoFile)
        ROIInfo = infoData['rois'][0]
        nROIs = ROIInfo.shape[0] # number of ROIs
        ROICentroids = np.zeros((nROIs,3),dtype=int)
        ROIMNICentroids = np.zeros((nROIs,3))
        voxelCoordinates = []
        ROIMaps = []
        for i, ROI in enumerate(ROIInfo):
            centroid = np.array(ROI['centroid'][0]) - np.array([1,1,1]) # correcting for the indexing difference between Matlab and Spyder
            if fixCentroids:
                ROIMap = ROI['map'] - np.ones(ROI['map'].shape,dtype=int)
                distances = np.zeros(ROIMap.shape[0])
                for j, voxel in enumerate(ROIMap):
                    distances[j] = np.sqrt(np.sum((voxel-centroid)**2))
                centroid = ROIMap[np.where(distances==np.amin(distances))[0][0]] # if multiple voxels are at the same distance from the centroid, the first one is picked
            ROICentroids[i,:] = centroid
            ROIMNICentroids[i,:] = ROI['centroidMNI'][0]
            if readVoxels:
                voxelCoordinates.extend(list(ROI['map'] - np.ones(ROI['map'].shape,dtype=int)))
                ROIMaps.append(ROI['map']-np.ones(ROI['map'].shape,dtype=int))
        voxelCoordinates = np.array(voxelCoordinates)
    # TODO: check the reading of pickle files when the file structure is more established    
    elif ROIInfoFile[-4::] == '.pkl':
        f = open(ROIInfoFile, "rb")
        ROIInfo = pickle.load(f)
        f.close()
        ROIMaps = ROIInfo['ROIMaps']
        ROICentroids = np.zeros((len(ROIMaps),3),dtype=int)
        ROIMNICentroids = np.zeros((len(ROIMaps),3),dtype=int)
        for i, ROIMap in enumerate(ROIMaps):
            centroid = getCentroid(ROIMap)
            ROICentroids[i,:] = centroid
            ROIMNICentroids[i,:] = spaceToMNI(centroid,resolution)
        voxelCoordinates = []
        if readVoxels:
            for ROIMap in ROIMaps:
                if len(ROIMap.shape) == 1:
                    voxelCoordinates.append(ROIMap)
                else:
                    voxelCoordinates.extend(ROIMap)
            voxelCoordinates = np.array(voxelCoordinates)
    else:
        print('Unknown file format; accepted formats are .mat and .pkl')
        
    return ROICentroids, ROIMNICentroids, voxelCoordinates, ROIMaps

def findROICentroids(template, fixCentroids=True):
    """
    Starting from a ROI template image, finds the coordinates of the ROI centroids.

    Parameters:
    -----------
    template: 3D numpy array where each element corresponds to a voxel. The 
                 value of each voxel is the index of the ROI the voxel belongs to.
                 Voxels outside of ROIs (= outside of the gray matter) have value 0.
    fixCentroids: bool, if fixCentroids=True, the function will return as centroid 
                 the one of ROI's voxels that is closest to the actual centroid (default = True).
                 
    Returns:
    --------
    ROICentroids: nROIs x 3 np.array, coordinates (in voxels) of the centroids of the ROIs
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    ROIMaps: list of ROISize x 3 np.arrays, coordinates (in voxels) of voxels belonging to each ROI
    """
    # TODO: check if the MNI coordinates of centroids are needed; if so, they can be easily calculated here
    ROIIndices = list(np.unique(template))
    ROIIndices.remove(0)
    nROIs = len(ROIIndices)
    ROICentroids = np.zeros((nROIs,3),dtype=int)
    voxelCoordinates = []
    ROIMaps = []
    for i, ROIInd in enumerate(ROIIndices):
        ROIVoxels = np.transpose(np.array(np.where(template == ROIInd)))
        voxelCoordinates.extend(ROIVoxels)
        ROIMaps.append(ROIVoxels)
        centroid = getCentroid(ROIVoxels)
        if fixCentroids and not np.any((ROIVoxels==centroid).all(axis=1)):
        # the actual centroid may be located outside of the ROI; as a fix, let's use the voxel closest to the centroid
            distances = np.zeros(ROIVoxels.shape[0])
            for j, voxel in enumerate(ROIVoxels):
                distances[j] = np.sqrt(np.sum((voxel-centroid)**2))
            centroid = np.array([int(x) for x in ROIVoxels[np.where(distances==np.amin(distances))[0][0]]]) # if multiple voxels are at the same distance from the centroid, the first one is picked
        ROICentroids[i,:] = centroid
    voxelCoordinates = np.array(voxelCoordinates)
    return ROICentroids,voxelCoordinates,ROIMaps

def createNii(ROIInfo, savePath, imgSize=[45,54,45], affine=np.eye(4)):
    """
    Based on the given ROIInfo, creates a ROI mask (in .nii format) and saves 
    it to a given path. To construct the image of the NIFTI file, a 3D
    matrix of zeros is created and values of voxels belonging to a ROI are set to
    the index of this ROI.
    
    Parameters:
    -----------
    ROIInfo: dict, contains:
                   ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
    savePath: str, path for saving the mask.
    imgSize: list of ints, dimensions of the image in the nii file. If saving a
             ROI mask created based on existing mask, set to the dimensions of the
             existing mask. (Default: [45,54,45]; the dimensions of the 4mm 
             Brainnetome mask)
    affine: np.array, an image coordination transformation (affine) matrix. (Default:
            an identity matrix)
             
    Returns:
    --------
    No direct output, saves the mask in NIFTI format to the given path
    """
    data = np.zeros(imgSize)
    ROIMaps = ROIInfo['ROIMaps']
    for i, ROI in enumerate(ROIMaps):
        if len(ROI.shape) == 1:
            data[ROI[0],ROI[1],ROI[2]] = i + 1
        else:
            for voxel in ROI:   
                data[voxel[0],voxel[1],voxel[2]] = i + 1
    img = nib.Nifti1Image(data,affine)     
    nib.save(img,savePath)
    
def readVoxelIndices(path, voxelCoordinates=[], layers=all):
    """
    Reads the saved ROIs and based on them constructs the voxel indices for calculating
    consistency. The input file should contain a multilayer network so that nodes of each
    layer are ROIs saved as strings in the format of '[(x,y,z),(x,y,z)]' where (x,y,z) are
    voxel coordinates
    
    Parameters:
    -----------
    path: str, path to the file where the network structure has been saved
    voxelCoordinates: nVoxels x 3 np.array, coordinates of all voxels. NOTE: these need to
                      be in the same order as voxel time series used to calculate consistency!
    layers: list of ints, the numbers of layers on which the voxel indices should be read
            (default = 'all', in which case all layers of the network are investigated)
    
    Returns:
    --------
    voxelIndices: list of lists (nLayers x nROIs) of np.arrays, each array containing indices of voxels of one ROI; 
                  these indices refer to voxels' locations in the voxel time series array.
    readVoxelCoordinates: list of lists (nLayers x nROIs) of np.arrays, each array containing the coordinates of
                  voxels of one ROI
    
    """
    M = network_io.read_pickle_file(path)
    voxelIndices = []
    readVoxelCoordinates = []
    if layers == 'all':
        layers = list(M.get_layers())
    for layer in layers:
        nodes = M.iter_nodes(layer)
        voxelIndicesPerLayer = []
        readVoxelCoordinatesPerLayer = []
        for node in nodes:
            voxelIndicesPerROI = []
            voxels = eval(node)
            readVoxelCoordinatesPerLayer.append(voxels)
            if len(voxelCoordinates)>0:
                for voxel in voxels:
                    voxelIndicesPerROI.append(np.where((voxelCoordinates==voxel).all(axis=1))[0][0])
            voxelIndicesPerLayer.append(np.array(voxelIndicesPerROI))
        voxelIndices.append(voxelIndicesPerLayer)
        readVoxelCoordinates.append(readVoxelCoordinatesPerLayer)
    return voxelIndices, readVoxelCoordinates

# Helper functions

def getRandomCentroids(nROIs, template):
    """
    Picks random coordinates for ROI centroids.
    
    Parameters:
    -----------
    nROIs: int, number of centroids
    template: 3D numpy array where each element corresponds to a voxel. The 
              value of each voxel included to the analysis should be >0 (e.g.
              the index of the ROI the voxel belongs to)-
              Voxels outside of ROIs (= outside of the gray matter) have value 0.
              
    Returns:
    --------
    ROICentroids: nROIs x 3 np.array, coordinates (in voxels) of the random centroids
    """
    x,y,z = np.where(template != 0)
    nVoxels = len(x)
    assert nVoxels >= nROIs,'Number of ROIs is larger than number of voxels, select a smaller number of ROIs'
    voxelCoordinates = np.concatenate((x,y,z)).reshape(3,nVoxels).T
    indices = random.sample(np.arange(nVoxels),nROIs)
    ROICentroids = np.zeros((nROIs,3),dtype=int)
    for i, index in enumerate(indices):
        ROICentroids[i,:] = voxelCoordinates[index,:]
    return ROICentroids

def findNeighbors(voxelCoords, resolution=1, allVoxels=[], nNeighbors=6):
    """
    Returns the neighbors (the ones sharing a face) of a voxel.
    
    Parameters:
    -----------
    voxelCoords: 1x3 np.array, coordinates of a voxel (either in voxels or in mm)
    resolution: double, distance between voxels if coordinates are given in mm;
                if coordinates are given in voxels, use the default value 1 (voxels
                are 1 voxel away from each other).
    allVoxels: iterable, coordinates of all acceptable voxels. If allVoxels is given,
               only neighbors in allVoxels are returned (default: []).
    nNeighbors: int, number of neighbors to return; options: 6 (those sharing faces),
                18 (sharing faces or edges), 26 (sharing faces, edges, or corners)
                
    Returns:
    --------    
    neighbors: 6x3 np.array, coordinates of the closest neighbors of the voxel
    """
    x = voxelCoords[0]
    y = voxelCoords[1]
    z = voxelCoords[2]    
    
    if nNeighbors == 6:
        neighbors = [[x+resolution,y,z],
                     [x-resolution,y,z],
                     [x,y+resolution,z],
                     [x,y-resolution,z],
                     [x,y,z+resolution],
                     [x,y,z-resolution]]
    if nNeighbors == 18:
        neighbors.extend([[x+resolution,y+resolution,z],
                          [x+resolution,y-resolution,z],
                          [x-resolution,y+resolution,z],
                          [x-resolution,y-resolution,z],
                          [x+resolution,y,z+resolution],
                          [x+resolution,y,z-resolution],
                          [x-resolution,y,z+resolution],
                          [x-resolution,y,z-resolution],
                          [x,y+resolution,z+resolution],
                          [x,y+resolution,z-resolution],
                          [x,y-resolution,z+resolution],
                          [x,y-resolution,z-resolution]])
    if nNeighbors == 26:
        neighbors.extend([[x+resolution,y+resolution,z+resolution],
                          [x+resolution,y+resolution,z-resolution],
                          [x+resolution,y-resolution,z+resolution],
                          [x+resolution,y-resolution,z-resolution],
                          [x+resolution,y-resolution,z-resolution],
                          [x-resolution,y+resolution,z+resolution],
                          [x-resolution,y+resolution,z-resolution],
                          [x-resolution,y-resolution,z+resolution],
                          [x-resolution,y-resolution,z-resolution]])
                         
    if not len(allVoxels) == 0:
        acceptedNeighbors = []
        for i, neighbor in enumerate(neighbors):
            if np.any((np.array(allVoxels) == neighbor).all(axis=1)):
                acceptedNeighbors.append(neighbor)
        neighbors = acceptedNeighbors                 
    return neighbors
    
def findROIlessVoxels(voxelCoordinates,ROIInfo):
    """
    Returns the indices and coordinates of voxels that do not belong to any ROI.
    
    Parameters:
    -----------
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    ROIInfo: dic, contains:
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
    Returns:
    --------
    ROIlessVoxels: dic, contains:
                   ROIlessIndices: NROIless x 1 np.array, indices of ROIless voxels in voxelCoordinates
                   ROIlessMap: NROIless x 3 np.array, coordinates of the ROIless voxels
    """
    ROIMaps = ROIInfo['ROIMaps']
    for i, ROI in enumerate(ROIMaps):
        if len(ROI.shape) == 1: # adding an extra dimension to enable concatenation
            ROI = np.array([ROI]) 
        if i == 0:
            inROIVoxels = ROI
        else:
            inROIVoxels = np.concatenate((inROIVoxels,ROI),axis=0)
    ROIlessIndices = []
    ROIlessMap = []
    for i, voxel in enumerate(voxelCoordinates):
        if not np.any((inROIVoxels == voxel).all(axis=1)): # voxel is not found in any ROI map
            ROIlessIndices.append(i)
            ROIlessMap.append(voxel)
    ROIlessIndices = np.array(ROIlessIndices)
    ROIlessMap = np.array(ROIlessMap)
    ROIlessVoxels = {'ROIlessIndices':ROIlessIndices,'ROIlessMap':ROIlessMap}
    return ROIlessVoxels

def findROIlessNeighbors(ROIIndex,voxelCoordinates,ROIInfo):
    """
    Finds the neighboring voxels of a ROI that do not belong to any ROI.
    
    Parameters:
    -----------
    ROIIndex: int, index of the ROI in the lists of ROIInfo (see below)
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    ROIInfo: dic, contains:
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
    Returns:
    --------
    ROIlessNeighbors: dic, contains:
                      ROIlessIndices: NNeihbors x 1 np.array, indices of ROIless neighbor voxels in voxelCoordinates
                      ROIlessMap: NNeighbors x 3 np.array, coordinates of ROIless neighbor voxels
    """
    ROIMap = ROIInfo['ROIMaps'][ROIIndex]
    if len(ROIMap.shape) == 1: # adding an outermost dimension to enable proper indexing later on
        ROIMap = [ROIMap]
    for i, voxel in enumerate(ROIMap):
        if i == 0:
            ROINeighbors = findNeighbors(voxel,allVoxels=voxelCoordinates)
        else:
            ROINeighbors.extend(findNeighbors(voxel,allVoxels=voxelCoordinates))
    if len(ROINeighbors) > 0:
        #print 'Found ' + str(len(ROINeighbors)) + 'ROIless neighbors'
        ROINeighbors = np.unique(ROINeighbors,axis=0) # removing dublicates
        ROIlessNeighbors = findROIlessVoxels(ROINeighbors,ROIInfo) 
        ROIlessMap = ROIlessNeighbors['ROIlessMap']
        ROIlessIndices = np.zeros(ROIlessMap.shape[0],dtype=int) # indices in the list of neighbors
        for i, voxel in enumerate(ROIlessMap):
            ROIlessIndices[i] = np.where((voxelCoordinates==voxel).all(axis=1)==1)[0][0] # finding indices in the voxelCoordinates array (indexing assumes that a voxel is present in the voxelCoordinates only once)
    else:
        ROIlessMap = np.array([])
        ROIlessIndices = np.array([])
    ROIlessNeighbors = {'ROIlessIndices':ROIlessIndices,'ROIlessMap':ROIlessMap}
    return ROIlessNeighbors

def findInROINeighbors(voxelIndex,ROILabel,voxelCoordinates,voxelLabels):
    """
    Finds the neighbors of a given voxel that belong to the given ROI.
    
    Parameters:
    -----------
    voxelIndex: int, index of the voxel to investigate in voxelCoordinates and
                voxelLabels
    ROILabel: int (or other identificator), label of the ROI in which to search
              for the neighbors
    voxelCoordinates: iterable, coordinates of all voxels
    voxelLabels: iterable, ROI labels (i.e. indices of ROIs where the 
                 voxels belong) of all voxels; voxels not belongin to any ROI have
                 label -1.
    
    Returns:
    --------
    inROINeighborIndices: list of ints, indices of the in-ROI neighbor voxels in voxelCoordinates
                     and voxelLabels
    """
    neighbors = findNeighbors(voxelCoordinates[voxelIndex],allVoxels=voxelCoordinates)
    neighborIndices = [np.where((voxelCoordinates == neighbor).all(axis=1))[0][0] for neighbor in neighbors]
    inROINeighborIndices = []
    for neighborIndex in neighborIndices:
        if voxelLabels[neighborIndex] == ROILabel:
            inROINeighborIndices.append(neighborIndex)
    return inROINeighborIndices

def findROIBoundary(ROIIndex,ROIInfo,voxelCoordinates,voxelLabels,includeOutsiders=False):
    """
    Finds ROI's boundary voxels (voxels with neighbors in other ROIs).
    
    Parameters:
    -----------
    ROIIndex: int, index of the ROI whose boundary should be found
    ROIInfo: dict, containing (at least):
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    voxelLabels: nVoxels x 1 np.array, ROI labels (i.e. indices of ROIs where the 
                 voxels belong) of all voxels; voxels not belongin to any ROI have
                 label -1.
    includeOutsiders: boolean, if includeOutsiders == True, label -1 is inculded in neighborLabels.
                      Default: False
                           
    Returns:
    --------
    boundaryIndices: list of ints, indices of boundary voxels in voxelCoordinates
    boundaryMap: NBoundary x 3 np.array, coordinates of boundary voxels
    neighborLabels: list of lists of ints, contains for each boundary voxel a list
                     of its neighboring ROIs. By default, label -1 (for voxels outside of ROIs)
                     is not included in the lists.
    """
    #TODO: check the best format for the outputs
    ROIMap = ROIInfo['ROIMaps'][ROIIndex]
    boundaryIndices = []
    boundaryMap = [] 
    neighborLabels = []
    for voxel in ROIMap:
        neighbors = findNeighbors(voxel,allVoxels=voxelCoordinates)
        neighborsInROI = [any((ROIMap==neighbor).all(axis=1)) for neighbor in neighbors]
        if not all(neighborsInROI):
            boundaryIndices.append(np.where((voxelCoordinates==voxel).all(axis=1))[0][0])
            boundaryMap.append(voxel)
            neighborROIs = []
            foreignNeighbors = []
            for i, element in enumerate(neighborsInROI):
                if element == False:
                    foreignNeighbors.append(neighbors[i])
            for foreignNeighbor in foreignNeighbors:
                neighborLabel = voxelLabels[np.where((voxelCoordinates==foreignNeighbor).all(axis=1))[0][0]]
                if includeOutsiders:
                    if not neighborLabel in neighborROIs:
                        neighborROIs.append(neighborLabel)
                else:
                    if not neighborLabel in neighborROIs and not neighborLabel == -1:
                        neighborROIs.append(neighborLabel)
            neighborLabels.append(neighborROIs)
    boundaryMap = np.array(boundaryMap)       
    return boundaryIndices,boundaryMap,neighborLabels

def isBoundary(voxelIndex,voxelCoordinates,voxelLabels):
    """
    Checks if a given voxel belongs to ROI boundary, i.e. has neighbors in other
    ROIs than its own
    
    Parameters:
    -----------
    voxelIndex: int, index of the voxel to investigate
    voxelCoordinatens: iterable, coordinates of all voxels
    voxelLabels: iterable, ROI labels (i.e. indices of ROIs where the 
                 voxels belong) of all voxels; voxels not belongin to any ROI have
                 label -1.
                 
    Returns:
    --------
    isBoundary: boolean, if True, the given voxel is in boundary
    neighborROIs: list of ints (or other ROI labels), the labels of neighboring
                  ROIs of the voxel. If isBoundary == False, neighborROIs == [].
    """
    inROI = voxelLabels[voxelIndex]
    neighbors = findNeighbors(voxelCoordinates[voxelIndex],allVoxels=voxelCoordinates)
    neighborIndices = [np.where((voxelCoordinates == neighbor).all(axis=1))[0][0] for neighbor in neighbors]
    neighborROIs = []
    for neighborIndex in neighborIndices:
        neighborROI = voxelLabels[neighborIndex]
        if neighborROI not in [inROI,-1]:
            if neighborROI not in neighborROIs:
                neighborROIs.append(neighborROI)
    if len(neighborROIs) > 0:
        isBoundary = True
    else:
        isBoundary = False
    return isBoundary, neighborROIs
    
def findVoxelLabels(voxelCoordinates,ROIInfo):
    """
    Finds voxel labels, i.e. the indices of the ROIs where the voxels belong.
    Voxels that don't belong to any ROI get tag -1
    
    Parameters:
    -----------
    ROIInfo: dict, containing (at least):
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    
    Returns:
    --------
    voxelLabels: nVoxels x 1 np.array, ROI labels of all voxels.
    """
    ROIMaps = ROIInfo['ROIMaps']
    voxelLabels = np.zeros(voxelCoordinates.shape[0],dtype=int) - 1
    for i, voxel in enumerate(voxelCoordinates):
        for j, ROI in enumerate(ROIMaps):
            if any((ROI == voxel).all(axis=1)):
                voxelLabels[i] = j
                break
    return voxelLabels
    
def addVoxel(ROIIndex, voxelIndex, ROIInfo, voxelCoordinates):
    """
    Adds the given voxel to the given ROI by updating the ROIInfo dictionary
    accordingly.
    
    Parameters:
    -----------
    ROIIndex: int, index of the ROI to be updated in ROIInfo['ROIMaps'] etc.
    voxelIndex: int, index of the voxel to be added to the ROI in voxelCoordinates
    ROIInfo: dict, containing:
                  ROICentroids: nROIs x 3 np.array, coordinates of the centroids
                  ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
                  ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
                             to each ROI. These indices refer to the rows of the
                             voxelCoordinates array. len(ROIVoxels) = nROIs.
                  ROISizes: nROIs x 1 np.array of ints. Sizes of ROIs defined as
                            number of voxels in the sphere
                  ROINames: list of strs, name of the ROI. If no name is given as
                            input parameter for a ROI, this is set to ''. 
                            len(ROINames) = NROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
                            
    Returns:
    --------
    ROIInfo: dict, original ROIInfo with the given ROI updated
    """
    ROIMap = ROIInfo['ROIMaps'][ROIIndex]
    ROIVoxels = ROIInfo['ROIVoxels'][ROIIndex]
    voxel = np.array([voxelCoordinates[voxelIndex]])
    if len(ROIMap.shape) == 1: # adding an outermost dimension for successful concatenation later on
            ROIMap = np.array([ROIMap])
    ROIMap = np.concatenate((ROIMap,voxel),axis=0)
    if isinstance(ROIVoxels,list):
        ROIVoxels.append(voxelIndex)
    else:
        ROIVoxels = np.concatenate((ROIVoxels,np.array([voxelIndex])),axis=0)
    ROIInfo['ROIMaps'][ROIIndex] = ROIMap
    ROIInfo['ROIVoxels'][ROIIndex] = ROIVoxels
    if 'ROISizes' in ROIInfo.keys():
        ROIInfo['ROISizes'][ROIIndex] = len(ROIVoxels)
    return ROIInfo

def removeVoxel(ROIIndex, voxelIndex, ROIInfo, voxelCoordinates):
    """
    Removes the given voxel from the given ROI by updating the ROIInfo dictionary
    accordingly.
    
     Parameters:
    -----------
    ROIIndex: int, index of the ROI to be updated in ROIInfo['ROIMaps'] etc.
    voxelIndex: int, index of the voxel to be removed from the ROI in voxelCoordinates
    ROIInfo: dict, containing (at least):
                  ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
                  ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
                             to each ROI. These indices refer to the rows of the 
                             voxelCoordinates array. len(ROIVoxels) = nROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
                            
    Returns:
    --------
    ROIInfo: dict, original ROIInfo with the given ROI updated
    """
    ROIMap = ROIInfo['ROIMaps'][ROIIndex]
    ROIVoxels = ROIInfo['ROIVoxels'][ROIIndex]
    voxel = voxelCoordinates[voxelIndex] 
    removeIndex = np.where((ROIMap == voxel).all(axis=1))[0][0]
    ROIMap = np.concatenate((ROIMap[0:removeIndex,:],ROIMap[removeIndex+1::,:]),axis=0)
    if isinstance(ROIVoxels,list):
        ROIVoxels.remove(voxelIndex)
    else:
        ROIVoxels = np.concatenate((ROIVoxels[0:removeIndex],ROIVoxels[removeIndex+1::]),axis=0)
    ROIInfo['ROIMaps'][ROIIndex] = ROIMap
    ROIInfo['ROIVoxels'][ROIIndex] = ROIVoxels
    if 'ROISizes' in ROIInfo.keys():
        ROIInfo['ROISizes'][ROIIndex] = len(ROIVoxels)
    return ROIInfo
    
def getCentroid(coordinates):
    """
    Calculates the centroid (rounded mean) of a given set of coordinates.
    
    Parameters:
    -----------
    coordinates: nRows x 3 np.array, points for calculating the centroid
    
    Returns:
    --------
    centroid: 1 x 3 np.array, centroid of the coordinates
    """
    centroid = np.array([decimal.Decimal(str(x)).quantize(decimal.Decimal('1'),rounding=decimal.ROUND_HALF_UP) for x in np.mean(coordinates,axis=0)],dtype=int)
    # this complicated rounding is used here in order to round up to the closest integer as Matlab does to get results consistent to those produced with Matlab
    return centroid

def getDistanceMatrix(ROICentroids, voxelCoords, save=False, savePath=''):
    """
    Calculates the centroid-to-voxel Euclidean distance between all ROI-voxel
    pairs. If the resolution (and number of voxels) is high, consider setting
    save=True in order to avoid unnecessarely repeating timely calculations.
    
    Parameters:
    -----------
    ROICentroids: nROIs x 3 np.array, coordinates of the centroids of the ROIs.
                  This can be a ROI centroid from an atlas but also any other
                  (arbitrary) point.
    voxelCoords: nVoxels x 3 np.array, coordinates of all voxels
    save: bool, will the distance matrix be saved in a file? (default=False)
    savePath: str, path for saving the distance matrix (default='')
        
    Returns:
    --------
    distanceMatrix: nROIs x nVoxels np.array, matrix of distances of each voxel from each ROI centroid
    """
    if len(ROICentroids) == 1:
        nROIs = 1
    else:
        nROIs = len(ROICentroids)
    nVoxels = len(voxelCoords)
    distanceMatrix = np.zeros((nROIs, nVoxels))
    if nROIs == 1:
        distanceMatrix[0,:] = np.sqrt(np.sum((voxelCoords-ROICentroids)**2,axis=1))
    else:
        for i, centroid in enumerate(ROICentroids):
            distanceMatrix[i,:] = np.sqrt(np.sum((voxelCoords-centroid)**2,axis=1))
    if save:
        distanceData = {}
        distanceData['distanceMatrix'] = distanceMatrix
        with open(savePath, 'wb') as f:
            pickle.dump(distanceData, f, -1)
    return distanceMatrix

def spaceToMNI(coordinate,resolution):
    """
    Transform a set of array indices into the MNI space.
    
    Parameters:
    -----------
    coordinate: 1 x 3 np.array, the indices to be transformed
    resolution: int, the physical distance (in mm) between the elements of the array
    
    Returns:
    --------
    MNICoordinate: 1 x 3 np.array, the corresponding coordinates in MNI space
    """
    origin = np.array([90, -126, -72])
    m = np.array([-1*resolution,resolution,resolution])
    MNICoordinate = coordinate * m + origin
    return MNICoordinate
    
# simple function to translate 1D vector coordinates to 3D matrix coordinates,
# for a 3D matrix of size sz
def indx_1dto3d(idx,sz):
    """
    A simple function to translate 1D vector coordinates to 3D matrix coordinates,
    for a 3D matrix of size sz
    NOTE: This function is directly copied from pyClusterROI (http://ccraddock.github.io/cluster_roi/)
    by Cameron Craddock et al.
    
    Parameters:
    -----------
    idx: int, 1D vector coordinate
    sz: tuple, shape of the 3D matrix
    
    Returns:
    --------
    (x,y,z): coordinates in the 3D matrix
    """
    x=np.divide(idx,np.prod(sz[1:3]))
    y=np.divide(idx-x*np.prod(sz[1:3]),sz[2])
    z=idx-x*np.prod(sz[1:3])-y*sz[2]
    return (x,y,z)

# simple function to translate 3D matrix coordinates to 1D vector coordinates,
# for a 3D matrix of size sz
def indx_3dto1d(idx,sz):
    """
    A simple function to translate 3D matrix coordinates to 1D vector coordinates,
    for a 3D matrix of size sz.
    NOTE: This function is directly copied from pyClusterROI (http://ccraddock.github.io/cluster_roi/)
    by Cameron Craddock et al.
    
    Parameters:
    -----------
    idx: coordinates in the 3D matrix
    sz: shape of the 3D matrix
    
    Returns:
    --------
    idx1: coordinate in the 1D vector
    """
    if(np.rank(idx) == 1):
        idx1=idx[0]*np.prod(sz[1:3])+idx[1]*sz[2]+idx[2]
    else:
        idx1=idx[:,0]*np.prod(sz[1:3])+idx[:,1]*sz[2]+idx[:,2]
    return idx1

def voxelLabelsToROIInfo(voxelLabels,voxelCoordinates,constructROIMaps=True):
    """
    Starting from the voxel label list and voxel coordinates, constructs the ROIInfo
    dict used as an input for many functions.
    
    Parameters:
    -----------
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    voxelLabels: nVoxels x 1 np.array, ROI labels (i.e. indices of ROIs where the 
                 voxels belong) of all voxels; voxels not belongin to any ROI have
                 label -1.
    constructROIMaps: boolean, if True, ROIMaps (see below) are constructed in
                      addition to ROIVoxels. Default = True; set to False to speed
                      things up
                 
    Returns:
    --------
    ROIInfo: dict, contains:
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                      belonging to each ROI. len(ROIMaps) = nROIs. if cosntructROIMaps = False,
                      ROIMaps = [].
             ROIVoxels: list of lists, indices of the voxels belonging
                        to each ROI. These indices refer to the rows of the 
                        voxelCoordinates array. len(ROIVoxels) = nROIs, len(ROIVoxels[i]) = size of ROI i.
    """
    ROIMaps = []
    ROIVoxels = []
    ROILabels = np.unique(voxelLabels)
    for ROILabel in ROILabels:
        if ROILabel < 0:
            continue
        else:
            voxels = list(np.where(voxelLabels == ROILabel)[0])
            ROIVoxels.append(voxels)
            if constructROIMaps:
                ROIMap = []
                for voxel in voxels:
                    if isinstance(voxelCoordinates,list):
                        ROIMap.append(voxelCoordinates[voxel])
                    else:
                        ROIMap.append(voxelCoordinates[voxel,:])
                ROIMaps.append(np.array(ROIMap))
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels}
    return ROIInfo

def voxelsInClustersToROIInfo(voxelsInClusters):
    """
    Transforms the voxelsInClusters structure (typical clustering output) to the
    ROIInfo dict (required input for many functions).
    
    Parameters:
    -----------
    voxelsInClusters: dict, where keys are ROI labels and values are lists of voxel
                      coordinates belonging to the ROIs     
                 
    Returns:
    --------
    ROIInfo: dict, contains:
             ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                      belonging to each ROI. len(ROIMaps) = nROIs.
             ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
                        to each ROI. These indices refer to the rows of the 
                        voxelCoordinates array. len(ROIVoxels) = nROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    """
    ROIMaps = []
    ROIVoxels = []
    voxelCoordinates = []
    ROILabels = voxelsInClusters.keys()
    voxelLabels = []
    counter = 0
    for ROILabel in ROILabels:
        if ROILabel < 0:
            continue
        else:
            voxelsInROI = voxelsInClusters[ROILabel]
            voxelCoordinates.extend(voxelsInROI)
            ROIMaps.append(np.array(voxelsInROI))
            ROIVoxels.append(range(counter,counter + len(voxelsInROI)))#(np.arange(counter,counter + len(voxelsInROI)))
            voxelLabels.extend([ROILabel]*len(voxelsInROI))
            counter = counter + len(voxelsInROI)
    voxelCoordinates = np.array(voxelCoordinates)
    voxelLabels = np.array(voxelLabels)
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels}
    return ROIInfo, voxelCoordinates, voxelLabels

def calculateTarget(nVoxels,sourceSize,sourceSum,sourceFlipSum,targetSize,targetSum,targetFlipSum,targetFunction='weighted mean consistency'):
    """
    Calculates the change in the target function caused by flipping one voxel from 
    source ROI to target ROI. The exact form of this change depends on the targetFunction.
    
    Note: if sourceSize in [0,1,2], this function returns 0 (these values yield division by zero).
    
    Parameters:
    -----------
    nVoxels: int, total number of voxels in all ROIs
    sourceSize: int, size of the source ROI
    sourceSum: double, sum of the correlation matrix between voxels of the source ROI, excluding diagonal
    sourceFlipSum: double, sum of the correlations between the voxel to be flipped and the rest of source ROI
    targetSize: int, size of the target ROI
    targetSum: double, sum of the correlation matrix between voxels of the target ROI, excluding diagonal
    targetFlipSum: double, sum of the correlations between the voxel to be flipped and the target ROI
    targetFunction: str, the overall function to be optimized in the flipping process, defines the exact
                    form of target. Options: 'weighted mean consistency' (default), 'mean consistency'
                    
    Returns:
    --------
    target: the change of the target function due to the flip
    """
    if sourceSize in [0,1,2]:
        target = 0
    else:
        target = 1./nVoxels*(1./targetSize*(-1*targetSum/(targetSize-1)+targetFlipSum)+1./(sourceSize-2)*(sourceSum/(sourceSize-1)-sourceFlipSum))
    return target

def calculateDistance(start,end):
    """
    Calculates the Euclidian distance between start and end.
    
    Parameters:
    -----------
    start, end: two iterables of three numeric values, coordinates of the start and end points
    
    Returns:
    --------
    distance: float, Euclidian distance between start and end
    """
    distance = np.sqrt(sum([(startC-endC)**2 for startC,endC in zip(start,end)]))
    return distance

def calculateThreshold(ROIVoxels, allVoxelTs, ROIIndex):
    """
    Calculates the consistency threshold data-driven: the consistency of a ROI
    (mean consistency between in-ROI voxels) can't be lower than the mean correlation
    of the in-ROI voxels to voxels of any other ROI.
    
    Parameters:
    -----------
    ROIVoxels: list of iterables, indices of the voxels belonging
               to each ROI. These indices refer to the rows of the voxelCoordinates array. len(ROIVoxels) = nROIs.
    allVoxelTs: nVoxels x nTime np.array containing voxel time series
    ROIIndex: int, index of the ROI to investigate
        
    Returns:
    --------
    threshold: float, the lowest mean correlation between the in-ROI voxels
               and the voxels of some other ROI
    """
    inROITs = allVoxelTs[ROIVoxels[ROIIndex],:]
    threshold = -1
    for i, ROI in enumerate(ROIVoxels):
        if i == ROIIndex:
            continue
        ROITs = allVoxelTs[ROI,:]
        ts = np.concatenate((inROITs,ROITs))
        correlations = np.corrcoef(ts)
        correlations = correlations[0:len(inROITs),len(inROITs)::]
        meanCorrelation = np.mean(correlations)
        if meanCorrelation > threshold:
            threshold = meanCorrelation
    return threshold

def calculateVoxelNeighborhoodCorrelation(voxelCoordinates,allVoxelTs):
    """
    Calculates the mean correlation between a voxel and its closest neighborhood
    (i.e. the six physically adjacent voxels).
    
    Parameters:
    -----------
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    allVoxelTs: nVoxels x nTime np.array containing voxel time series, needs to be in the same order 
                as voxelCoordinates
    
    Returns:
    --------
    voxelNeighborhoodCorrelations: double, mean correlation in voxel neighborhoods
    """
    voxelCoordinates = np.array(voxelCoordinates)
    correlations = []
    for voxel in voxelCoordinates:
        voxelIndex = np.where((voxelCoordinates==voxel).all(axis=1))[0][0]
        voxelTs = allVoxelTs[voxelIndex,:]
        neighbors = findNeighbors(voxel,allVoxels=voxelCoordinates)
        for neighbor in neighbors:
            neighborIndex = np.where((voxelCoordinates==neighbor).all(axis=1))[0][0]
            neighborTs = allVoxelTs[neighborIndex,:]
            correlations.append(pearsonr(voxelTs,neighborTs)[0])
    voxelNeighborhoodCorrelation = np.mean(correlations)
    return voxelNeighborhoodCorrelation
    
def makeLocalConnectivity(imdat, thresh):
    """
    Creates the local connectivity matrix used for the Craddock spectral ncut
    clustering method. This matrix contains for each voxel the Pearson correlation
    coefficient with the 27 closest neighbors.
    
    NOTE: This function makes strongly use of pyClusterROI (http://ccraddock.github.io/cluster_roi/)
    by Cameron Craddock et al.
    
    Parameters:
    -----------
    imdat: x*y*z*t np.array, fMRI measurement data to be used for the clustering.
         Three first dimensions correspond to voxel coordinates while the fourth is time.
         For voxels outside of the gray matter, all values must be set to 0.
    thresh: float, threshold value, correlation coefficients lower than this value
           will be removed from the matrix (set to zero).
    
    Returns:
    --------
    localConnectivity: 1D np.array that contains (in a concatenated form) the
                       x and y indices of connectivity values and the corresponding 
                       values. Can be used as an input for nCutClustering.
    """
    neighbors=np.array([[-1,-1,-1],[0,-1,-1],[1,-1,-1],
                     [-1, 0,-1],[0, 0,-1],[1, 0,-1],
                     [-1, 1,-1],[0, 1,-1],[1, 1,-1],
                     [-1,-1, 0],[0,-1, 0],[1,-1, 0],
                     [-1, 0, 0],[0, 0, 0],[1, 0, 0],
                     [-1, 1, 0],[0, 1, 0],[1, 1, 0],
                     [-1,-1, 1],[0,-1, 1],[1,-1, 1],
                     [-1, 0, 1],[0, 0, 1],[1, 0, 1],
                     [-1, 1, 1],[0, 1, 1],[1, 1, 1]])
                     
    # we need a gray matter mask; let's use a copy of the (already masked) imgdata                 
    mskdat = np.copy(imdat)[:,:,:,0]
    msz = mskdat.shape
    # convert the 3D mask array into a 1D vector
    mskdat=np.reshape(mskdat,np.prod(msz))
    # determine the 1D coordinates of the non-zero 
    # elements of the mask
    iv=np.nonzero(mskdat)[0]
    m=len(iv)
    print m, '# of non-zero voxels in the mask'
    
    # reshape fmri data to a num_voxels x num_timepoints array	
    sz = imdat.shape
    imdat=np.reshape(imdat,(np.prod(sz[:3]),sz[3]))
    
    # construct a sparse matrix from the mask
    msk=csc_matrix((range(1,m+1),(iv,np.zeros(m))),shape=(np.prod(sz[:-1]),1))
    sparse_i=[]
    sparse_j=[]
    sparse_w=[]

    negcount=0
    
    # loop over all of the voxels in the mask 	
    for i in range(0,m):
        if i % 1000 == 0: print 'voxel #', i
        # calculate the voxels that are in the 3D neighborhood
        # of the center voxel
        ndx3d=indx_1dto3d(iv[i],sz[:-1])+neighbors
        ndx1d=indx_3dto1d(ndx3d,sz[:-1])
        
        # restrict the neigborhood using the mask
        ondx1d=msk[ndx1d].todense()
        ndx1d=ndx1d[np.nonzero(ondx1d)[0]]
        ndx1d=ndx1d.flatten()
        ondx1d=np.array(ondx1d[np.nonzero(ondx1d)[0]])
        ondx1d=ondx1d.flatten()

        # determine the index of the seed voxel in the neighborhood
        nndx=np.nonzero(ndx1d==iv[i])[0]

        # exctract the timecourses for all of the voxels in the 
        # neighborhood
        tc=matrix(imdat[ndx1d,:])
	 
        # make sure that the "seed" has variance, if not just
        # skip it
        if np.var(tc[nndx,:]) == 0:
            continue

        # calculate the correlation between all of the voxel TCs
        R=np.corrcoef(tc)
        if np.rank(R) == 0:
            R=np.reshape(R,(1,1))

        # extract just the correlations with the seed TC
        R=R[nndx,:].flatten()

        # set NaN values to 0
        R[np.isnan(R)]=0
        negcount=negcount+sum(R<0)

        # set values below thresh to 0
        R[R<thresh]=0

        # determine the non-zero correlations (matrix weights)
        # and add their indices and values to the list 
        nzndx=np.nonzero(R)[0]
        if(len(nzndx)>0):
            sparse_i=np.append(sparse_i,ondx1d[nzndx]-1,0)
            sparse_j=np.append(sparse_j,(ondx1d[nndx]-1)*np.ones(len(nzndx)))
            sparse_w=np.append(sparse_w,R[nzndx],0) # The axis here used to be 1, which raided an error, so I changed it to 0
            
    # concatenate the i, j and w_ij into a single vector	
    localConnectivity=sparse_i
    localConnectivity=np.append(localConnectivity,sparse_j)
    localConnectivity=np.append(localConnectivity,sparse_w)
    
    return localConnectivity

def nCutClustering(connectivity,k):
    """
    Performs the spectral ncut clustering described in Craddock et al. (2013) at
    the level of a single local connectivity matrix.
    NOTE: This function makes strongly use of pyClusterROI (http://ccraddock.github.io/cluster_roi/)
    by Cameron Craddock et al.
    NOTE2: This function requires the python_ncut_lib distributed with pyClusterROI
    saved in the location accessible through PYTHONPATH.
    
    Parameters:
    -----------
    connectivity: 1D np.array that contains (in a concatenated form) the x and y indices of 
                  connectivity values and the corresponding values.
    k: int, number of clusters to generate. Note that this is the aimed number of clusters,
       and the actual number of produced clusters may be slightly smaller.
    
    Returns:
    --------
    group_img: 1D np.array, the value of each elements shows the index of the
               cluster to which the corresponding voxel belongs
    """
    import python_ncut_lib as ncut
    # calculate the number of non-zero weights in the connectivity matrix
    n=len(connectivity)/3
    # reshape the 1D vector read in from infile in to a 3xN array
    a=np.reshape(connectivity,(3,n))
    m=max(max(a[0,:]),max(a[1,:]))+1
    # make the sparse matrix, CSC format is supposedly efficient for matrix
    # arithmetic
    W=csc_matrix((a[2,:],(a[0,:],a[1,:])), shape=(m,m))
    
    # calculating the eigendecomposition of the Laplacian 
    eigenval,eigenvec = ncut.ncut(W,k)
    
    # clustering
    eigk=eigenvec[:,:k]
    eigenvec_discrete = ncut.discretisation(eigk)

    # transform the discretised eigenvectors into a single vector
    # where the value corresponds to the cluster # of the corresponding
    # voxel (each voxel belongs to excactly one cluster)
    group_img=eigenvec_discrete[:,0]
    for i in range(1,k):
        group_img=group_img+(i+1)*eigenvec_discrete[:,i]
        
    return group_img

def getKendallW(timeSeries):
    """
    Calculates the Kendall's coefficience of concordance (Kendall's W; Kendall & Gibbons 1990) 
    for a bunch of time series.
    
    Parameters:
    -----------
    timeSeries: np.array (voxels x time), time series of the voxels
    
    Returns:
    --------
    W: float, Kendall's W
    """
    K, n = timeSeries.shape
    rankSum = np.zeros(n)
    for ts in timeSeries:
        temp = ts.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(ts))
        rankSum = rankSum + ranks
    meanRankSum = np.mean(rankSum)
    W = 12*(np.sum(rankSum**2)-n*meanRankSum**2)/(K**2*(n**3-n))
    return W
    
  
def calculateReHo(imgdata,voxelCoords,nNeighbors=6,resolution=1,allVoxels=[]):
    """
    A function for calculating the Regional Homogeneity (Zang et al. 2004; NeuroImage)
    of a voxel (ReHo is defined as the Kendall's coefficient of condolence in voxel's
    neighborhood).
    
    Parameters:
    -----------
    imgdata: x*y*z*t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
    voxelCoords: 1x3 np.array, coordinates of a voxel (either in voxels or in mm)
    nNeighbors: int, number or neighbors used for calculating ReHo; options: 6 (faces),
                18 (faces + edges), 26 (faces + edges + corners) (default = 6)
    resolution: double, distance between voxels if coordinates are given in mm;
                if coordinates are given in voxels, use the default value 1 (voxels
                are 1 voxel away from each other).
    allVoxels: iterable, coordinates of all acceptable voxels. If allVoxels is given,
               only neighbors in allVoxels are returned (default: []).
               
    Returns:
    --------
    ReHo: float: Regional Homogeneity of the voxel 
    
    """
    assert nNeighbors in [6,18,26], "Bad number of neigbors; select either 6 (faces), 18 (faces + edges) or 26 (faces + edges + corners)"
    neighbors = findNeighbors(voxelCoords,nNeighbors,resolution,allVoxels)
    neighborTs = imgdata[neighbors,:]
    ReHo = getKendallW(neighborTs)
    return ReHo
    
def updateQueue(ROIIndex, priorityQueue, targetFunction, centroidTs, allVoxelTs, ROIVoxels,
                consistencies=[], ROISizes = [], consistencyType='pearson c',fTransform=False):
    """
    For the given priority queue of voxels, calculates the priority value of each 
    element and finds the element associated with the highest priority value. This
    is a helper function for growOptimizedROIs.
    
    Parameters:
    -----------
    ROIIndex: int, index of the ROI whose priority queue will be updated
    priorityQueue: list of ints, elements of the queue are indices that refer to
                   rows of allVoxelTs.
    targetFunction: str, the priority value function. Options: 'correlationWithCentroid',
                    'spatialConsistency', 'weighted mean consistency', 'local weighted consistency'
    centroidTs: nTimepoints np.array, time series of the ROI centroid
    allVoxelTs: nVoxels x nTimepoints np.array, time series of all voxels
    ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
               to each ROI. These indices refer to the rows of the 
               voxelCoordinates array. len(ROIVoxels) = nROIs
    consistencies: iterable (len(consistencies) = nROIs), spatial consistencies of all ROIs. Used if
                   targetFunction == 'weighted mean consistency' or 'local weighted consistency'
    ROISizes: iterable (len(consistencies = nROIs)), sizes of all ROIs. Used if targetFunction ==
              'weighted mean consistency' or 'local weighted consistency'
    consistencyType: str, definition of spatial consistency to be used if 
                     targetFunction == 'spatialConsistency' (default: 'pearson c' (mean Pearson correlation coefficient))
    fTransform: bool, should Fisher Z transform be applied if targetFunction == 'spatialConsistency' 
                (default=False)
    
    Returns:
    --------
    additionCandidate: the queue element with the highest priority value
    maximalMeasure: the priority value associated with additionCandidate
    """
    if targetFunction == 'correlationWithCentroid':
        priorityMeasures = [np.corrcoef(centroidTs,allVoxelTs[priorityIndex])[0][1] for priorityIndex in priorityQueue]
    elif targetFunction == 'spatialConsistency':    
        priorityMeasures = np.zeros(len(priorityQueue))
        print('len priority queue: ' + str(len(priorityQueue)))
        for j, voxel in enumerate(priorityQueue):
            voxelIndices = np.concatenate((ROIVoxels,np.array([voxel])))
            priorityMeasures[j] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices))
    elif targetFunction == 'weighted mean consistency':
        priorityMeasures = np.zeros(len(priorityQueue))
        print('len priority queue: ' + str(len(priorityQueue)))
        tempConsistencies = list(consistencies)
        tempSizes = list(ROISizes)
        tempSizes[ROIIndex] = tempSizes[ROIIndex] + 1
        for j, voxel in enumerate(priorityQueue):
            voxelIndices = np.concatenate((ROIVoxels[ROIIndex],np.array([voxel])))
            tempConsistencies[ROIIndex] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices))
            priorityMeasures[j] = sum([tempConsistency*tempSize for tempConsistency,tempSize in zip(tempConsistencies,tempSizes)])/sum(tempSizes) 
    elif targetFunction == 'local weighted consistency':
        priorityMeasures = np.zeros(len(priorityQueue))
        print('len priority queue: ' + str(len(priorityQueue)))
        print('ROI to update: ' + str(ROIIndex))
        for j, voxel in enumerate(priorityQueue):
            voxelIndices = np.concatenate((ROIVoxels,np.array([voxel])))
            priorityMeasures[j] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices)) * (ROISizes[ROIIndex] + 1)
    additionCandidate = priorityQueue[np.argmax(priorityMeasures)]
    maximalMeasure = np.amax(priorityMeasures)
    
    return additionCandidate, maximalMeasure
        
    

def checkFlipValidity(flip,ROIVoxels,voxelCoordinates,voxelLabels):
    """
    Checks if the given flip is valid, i.e. the given voxel can be flipped away
    from the source ROI without breaking the ROI into multiple distinct parts.
    In practise, after the flip each source-ROI neighbor of the flipped voxel
    must be connected to each other by a path containing source-ROI voxels only.
    To check the existance of the paths, the A* algorithm is used.
    
    Parameters:
    -----------
    flip: set, containing:
          voxel: int, index of the voxel to be flipped; this index refers to voxelCoordinates
          sourceROI: int, index of the ROI from which to flip; this index refers to ROIVoxels
          targetROI: int, index of the ROI to which to flip; this index refers to ROIVoxels
    ROIVoxels: list of iterables, indices of the voxels belonging
               to each ROI. These indices refer to the rows of the voxelCoordinates array. len(ROIVoxels) = nROIs.
    voxelCoordinates: nVoxels x 3 np.array, coordinates (in voxels) of all voxels
    voxelLabels: iterable, ROI labels (i.e. indices of ROIs where the 
                 voxels belong) of all voxels; voxels not belongin to any ROI have
                 label -1.
    
    Returns:
    --------
    valid: boolean, if True, the flip is valid
    """
    voxelIndex,sourceROI,_ = flip
    sourceVoxels = ROIVoxels[sourceROI]
    neighbors = findInROINeighbors(voxelIndex,sourceROI,voxelCoordinates,voxelLabels)
    if len(neighbors) == 1:
        valid = True
    else:
        valid = [False] * (len(neighbors) - 1)
        for i in range(len(neighbors)-1):
            if valid[i]:
                continue
            start = neighbors[i]
            end = neighbors[i+1]
            # the A* algorithm:
            closedSet = []
            openSet = [start]
            cameFrom = {}
            gScore = {sourceVoxel:np.Infinity for sourceVoxel in sourceVoxels}
            gScore[start] = 0
            fScore = []
            [heapq.heappush(fScore,(np.Infinity,sourceVoxel)) for sourceVoxel in sourceVoxels]
            heapq.heappush(fScore,(calculateDistance(voxelCoordinates[start],voxelCoordinates[end]),start))
            totalPath = []
            while len(openSet) > 0:
                currentFScore,currentVoxel = heapq.heappop(fScore)
                if not currentVoxel in openSet:
                    continue
                if currentVoxel == end:
                    totalPath.append(currentVoxel)
                    while currentVoxel in cameFrom.keys():
                        currentVoxel = cameFrom[currentVoxel]
                        totalPath.append(currentVoxel)
                    break
                openSet.remove(currentVoxel)
                closedSet.append(currentVoxel)
                currentNeighbors = findInROINeighbors(currentVoxel,sourceROI,voxelCoordinates,voxelLabels)
                if voxelIndex in currentNeighbors:
                    currentNeighbors.remove(voxelIndex) # The path can't go through the flipped voxel
                for currentNeighbor in currentNeighbors:
                    if currentNeighbor in closedSet:
                        continue
                    tentativeGScore = gScore[currentVoxel] + 1 # the distance between any voxel and its neighbors is 1
                    if currentNeighbor not in openSet:
                        openSet.append(currentNeighbor)
                    elif tentativeGScore > gScore[currentNeighbor]:
                        continue
                    cameFrom[currentNeighbor] = currentVoxel
                    gScore[currentNeighbor] = tentativeGScore
                    heapq.heappush(fScore,(calculateDistance(voxelCoordinates[currentNeighbor],voxelCoordinates[end]),currentNeighbor)) # NOTE: this doesn't remove the old element from the fScore, only add the same element again with a different (possibly smaller) score. Does this work?
        
            if len(totalPath) > 0:
                valid[i] = True
                # TODO: check what to do with the following lines: the idea would be that if a neighbor is already a member of this path, its paths doesn't need to be checked separately but how to do this efficiently/logically?
#                for k in range(i+2,len(neighbors)):
#                    if neighbors[k] in totalPath:
#                        valid[k] = True
        valid = all(valid)
    return valid

def getInAndBwROIMasks(inROIIndices):
    """
    Returns the masks for picking correlations within and between rois from
    a voxel-voxel correlation matrix where voxels are ordered by their ROI identity
    
    Parameters:
    -----------
    inROIIndices: list of arrays; indices of voxels belonging to each ROI
    
    Returns:
    --------
    withinROIMask, betweenROIMask: nROIs x nROIs arrays where elements a) inside a ROI
                                   or b) between ROIs are 1, others 0
    """
    inROIxIndices = []
    inROIyIndices = []
    nVoxels = 0
    offset = 0
    for ROI in inROIIndices:
        s = len(ROI)
        nVoxels = nVoxels + s
        template = np.zeros((s, s))
        triu = np.triu_indices_from(template, 1)
        inROIxIndices.extend(triu[0] + offset)
        inROIyIndices.extend(triu[1] + offset)
        offset = offset + s
    withinROIMask = np.zeros((nVoxels,nVoxels))
    withinROIMask[inROIxIndices, inROIyIndices] = 1
    betweenROIMask = np.ones((nVoxels,nVoxels))
    betweenROIMask = np.triu(betweenROIMask - withinROIMask - np.eye(nVoxels))
    return (withinROIMask, betweenROIMask)
            
# Consistency calculation
    
def calculateSpatialConsistency(params):
    """
    Calculates the spatial consistency of a chunk of voxels. By default,
    spatial consistency is defined as the mean Pearson correlation coefficient
    between the voxel time series.
    
    Parameters:
    -----------
    params: tuple, containing:
    
        cfg: dict, containing:           
            allVoxelTs: nVoxels x nTime np.array containing voxel time series
            consistencyType: str, definition of spatial consistency to be used; default:
                  'pearson c' (mean Pearson correlation coefficient)
            fTransform: bool, are the correlations Fisher f transformed before averaging
                        when consistencyType = 'pearson c' (default=False)        
        voxelIndices: np.array or other iterable, indices of voxels; these indices should refer to voxels' 
                      locations in the voxel time series array; note that the chunk
                      must contain at least one voxel
          
    Returns:
    --------
    spatialConsistency: dbl, value of spatial consistency
    """
    cfg = params[0]
    allVoxelTs = cfg['allVoxelTs']
    if 'consistencyType' in cfg.keys():
        consistencyType = cfg['consistencyType']
    else:
        consistencyType = 'pearson c'
    if 'fTransform' in cfg.keys():
        fTransform = cfg['fTransform']
    else:
        fTransform = False
    voxelIndices = params[1]
    
    assert len(voxelIndices) > 0, "Detected an empty ROI, cannot calculate consistency"

    if len(voxelIndices) == 1:
        spatialConsistency = 1. # a single voxel is always fully consistent
    else: 
        voxelTs = allVoxelTs[voxelIndices,:]
        if consistencyType == 'pearson c':
            correlations = np.corrcoef(voxelTs)
            correlations = correlations[np.tril_indices(voxelTs.shape[0],k=-1)] # keeping only the lower triangle, diagonal is discarded
            if fTransform:
                correlations = np.arctanh(correlations)
                spatialConsistency = np.tanh(np.mean(correlations))
            else:
                spatialConsistency = np.mean(correlations)
        else:
            spatialConsistency = 0
    if np.isnan(spatialConsistency):
        print('nan detected!')
    return spatialConsistency
    
def calculateSpatialConsistencyInParallel(voxelIndices,allVoxelTs,consistencyType='pearson c', fTransform=False, nCPUs=5):
    """
    A wrapper function for calculating the spatial consistency in parallel across
    ROIs.
    
    Parameters:
    -----------
    voxelIndices: list of np.arrays, each array containing indices of voxels of one ROI; 
               these indices should refer to voxels' 
               locations in the allVoxelTs parameter; note that the chunk
               must contain at least one voxel
    allVoxelTs: structured np.array with a field name 'roi_voxel_ts' (and possible additional 
                fields), this field contains voxel time series
    consistencyType: str, definition of spatial consistency to be used; default:
          'pearson c' (mean Pearson correlation coefficient), other options:
    fTransform: bool, are the correlations Fisher f transformed before averaging
                when consistencyType = 'pearson c' (default=False)
    nCPUs = int, number of CPUs to be used for the parallel computing (default = 5)
    
    
    Returns:
    --------
    spatialConsistencies: list of doubles, spatial consistencies of the ROIs defined
                          by voxelIndices
    """
    cfg = {'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform}
    if True:
        paramSpace = [(cfg,voxelInd) for voxelInd in voxelIndices]
        pool = Pool(max_workers = nCPUs)
        spatialConsistencies = list(pool.map(calculateSpatialConsistency,paramSpace,chunksize=1))
    else: # this is a debugging case
        spatialConsistencies = np.zeros(len(voxelIndices))
        for i, voxelInd in enumerate(voxelIndices):
            spatialConsistencies[i] = calculateSpatialConsistency((cfg,voxelInd))

    return spatialConsistencies

def calculateSpatialConsistencyPostHoc(dataFiles,layersetwiseNetworkSavefolders,networkFiles,
                                       nLayers,timewindow,overlap,consistencyType='pearson c',fTransform=False,nCPUs=5,
                                       savePath=None,subjectIndex=None):
    """
    Calculates spatial consistency and size of earlier-created ROIs. The ROIs should be
    saved by pipeline.isomorphism_classes_from_file; they cant be
    e.g. an outcome of consistency optimization or random chunks of voxels. 
    
    Parameters:
    -----------
    dataFiles: lists of strs, paths to the .nii files used for calculating spatial
                consistency.
    layersetwiseNetworksSavefolders: list of strs, paths to the folders where
                                       networks created by pipeline.isomorphism_classes_from_file
                                       (and related ROI information) have been saved. Must be of
                                       the same length as data_files.
    networkFiles: list of strs, names of all network files saved in layersetwise_networks_savefolder
                   (should be the same for all savefolders)
    nLayers: int, number of layers used for constructing networks in isomorphism_classes_from file
              (used for reading data)
    timewindow: int, length of time window used in isomorphism_classes_from_file
    overlap: int, time window overlap used in isomorphism_classes_from_file
    consistencyType: str, definition of spatial consistency to be used; default:
                     'pearson c' (mean Pearson correlation coefficient), other options:
    fTransform: bool, are the correlations Fisher f transformed before averaging
                when consistencyType = 'pearson c' (default=False)
    nCPUs: int, number of CPUs to be used for the parallel computing (default = 5)
    savePath: str, path to which save the calculated consistencies (default = None, no saving)
    subjectIndex: int, index of the subject to be analyzed. If subject index is not None, only
                  the subjectIndex-th dataFile and layersetwiseNetworkSaveFolder will be used.
                  (default = None)
    
    Returns:
    --------
    spatialConsistency_data: dict, contains:
                              'data_files':data_files
                              'layersetwise_network_savefolders':layersetwise_network_savefolders
                              'network_files':network_files
                              'n_layers': n_layers
                              'consistency_type': consistency_type
                              'f_transform': f_transform
                              'timewindow': timewindow
                              'overlap': overlap
                              'spatial_consistencies': list of floats, spatial consistencies of all ROIs 
                                                       (len = len(layersetwise_network_savefolders)*len(network_files)*n_timewindows*n_ROIs)
                              'roi_sizes': list of ints, sizes of all ROIs
    """
    spatialConsistencies = []
    roiSizes = []
    if not subjectIndex == None:
        dataFiles = [dataFiles[subjectIndex]]
        layersetwiseNetworkSavefolders = [layersetwiseNetworkSavefolders[subjectIndex]]
    # looping over network_savefolders (can be over subjects but also over a single subject in multiple runs)
    for dataFile, layersetwiseNetworkSavefolder in zip(dataFiles,layersetwiseNetworkSavefolders):
        # reading data; later on, this will be used to calculate consistencies
        img = nib.load(dataFile) 
        imgdata = img.get_data()
        nTime = imgdata.shape[-1]
        # finding end and start points of time windows that correspond to layers (consistency will be calculated inside windows)
        k = network_construction.get_number_of_layers(imgdata.shape,timewindow,overlap)
        startTimes,endTimes = network_construction.get_start_and_end_times(k,timewindow,overlap)
        layerIndex = 0
        for networkFile in networkFiles:
            _,voxelCoordinates = readVoxelIndices(layersetwiseNetworkSavefolder+'/'+networkFile,layers='all')
            for voxelCoordinatesPerLayer, startTime, endTime in zip(voxelCoordinates, startTimes[layerIndex:layerIndex+nLayers], endTimes[layerIndex:layerIndex+nLayers]):
                roiSizesPerLayer = [len(voxelCoords) for voxelCoords in voxelCoordinatesPerLayer]
                roiSizes.extend(roiSizesPerLayer)
                nVoxels = sum(roiSizesPerLayer)
                allVoxelTs = np.zeros((nVoxels,nTime))
                voxelIndices = []
                offset = 0
                for ROI in voxelCoordinatesPerLayer:
                    s = len(ROI)
                    for i, voxel in enumerate(ROI):
                        allVoxelTs[offset+i,:]=imgdata[voxel[0],voxel[1],voxel[2],:]
                    voxelIndices.append(np.arange(offset,offset+s))
                    offset += s
                spatialConsistencies.extend(calculateSpatialConsistencyInParallel(voxelIndices,allVoxelTs[:,startTime:endTime],consistencyType,fTransform,nCPUs))
            layerIndex += nLayers
    spatialConsistencyData = {'data_files':dataFiles,'layersetwise_network_savefolders':layersetwiseNetworkSavefolders,
                                'network_files':networkFiles,'n_layers':nLayers,'consistency_type':consistencyType,
                                'f_transform':fTransform,'timewindow':timewindow,'overlap':overlap,'spatial_consistencies':spatialConsistencies,
                                'roi_sizes':roiSizes}
    if not savePath==None:
        with open(savePath, 'wb') as f:
            pickle.dump(spatialConsistencyData, f, -1)
            
    return spatialConsistencyData

def calculateCorrelationsInAndBetweenROIs(dataFiles,layersetwiseNetworkSavefolders,
                                      networkFiles,nLayers,timewindow,overlap,savePath=None,
                                      nBins=100,returnCorrelations=False,subjectIndex=None,
                                      normalizeDistributions=True):
    """
    Starting from ROIs saved earlier by pipeline.isomorphism_classes_from_file,
    calculates the Pearson correlation coefficients between voxels in the same ROI
    and in different ROIs and their distribution. The distributions are calculated
    using nBins equal-sized bins ranging from -1 to 1.
    
    Parameters:
    -----------
    dataFiles: lists of strs, paths to the .nii files used for calculating the correlations
    layersetwiseNetworksSavefolders: list of strs, paths to the folders where
                                       networks created by pipeline.isomorphism_classes_from_file
                                       (and related ROI information) have been saved. Must be of
                                       the same length as data_files.
    networkFiles: list of strs, names of all network files saved in layersetwise_networks_savefolder
                   (should be the same for all savefolders)
    nLayers: int, number of layers used for constructing networks in isomorphism_classes_from file
              (used for reading data)
    timewindow: int, length of time window used in isomorphism_classes_from_file
    overlap: int, time window overlap used in isomorphism_classes_from_file
    savePath: str, path to which save the correlations (default = None, no saving)
    nBins: int, number of bins used to calculate the distribution (default = 100)
    returnCorrelations: boolean, if True, lists of all in-ROI and between-ROI correlations
                        are returned in addition to the distribution; note that these lists
                        are in most cases extremely long (default = False)
    subjectIndex: int, index of the subject to be analyzed. If subject index is not None, only
                  the subjectIndex-th dataFile and layersetwiseNetworkSaveFolder will be used.
                  (default = None)
    normalizeDistributions: bool, if set to False, the number of samples in each bin is calculated 
                            instead of normalized PDFs. (default = True)

    Returns:
    --------
    correlationData: dict, contains:
                              'dataFiles':dataFiles
                              'layersetwiseNetworkSavefolders':layersetwiseNetworkSavefolders
                              'networkFiles':networkFiles
                              'nLayers': nLayers
                              'timewindow': timewindow
                              'overlap': overlap
                              'inROICorrelations': list of floats, correlation values between voxels in the same ROI
                              'betweenROICorrelations': list of floats, correlation values between voxels in different ROIs
                              'inROIDistribution': np.array of floats, distribution of correlation values between voxels in the same ROI
                              'betweenROIDistribution': np.array of floats, distribution of correlation values between voxels in different ROIs
                              'binCenters': np.array of floats, centers of bins (same bins used both in-ROI and between-ROI distributions)
    """
    inROICorrelations = []
    betweenROICorrelations = []
    if not subjectIndex == None:
        dataFiles = [dataFiles[subjectIndex]]
        layersetwiseNetworkSavefolders = [layersetwiseNetworkSavefolders[subjectIndex]]
    # looping over network_savefolders (can be over subjects but also over a single subject in multiple runs)
    initializeDistribution = True
    for dataFile, layersetwiseNetworkSavefolder in zip(dataFiles,layersetwiseNetworkSavefolders):
        # reading data; later on, this will be used to calculate consistencies
        img = nib.load(dataFile) 
        imgdata = img.get_data()
        nTime = imgdata.shape[-1]
        # finding end and start points of time windows that correspond to layers (correlations will be calculated inside windows)
        k = network_construction.get_number_of_layers(imgdata.shape,timewindow,overlap)
        startTimes,endTimes = network_construction.get_start_and_end_times(k,timewindow,overlap)
        layerIndex = 0
        for networkFile in networkFiles:
            _,voxelCoordinates = readVoxelIndices(layersetwiseNetworkSavefolder+'/'+networkFile,layers='all')
            for voxelCoordinatesPerLayer, startTime, endTime in zip(voxelCoordinates, startTimes[layerIndex:layerIndex+nLayers], endTimes[layerIndex:layerIndex+nLayers]):
                nVoxels = sum([len(ROI) for ROI in voxelCoordinatesPerLayer])
                allVoxelTs = np.zeros((nVoxels,nTime))
                voxelIndices = []
                offset = 0
                for ROI in voxelCoordinatesPerLayer:
                    s = len(ROI)
                    for i, voxel in enumerate(ROI):
                        allVoxelTs[offset+i,:]=imgdata[voxel[0],voxel[1],voxel[2],:]
                    voxelIndices.append(np.arange(offset,offset+s))
                    offset += s
                allVoxelCorrelations = np.corrcoef(allVoxelTs[:,startTime:endTime])
                inROIMask, betweenROIMask = getInAndBwROIMasks(voxelIndices)
                inCorrs = allVoxelCorrelations[np.where(inROIMask > 0)]
                betweenCorrs = allVoxelCorrelations[np.where(betweenROIMask > 0)]
                # calculating the sum of observations in each bin
                if initializeDistribution:
                    bins = np.arange(-1.,1,2.0/nBins) # Correlation is always limited between -1 and 1 so we can hardcode the boundaries
                    bins = np.concatenate((bins, np.array([1.0])))
                    inROIDistribution, binEdges,_ = binned_statistic(inCorrs,inCorrs,statistic='count',bins=bins)
                    betweenROIDistribution,_,_ = binned_statistic(betweenCorrs,betweenCorrs,statistic='count',bins=bins)
                    initializeDistribution = False
                else:
                    inROIDistribution = inROIDistribution + binned_statistic(inCorrs,inCorrs,statistic='count',bins=bins)[0]
                    betweenROIDistribution = betweenROIDistribution + binned_statistic(betweenCorrs,betweenCorrs,statistic='count',bins=bins)[0]
                if returnCorrelations:
                    inROICorrelations.extend(inCorrs)
                    betweenROICorrelations.extend(betweenCorrs)
            layerIndex += nLayers
    if normalizeDistributions:
        # normalizing distributions
        inROIDistribution = inROIDistribution/float(np.sum(inROIDistribution*np.abs(binEdges[0]-binEdges[1])))        
        betweenROIDistribution = betweenROIDistribution/float(np.sum(betweenROIDistribution*np.abs(binEdges[0]-binEdges[1])))
    binCenters = 0.5*(binEdges[:-1]+binEdges[1:])            
    correlationData = {'dataFiles':dataFiles,'layersetwiseNetworkSavefolders':layersetwiseNetworkSavefolders,
                                'networkFiles':networkFiles,'nLayers':nLayers,'timewindow':timewindow,
                                'overlap':overlap,'inROICorrelations':inROICorrelations,
                                'betweenROICorrelations':betweenROICorrelations,'inROIDistribution':inROIDistribution,
                                'betweenROIDistribution':betweenROIDistribution,'binCenters':binCenters}
    if not savePath==None:
        with open(savePath, 'wb') as f:
            pickle.dump(correlationData, f, -1)
            
    return correlationData
        
# ROI construction without optimization
    
def growSphericalROIs(ROICentroids, imgdata, nROIs=246, template=None, equalSized=False):
    """
    Constructs a set of (approximately) spherical ROIs with a given radius located
    around a given set of centroid points.
    
    Parameters:
    -----------
    ROICentroids: nROIs x 3 np.array, coordinates of the centroids of the ROIs.
                  This can be a ROI centroid from an atlas but also any other
                  (arbitrary) point. Set to 'random' to use random seeds.
    imgdata: x*y*z*t np.array, fMRI measurement data to be used for the clustering.
             Three first dimensions correspond to voxel coordinates while the fourth is time.
             For voxels outside of the gray matter, all values must be set to 0.
    nROIs: int, number of ROIs. Used to define the number of ROIs if ROICentroids = 'random'.
           default = 246 (number of ROIs in the Brainnetome parcellation)
    template: 3D numpy array where each element corresponds to a voxel. The value of each voxel is the index of the ROI the voxel belongs to.
              Voxels outside of ROIs (= outside of the gray matter) have value 0. Template is used only if ROICentroids == 'random' (default = None)
    equalSized: boolean, if equalSized == True, all grown ROIs will be of the same size. Note that this option will leave
                some voxels outside of ROIs: growing will stop as soon as one of the ROIs doesn't have space to grow.
                            
    Returns:
    --------
    voxelLabels: nVoxels x 1 np.array, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    """
    if ROICentroids != 'random':
        nROIs = ROICentroids.shape[0]
    
    if ROICentroids == 'random':
        ROICentroids = getRandomCentroids(nROIs,template)
        
    voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    voxelLabels = np.zeros(len(voxelCoordinates)) - 1
    
    distanceMatrix = getDistanceMatrix(ROICentroids, voxelCoordinates) # distanceMatrix = nROIs x nVoxel
    distanceMask = np.ones(distanceMatrix.shape)
        
    ROIVoxels = []
    for ROIIndex, centroid in enumerate(ROICentroids):
        centroidIndex = np.where((voxelCoordinates==centroid).all(axis=1)==1)[0]
        ROIVoxels.append(np.array(centroidIndex))
        voxelLabels[centroidIndex] = ROIIndex
        distanceMask[:,centroidIndex] = 0
    
    ROIMaps = [np.array(centroid) for centroid in ROICentroids]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels,'ROISizes':np.ones(nROIs,dtype=int),'ROINames':[]}
    ROINeighbors = np.ones(nROIs) # this is a boolean array telling if a ROI still has neighbors that can be added
    
    if equalSized:
        centroidDistances = getDistanceMatrix(ROICentroids,ROICentroids)[np.triu_indices(nROIs,1)]
        maxRadius = np.amin(centroidDistances)/2
    else:
        maxRadius = len(voxelCoordinates)
    
    radius = 1
    
    while np.sum(ROINeighbors) > 0 and radius < maxRadius:
        print('Growing ROIs, radius: ' + str(radius) + ', ' + str(np.sum(ROINeighbors)) + ' ROIs able to grow')
        for ROIIndex, distances in enumerate(distanceMatrix):
            distances = distances * distanceMask[ROIIndex,:]
            if ROINeighbors[ROIIndex] == 0:
                continue
            distanceNeighbors = np.where(((0 < distances) & (distances <= radius)))[0]
            physicalNeighbors = set(findROIlessNeighbors(ROIIndex,voxelCoordinates,{'ROIMaps':ROIMaps})['ROIlessIndices'])
            neighbors = [neighbor for neighbor in distanceNeighbors if neighbor in physicalNeighbors]
            if len(neighbors) == 0:
                ROINeighbors[ROIIndex] = 0
                continue
            for neighbor in neighbors:
                ROIInfo = addVoxel(ROIIndex,neighbor,ROIInfo,voxelCoordinates)
                voxelLabels[neighbor] = ROIIndex
                distanceMask[:,neighbor] = 0 # masking away voxels that have already been added to a ROI
        radius = radius + 1
    voxelLabels = np.array([int(label) for label in voxelLabels])
    #import pdb; pdb.set_trace()            
    return voxelLabels, voxelCoordinates, radius
    
# Optimization

def growOptimizedROIs(cfg,verbal=True):
    """
    Starting from given centroids, grows a set of ROIs optimized in terms of
    spatial consistency. Optimization is based on a priority queue system: at each
    step, a measure is calculated for each centroid-voxel pair and the voxel
    with the highest measure value is added to the ROI in question.
    
    Possible measures are:
        1) the correlation between the ROI centroid time series and voxel time series
        2) the spatial consistency (mean Pearson correlation coefficient) of the voxels
           already in the ROI and the candidate voxel
    
    Parameters:
    -----------
    cfg: dict, contains:
         ROICentroids: nROIs x 3 np.array, coordinates of the centroids of the ROIs.
                  This can be a ROI centroid from an atlas but also any other
                  (arbitrary) point. Set ROICentroids to 'random' to use random seeds.
         names: list of strs, names of the ROIs, can be e.g. the anatomical name associated with
                  the centroid. Default = ''.
         imgdata: x*y*z*t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
         threshold: float or string. The lowest centroid-voxel correlation that leads to adding a voxel. All
                    thresholding approaches may lead to parcellation where some voxels don't belong to any ROI.
                    For no thresholding, use threshold=-1. (default=-1)
                    options:
                    - float: no voxel is added to a ROI if the measure (see above) < threshold
                    - 'data-driven': no voxel is added to a ROI if after adding the voxel the mean correlation of the 
                    voxels of this ROI is lower than the mean correlation between the voxels of this ROI and the voxels
                    of any other ROI. However, the same voxel can be considered again later.
                    - 'strict data-driven': similar as above but when a voxel has been found to be sub-threshold
                    for a ROI, it is entirely removed from the base of possible voxels for this ROI
                    - 'voxel-wise': no voxel is added to a ROI if its average correlation to the voxels of this
                    ROI is lower than its average correlation to some other ROI
                    - 'maximal-voxel-wise': same as voxel-wise above but only the N strongest correlations per ROI
                    are taken into account; for setting N see the nCorrelatiosnForThresholding parameter below
                    - 'voxel-neighbor': no voxel is added to a ROI if its average correlation to the voxels of this
                    ROI is lower than the average correlation of a voxel to its closest (6-voxel) neighborhood. This
                    threshold value is calculated as an average across all voxels before starting to build the ROIs.
         targetFunction: str, measure that will be optimized (options: correlationWithCentroid, spatialConsistency, weighted
                         mean consistency, local weighted consistency)
         consistencyType: str, definition of spatial consistency to be used if 
                          targetFunction == 'spatialConsistency' (default: 'pearson c' (mean Pearson correlation coefficient))
         fTransform: bool, should Fisher Z transform be applied if targetFunction == 'spatialConsistency' 
                     (default=False)
         template: 3D numpy array where each element corresponds to a voxel. The value of voxels included in the analysis should
                   be >0 (e.g. the index of the ROI the voxel belongs to). Voxels outside of ROIs (= outside of the gray matter) 
                   have value 0. Template is used only if cfg[ROICentroids] == 'random' (default = None)
         nROIs: int, number of ROIs. Only used if cfg[ROICentroids] == 'random' (default = 100)
         verbal: bool, if verbal == True, more progress information is printed (default = True)
         nCorrelationsForThresholding: int, the number of strongest correlations considered if threshold == 'maximal-voxel-wise'
                                       (default = 5)
    
    Returns:
    --------
    voxelLabels: nVoxels x 1 np.array, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    meanConsistency: double, mean consistency of the final ROIs
    """
    # Setting up: reading parameters
    if not 'template' in cfg.keys():
        cfg['template'] = None
    if not 'nROIs' in cfg.keys():
        cfg['nROIs'] = 100
    if cfg['ROICentroids'] == 'random':
        template = cfg['template']
        nROIs = cfg['nROIs']
        ROICentroids = getRandomCentroids(nROIs,template)
    else:
        ROICentroids = cfg['ROICentroids']
    imgdata = cfg['imgdata']
    voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    if 'threshold' in cfg.keys():
        threshold = cfg['threshold']
    else:
        threshold = -1
    targetFunction = cfg['targetFunction']
    if targetFunction == 'spatialConsistency' or targetFunction == 'weighted mean consistency' or targetFunction == 'local weighted consistency':
        if 'consistencyType' in cfg.keys():
            consistencyType = cfg['consistencyType']
        else:
            consistencyType = 'pearson c'
        if 'fTransform' in cfg.keys():
            fTransform = cfg['fTransform']
        else:
            fTransform = False
    if 'verbal' in cfg.keys():
        verbal = cfg['verbal']
    else:
        verbal = True
    if threshold == 'maximal-voxel-wise':
        if 'nCorrelationsForThresholding' in cfg.keys():
            nCorrelationsForThresholding = cfg['nCorrelationsForThresholding']
        else:
            nCorrelationsForThresholding = 5
    
    nVoxels = len(voxelCoordinates)
    nROIs = len(ROICentroids)
    nTime = imgdata.shape[3]
    
    allVoxelTs = np.zeros((nVoxels,nTime))
    for i,voxel in enumerate(voxelCoordinates):
        allVoxelTs[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
    
    # Setting up: defining initial priority queues, priority measures (centroid-voxel correlations) and candidate voxels to be added per ROI
    ROIMaps = [np.array(centroid) for centroid in ROICentroids]
    ROIVoxels = [np.array(np.where((voxelCoordinates==centroid).all(axis=1)==1)[0]) for centroid in ROICentroids]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels,'ROISizes':np.ones(nROIs,dtype=int),'ROINames':cfg['names']}
    priorityQueues = [findROIlessNeighbors(i,voxelCoordinates,{'ROIMaps':ROIMaps})['ROIlessIndices'].tolist() for i in range(nROIs)] # priority queues change so it's better to keep them as lists
    centroidTs = np.zeros((nROIs,nTime))
    voxelLabels = np.zeros(nVoxels,dtype=int) - 1
    for i, ROIIndex in enumerate(ROIVoxels):
        centroidTs[i,:] = allVoxelTs[ROIIndex[0],:]
        voxelLabels[ROIIndex[0]] = i

    additionCandidates = np.zeros(nROIs,dtype=int)
    maximalMeasures = np.zeros(nROIs)
    
    if targetFunction == 'weighted mean consistency':
        consistencies = [calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},ROI)) for ROI in ROIVoxels]
        ROISizes = [len(ROI) for ROI in ROIVoxels]
    if targetFunction == 'local weighted consistency':
        ROISizes = [len(ROI) for ROI in ROIVoxels]
    
    for i,(priorityQueue,centroid,ROI) in enumerate(zip(priorityQueues,centroidTs,ROIInfo['ROIVoxels'])):
        if targetFunction == 'correlationWithCentroid':
            priorityMeasures = [np.corrcoef(centroid,allVoxelTs[priorityIndex])[0][1] for priorityIndex in priorityQueue]
        elif targetFunction == 'spatialConsistency':
            priorityMeasures = np.zeros(len(priorityQueue))
            for j, voxel in enumerate(priorityQueue):
                voxelIndices = np.concatenate((ROI,np.array([voxel])))
                priorityMeasures[j] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices))
        elif targetFunction == 'weighted mean consistency':
            priorityMeasures = np.zeros(len(priorityQueue))
            tempConsistencies = list(consistencies)
            tempSizes = list(ROISizes)
            tempSizes[i] = tempSizes[i] + 1
            for j, voxel in enumerate(priorityQueue):
                voxelIndices = np.concatenate((ROI,np.array([voxel])))
                tempConsistencies[i] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices))
                priorityMeasures[j] = sum([tempConsistency*tempSize for tempConsistency,tempSize in zip(tempConsistencies,tempSizes)])/sum(tempSizes) 
        elif targetFunction == 'local weighted consistency':
            priorityMeasures = np.zeros(len(priorityQueue))
            for j, voxel in enumerate(priorityQueue):
                voxelIndices = np.concatenate((ROI,np.array([voxel])))
                priorityMeasures[j] = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},voxelIndices)) * (ROISizes[i] + 1)
        if len(priorityMeasures) == 0: #This is a rare case: a ROI has a centroid with no neighbors. So, it's not possible to grow this ROI. 
            additionCandidates[i] = -1
            maximalMeasures[i] = -1
        else:
            additionCandidates[i] = priorityQueue[np.argmax(priorityMeasures)]
            maximalMeasures[i] = np.amax(priorityMeasures)
        
    nInQueue = sum([len(priorityQueue) for priorityQueue in priorityQueues])
    selectedMeasures = []
    mask = np.ones(len(maximalMeasures))
    if threshold == 'voxel-neighborhood':
        thresholdValue = calculateVoxelNeighborhoodCorrelation(voxelCoordinates,allVoxelTs)

    # Actual optimization takes place inside the while loop: 
    while nInQueue>0:
        # Selecting the ROI to be updated and voxel to be added to that ROI (based on the consistency change caused by adding the voxel)
        print(str(int(sum(mask))) + ' candidate ROIs for updating')
        ROIToUpdate = np.argmax(maximalMeasures*mask)
        voxelToAdd = additionCandidates[ROIToUpdate]
        # Checking that adding the voxel doesn't yield sub-threshold consistencies
        if threshold == 'data-driven' or threshold == 'strict data-driven':
            testVoxels = list(ROIInfo['ROIVoxels'])
            testVoxels[ROIToUpdate] = np.concatenate((testVoxels[ROIToUpdate],np.array([voxelToAdd])))
            thresholdValue = calculateThreshold(testVoxels,allVoxelTs,ROIToUpdate)
            print('Threshold value: ' + str(thresholdValue))
            candidateConsistency = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},np.concatenate((ROIInfo['ROIVoxels'][ROIToUpdate],np.array([voxelToAdd])))))
            if candidateConsistency <= thresholdValue:
                mask[ROIToUpdate] = 0
                if np.amax(mask) == 0 or np.all([maxMeasure in [-1,0] for maxMeasure in maximalMeasures*mask]):
                    break
                else:
                    continue
            elif threshold == 'data-driven':
                mask = np.ones(len(maximalMeasures))
        elif threshold == 'voxel-wise' or threshold == 'maximal-voxel-wise':
            testCorrelations = []
            tsToAdd = allVoxelTs[voxelToAdd,:]
            if threshold == 'voxel-wise':
                for i, ROI in enumerate(ROIInfo['ROIVoxels']):
                    testCorrelations.append(np.mean([pearsonr(tsToAdd,allVoxelTs[ROIVoxel,:])[0] for ROIVoxel in ROI]))
            elif threshold == 'maximal-voxel-wise':
                for i, ROI in enumerate(ROIInfo['ROIVoxels']):
                    testCorrelations.append(np.mean(sorted([(pearsonr(tsToAdd,allVoxelTs[ROIVoxel,:])[0]) for ROIVoxel in ROI])[(-1*nCorrelationsForThresholding)::]))
            candidateCorrelation = testCorrelations.pop(ROIToUpdate)
            thresholdValue = max(testCorrelations)
            if candidateCorrelation <= thresholdValue:
                priorityQueues[ROIToUpdate].remove(voxelToAdd)
                nInQueue = nInQueue - 1
                if len(priorityQueues[ROIToUpdate]) > 0:
                    consistencies = np.array([calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},ROI)) for ROI in ROIVoxels])
                    ROISizes = np.array([len(ROI) for ROI in ROIInfo['ROIVoxels']])
                    #import pdb; pdb.set_trace()
                    additionCandidate, maximalMeasure = updateQueue(ROIToUpdate,priorityQueues[ROIToUpdate],targetFunction,centroidTs[ROIToUpdate],allVoxelTs,ROIInfo['ROIVoxels'],consistencies,ROISizes,consistencyType,fTransform)
                    additionCandidates[ROIToUpdate] = additionCandidate
                    maximalMeasures[ROIToUpdate] = maximalMeasure  
                else:
                    maximalMeasures[ROIToUpdate] = -1
                continue  
        elif threshold == 'voxel-neighborhood':
            tsToAdd = allVoxelTs[voxelToAdd]
            candidateCorrelation = np.mean([pearsonr(tsToAdd,allVoxelTs[ROIVoxel,:])[0] for ROIVoxel in ROIInfo['ROIVoxels'][ROIToUpdate]])
            if candidateCorrelation <= thresholdValue:
                priorityQueues[ROIToUpdate].remove(voxelToAdd)
                nInQueue = nInQueue - 1
                if len(priorityQueues[ROIToUpdate]) > 0:
                    consistencies = np.array([calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},ROI)) for ROI in ROIVoxels])
                    ROISizes = np.array([len(ROI) for ROI in ROIInfo['ROIVoxels']])
                    #import pdb; pdb.set_trace()
                    additionCandidate, maximalMeasure = updateQueue(ROIToUpdate,priorityQueues[ROIToUpdate],targetFunction,centroidTs[ROIToUpdate],allVoxelTs,ROIInfo['ROIVoxels'],consistencies,ROISizes,consistencyType,fTransform)
                    additionCandidates[ROIToUpdate] = additionCandidate
                    maximalMeasures[ROIToUpdate] = maximalMeasure  
                else:
                    maximalMeasures[ROIToUpdate] = -1
                continue  
        elif np.amax(maximalMeasures) <= threshold:
            break
        if verbal:
            print(str(nInQueue) + ' voxels in priority queues')
            totalROISize = sum(len(ROIVox) for ROIVox in ROIInfo['ROIVoxels'])
            print(str(totalROISize) + ' voxels in ROIs')

        # Adding the voxel to the ROI
        ROIInfo = addVoxel(ROIToUpdate,voxelToAdd,ROIInfo,voxelCoordinates)
        selectedMeasures.append(np.amax(maximalMeasures))
        voxelLabels[voxelToAdd] = ROIToUpdate
        
        # Updating priority queues: adding the ROIless neighbors of the updated ROI
        neighbors = findROIlessVoxels(findNeighbors(voxelCoordinates[voxelToAdd],allVoxels=voxelCoordinates),ROIInfo)['ROIlessMap'] # Here, we search for the ROIless neighbors of the added voxel; findROIlessNeighbors() can't be used since it finds the ROIless neighbors of a ROI
        ROIlessIndices = [np.where((voxelCoordinates == neighbor).all(axis=1)==1)[0][0] for neighbor in neighbors]
        for ROIlessIndex in ROIlessIndices:
            if not ROIlessIndex in priorityQueues[ROIToUpdate]:
                priorityQueues[ROIToUpdate].append(ROIlessIndex)
        
        # Updating priority queues: removing the added voxel from all priority queues and updating the candidate voxels and maximal measures of the queues that changed        
        if targetFunction == 'weighted mean consistency' or targetFunction == 'local weighted consistency':
            consistencies = np.array([calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'consistencyType':consistencyType,'fTransform':fTransform},ROI)) for ROI in ROIVoxels])
            ROISizes = np.array([len(ROI) for ROI in ROIInfo['ROIVoxels']])
        else:
            consistencies = []
            ROISizes = []
            
        for i, priorityQueue in enumerate(priorityQueues):
            if voxelToAdd in priorityQueue:
                priorityQueue.remove(voxelToAdd) # voxel can belong to more than one priority que; let's remove it from all of them
                if len(priorityQueue) > 0:
                    additionCandidate, maximalMeasure = updateQueue(i,priorityQueue,targetFunction,centroidTs[i],allVoxelTs,ROIInfo['ROIVoxels'],consistencies,ROISizes,consistencyType,fTransform)
                    additionCandidates[i] = additionCandidate
                    maximalMeasures[i] = maximalMeasure
                else:
                    maximalMeasures[i] = -1
        
        nInQueue = sum([len(priorityQueue) for priorityQueue in priorityQueues])
    
    spatialConsistency = calculateSpatialConsistencyInParallel(ROIInfo['ROIVoxels'],allVoxelTs,consistencyType=consistencyType,fTransform=fTransform,nCPUs=5)
    meanConsistency = np.mean(spatialConsistency)

    return voxelLabels, voxelCoordinates, meanConsistency
    
def growOptimizedROIsInParallel(cfg, nIter=100, nCPUs=5):
    """
    Creates nIter sets of random seeds and creates optimized ROIs based on these
    seeds, aftr which finds the best set of ROIs based on mean spatial consistency.
    
    Parameters:
    cfg: dict, contains:
         names: list of strs, names of the ROIs, can be e.g. the anatomical name associated with
                  the centroid. Default = ''.
    -----------
         imgdata: x x y x z x t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
         threshold: float, the lowest centroid-voxel correlation that leads to adding a voxel (default=-1)
         targetFunction: str, measure that will be optimized (options: correlationWithCentroid, spatialConsistency)
         consistencyType: str, definition of spatial consistency to be used if 
                          targetFunction == 'spatialConsistency' (default: 'pearson c' (mean Pearson correlation coefficient))
         fTransform: bool, should Fisher Z transform be applied if targetFunction == 'spatialConsistency' 
                     (default=False)
         template: 3D numpy array where each element corresponds to a voxel. The value of each voxel is the index of the ROI the voxel belongs to.
                   Voxels outside of ROIs (= outside of the gray matter) have value 0.
         nROIs: int, number of ROIs(default = 100).
    nIter: int, number of random seed sets to generate (default = 100)
    nCPUs: int, number of CPUs to be used for the parallel computing (default = 5)
    
    Returns:
    --------
    voxelLabels: nVoxels x 1 np.array, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelIndices: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    """
    #cfg['ROICentroids'] = 'random'
    if not 'nROIs' in cfg.keys():
        cfg['nROIs'] = 100
    #cfg['verbal'] = False
    print('Starting optimization')
    
    if False:
        paramSpace = [(cfg) for iterInd in np.arange(nIter)]
        pool = Pool(max_workers = nCPUs)
        results = list(pool.map(growOptimizedROIs,paramSpace,chunksize=1))
        voxelLabelList = []
        voxelCoordinateList = []
        meanConsistencies = []
        for result in results:
            voxelLabelList.append(result[0])
            voxelCoordinateList.append(result[1])
            meanConsistencies.append(result[2])
    else: # this is a debugging case
        voxelLabelList = []
        voxelCoordinateList = []
        meanConsistencies = []
        for i in range(nIter):
            voxelLabels,voxelCoordinates,meanConsistency = growOptimizedROIs(cfg)
            voxelLabelList.append(voxelLabels)
            voxelCoordinateList.append(voxelCoordinates)
            meanConsistencies.append(meanConsistency)
            
    #import pdb; pdb.set_trace()        
    voxelLabels = voxelLabelList[np.argmax(meanConsistencies)]
    voxelIndices = voxelCoordinateList[np.argmax(meanConsistencies)]
    
    print('Optimization done!')
    return voxelLabels, voxelIndices

def optimizeParcellationByFlippingInitial(cfg):
    """
    Optimizes a given parcellation (i.e. clustering of voxels to ROIs) in terms of
    consistency by flipping boundary voxels of ROIs to neighboring ROIs. The function
    has three input format options, corresponding the three ways used to represent
    parcellations in the pipeline.
    
    The flipping process is as following:
        1) calculate target function for all ROIs
        2) find boundary voxels of all ROIs; these are the flipping candidate voxels
        3) for each ROI, voxel by voxel check if flipping a candidate voxel to a 
           neighboring ROI increases the target function value
           3.1) if so, move the candidate voxel to the new ROI and update boundaries of both ROIs
           3.2) if not, don't change the ROIs but remove the candidate voxel from the boundary
        
    NOTE: optimizeParcellationByFlipping below is a more sophisticated version of this function!
    
    Parameters:
    -----------
    cfg: dict, contains:
         inputType: str, specifies the format of the input. Option: 'ROIInfo', 'voxelLabels', 'voxelsInClusters'.
                    For descriptions of the input formats, see below. (default = 'ROIInfo')
                    
         ROIInfo: dict, the input used when inputType == 'ROIInfo'. This is the default input format, other
                  formats are converted to ROIInfo. Contains:
                  ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
                  ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
                             to each ROI. These indices refer to the rows of the 
                             voxelCoordinates array. len(ROIVoxels) = nROIs.
         OR
         voxelLabels: nVoxels x 1 np.array, ROI labels (i.e. indices of ROIs where the 
                      voxels belong) of all voxels; voxels not belongin to any ROI have
                      label -1.
         OR
         voxelsInClusters: dict, where keys are ROI labels and values are lists of voxel
                           coordinates belonging to the ROIs
                           
         imgdata: x x y x z x t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
         targetFunction: str, measure that will be optimized (options: 'mean consistency', 'weighted mean consistency')
         fTransform: bool, should Fisher Z transform be applied if targetFunction == 'spatialConsistency' 
                     (default=False)
         
    Returns:
    --------
    voxelLabels: list, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    targets: list, values of the target functions through the flipping process
    """
    # setting up: reading parameters
    if 'inputType' in cfg.keys():
        inputType = cfg['inputType']
    else:
        inputType = 'ROIInfo'
    if 'fTransform' in cfg.keys():
        fTransform = cfg['fTransform']
    else:
        fTransform = False
    targetFunction = cfg['targetFunction']
    imgdata = cfg['imgdata']
    
    if inputType == 'voxelLabels':
        voxelLabels = cfg['voxelLabels']
        voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True))) # indices of voxels belonging to gray matter in the imgdata, independent of if they belong to a ROI
        ROIInfo = voxelLabelsToROIInfo(voxelLabels,voxelCoordinates)
    elif inputType == 'voxelsInClusters':
        ROIInfo, voxelCoordinates, voxelLabels = voxelsInClustersToROIInfo(cfg['voxelsInClusters']) # this return only coordinates of voxels belonging to a ROI 
    elif inputType == 'ROIInfo':
        voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
        voxelLabels = findVoxelLabels(voxelCoordinates,ROIInfo)
    
    ROIVoxels = ROIInfo['ROIVoxels']
    
    nVoxels = voxelCoordinates.shape[0]
    nROIs = len(ROIVoxels)
    nTime = imgdata.shape[-1]
    allVoxelTs = np.zeros((nVoxels,nTime))
    for i,voxel in enumerate(voxelCoordinates):
        allVoxelTs[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
        
    # initializing target function
        
    consistencies = []
    
    for i, voxelIndices in enumerate(ROIVoxels):
        consistencies.append(calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'fTransform':fTransform},voxelIndices)))
    
    if targetFunction == 'mean consistency': # normal average
        target = sum(consistencies) / len(consistencies)
    elif targetFunction == 'weighted mean consistency': # weighted average
        ROISizes = np.array([len(voxelIndices) for voxelIndices in ROIVoxels])
        consistencies = consistencies * ROISizes
        target = sum(consistencies) / sum(ROISizes)
    
    targets = []
        
    # finding ROI boundaries
    
    boundaryIndices = []
    neighborLabels = []
    for ROILabel in range(len(ROIVoxels)):
        indBoundaryIndices,_, indNeighborLabels = findROIBoundary(ROILabel,ROIInfo,voxelCoordinates,voxelLabels)
        boundaryIndices.append(indBoundaryIndices)
        neighborLabels.append(indNeighborLabels)
    
    # actual optimization starts here:
    # TODO: how to handle a case where all neighbors of a voxel have already been flipped away = the voxel is not physically connected to its ROI?
    for ROIFromFlip in range(nROIs): 
        print('Optimizing ROI ' + str(ROIFromFlip))
        while len(boundaryIndices[ROIFromFlip]) > 0: # not all boundary voxels of the ROI have been tested yet
            targets.append(target)
            voxelToFlip = boundaryIndices[ROIFromFlip][0] # Idea here: we always pick the first voxel and remove it afterwards
            while len(neighborLabels[ROIFromFlip][0]) > 0: # flipping to all neighboring ROIs has not been tested yet
                ROIToFlip = neighborLabels[ROIFromFlip][0][0]
                neighborLabels[ROIFromFlip][0].remove(ROIToFlip)
                # temporarely move the flip candidate voxel to a new ROI and calculate target
                fromVoxels = list(ROIVoxels[ROIFromFlip]) # Here, list() is used for creating a copy: we don't want to update the ROIs yet
                fromVoxels.remove(voxelToFlip)
                if not fromVoxels == []:
                    fromConsistency = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'fTransform':fTransform},fromVoxels))
                else:
                    fromConsistency = 0
                toVoxels = list(ROIVoxels[ROIToFlip])
                toVoxels.append(voxelToFlip)
                toConsistency = calculateSpatialConsistency(({'allVoxelTs':allVoxelTs,'fTransform':fTransform},toVoxels))
                tempConsistencies = list(consistencies)
                tempConsistencies.remove(consistencies[ROIFromFlip])
                tempConsistencies.remove(consistencies[ROIToFlip])
                if targetFunction == 'mean consistency': # normal average
                    tempConsistencies.extend([fromConsistency,toConsistency])
                    tempTarget = np.mean(tempConsistencies)
                elif targetFunction == 'weighted mean consistency': # weighted average
                    tempConsistencies.extend([fromConsistency*(ROISizes[ROIFromFlip]-1),toConsistency*(ROISizes[ROIToFlip]+1)])
                    tempTarget = sum(tempConsistencies) / sum(ROISizes)
                # if value of target function increases, let's update:
                if tempTarget > target:
                    # updating ROI maps and voxel labels
                    removeVoxel(ROIFromFlip,voxelToFlip,ROIInfo,voxelCoordinates)
                    addVoxel(ROIToFlip,voxelToFlip,ROIInfo,voxelCoordinates)
                    voxelLabels[voxelToFlip] = ROIToFlip
                    # updating boundary of old ROI
                    k = boundaryIndices[ROIFromFlip].index(voxelToFlip)
                    boundaryIndices[ROIFromFlip].remove(voxelToFlip)
                    neighborLabels[ROIFromFlip].remove(neighborLabels[ROIFromFlip][k])
                    fromROINeighbors = findInROINeighbors(voxelToFlip,ROIFromFlip,voxelCoordinates,voxelLabels)
                    for neighbor in fromROINeighbors:
                        boundaryVar, labels = isBoundary(neighbor,voxelCoordinates,voxelLabels)
                        if boundaryVar and neighbor not in boundaryIndices[ROIFromFlip]:
                            boundaryIndices[ROIFromFlip].append(neighbor)
                            neighborLabels[ROIFromFlip].append(labels)
                    # updating boundary of new ROI
                    boundaryVar, labels = isBoundary(voxelToFlip,voxelCoordinates,voxelLabels)
                    if boundaryVar:
                        boundaryIndices[ROIToFlip].append(voxelToFlip)
                        neighborLabels[ROIToFlip].append(isBoundary(voxelToFlip,voxelCoordinates,voxelLabels)[1])                    
                    toROINeighbors = findInROINeighbors(voxelToFlip,ROIToFlip,voxelCoordinates,voxelLabels)
                    for neighbor in toROINeighbors:
                        boundaryVar,_ = isBoundary(neighbor,voxelCoordinates,voxelLabels)
                        if not boundaryVar and neighbor in boundaryIndices[ROIToFlip]:
                            k = boundaryIndices[ROIToFlip].index(neighbor)
                            boundaryIndices[ROIToFlip].remove(neighbor)
                            neighborLabels[ROIToFlip].remove(neighborLabels[ROIToFlip][k])
                    # updating target, consistencies, and ROI sizes
                    target = tempTarget
                    if targetFunction == 'mean consistency':
                        consistencies[ROIFromFlip] = fromConsistency
                        consistencies[ROIToFlip] = toConsistency
                    elif targetFunction == 'weighted mean consistency':
                        ROISizes[ROIFromFlip] = ROISizes[ROIFromFlip] - 1
                        ROISizes[ROIToFlip] = ROISizes[ROIToFlip] + 1
                        consistencies[ROIFromFlip] = fromConsistency * ROISizes[ROIFromFlip]
                        consistencies[ROIToFlip] = toConsistency * ROISizes[ROIToFlip]
                    break
            # if value of target function doesn't increase, let's just remove the flipping candidate after testig all of its neighbor ROIs
            if tempTarget < target:
                k = boundaryIndices[ROIFromFlip].index(voxelToFlip)
                boundaryIndices[ROIFromFlip].remove(voxelToFlip)
                neighborLabels[ROIFromFlip].remove(neighborLabels[ROIFromFlip][k])

    return voxelLabels, voxelCoordinates, targets

def optimizeParcellationByFlipping(cfg):
    """
    Optimizes a given parcellation in terms of (weighted) mean consistency by
    flipping boundary voxels of ROIs to neighboring ROIs.    
    # TODO: add here a description of the algorithm!
    
    Parameters:
    -----------
    cfg: dict, contains:
         inputType: str, specifies the format of the input. Option: 'ROIInfo', 'voxelLabels', 'voxelsInClusters'.
                    For descriptions of the input formats, see below. (default = 'ROIInfo')
                    
         ROIInfo: dict, the input used when inputType == 'ROIInfo'. This is the default input format, other
                  formats are converted to ROIInfo. Contains:
                  ROIMaps: list of ROISizes x 3 np.arrays, coordinates of voxels
                           belonging to each ROI. len(ROIMaps) = nROIs.
                  ROIVoxels: list of ROISizes x 1 np.array, indices of the voxels belonging
                             to each ROI. These indices refer to the rows of the 
                             voxelCoordinates array. len(ROIVoxels) = nROIs.
         OR
         voxelLabels: nVoxels x 1 np.array, ROI labels (i.e. indices of ROIs where the 
                      voxels belong) of all voxels; voxels not belongin to any ROI have
                      label -1.
         OR
         voxelsInClusters: dict, where keys are ROI labels and values are lists of voxel
                           coordinates belonging to the ROIs
                           
         imgdata: x x y x z x t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
         targetFunction: str, measure that will be optimized (options: 'mean consistency', 'weighted mean consistency')

    Returns:
    --------
    voxelLabels: list, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    targets: list, values of the target functions through the flipping process
    """
    # setting up: reading parameters
    if 'inputType' in cfg.keys():
        inputType = cfg['inputType']
    else:
        inputType = 'ROIInfo'
    targetFunction = cfg['targetFunction']
    imgdata = cfg['imgdata']
    
    if inputType == 'voxelLabels':
        voxelLabels = cfg['voxelLabels']
        voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True))) # indices of voxels belonging to gray matter in the imgdata, independent of if they belong to a ROI
        ROIInfo = voxelLabelsToROIInfo(voxelLabels,voxelCoordinates,constructROIMaps=False)
    elif inputType == 'voxelsInClusters':
        ROIInfo, voxelCoordinates, voxelLabels = voxelsInClustersToROIInfo(cfg['voxelsInClusters']) # this return only coordinates of voxels belonging to a ROI 
    elif inputType == 'ROIInfo':
        voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True))) # indices of voxels belonging to gray matter
        voxelLabels = findVoxelLabels(voxelCoordinates,ROIInfo)
    ROIVoxels = ROIInfo['ROIVoxels']
    # setting up: reading voxel time series
    nVoxels = voxelCoordinates.shape[0]
    nTime = imgdata.shape[-1]
    allVoxelTs = np.zeros((nVoxels,nTime))
    for i,voxel in enumerate(voxelCoordinates):
        allVoxelTs[i,:] = imgdata[voxel[0],voxel[1],voxel[2],:]
        
    # initialization: calculating sums of correlations inside ROIs
    # TODO: check if it's ok that correlationSums and ROISizes are lists (are array features required later on?)
    correlationSums = []
    ROISizes = []
    for ROIIndex, voxels in enumerate(ROIVoxels):
        ROISizes.append(len(voxels))
        if len(voxels) == 1:
            correlationSums.append = 1. # a single voxel is always fully correlated with itself
        else: 
            voxelTs = allVoxelTs[voxels,:]
            correlations = np.corrcoef(voxelTs)
            correlationSums.append((np.sum(correlations) - len(voxels))/2) # -len(voxels) removes the sum of diagonal elements, /2 is done because the sum contains both C(a,b) and C(b,a)
            
    # initialization: finding ROI boundaries and constructing flip tuples
    flips = {} # TODO: this is a placeholder; replace with a reasonable data type!!!
    for ROIIndex, voxels in enumerate(ROIVoxels):
        print('Defining boundaries for ROI ' + str(ROIIndex))
        rejectedFlips = 0
        sourceSize = ROISizes[ROIIndex]
        sourceSum = correlationSums[ROIIndex]
        for voxel in voxels:
            inBoundary, neighborROIs = isBoundary(voxel,voxelCoordinates,voxelLabels)
            if inBoundary:
                voxelTs = allVoxelTs[voxel]
                sourceVoxels = list(voxels)
                sourceVoxels.remove(voxel)
                sourceFlipSum = sum([pearsonr(allVoxelTs[sourceVoxel,:],voxelTs)[0] for sourceVoxel in sourceVoxels])
                for neighborROI in neighborROIs:
                    flip = (voxel, ROIIndex, neighborROI)
                    if checkFlipValidity(flip,ROIVoxels,voxelCoordinates,voxelLabels):
                        targetSize = ROISizes[neighborROI]
                        targetSum = correlationSums[neighborROI]
                        targetVoxels = ROIVoxels[neighborROI]
                        targetFlipSum = sum([pearsonr(allVoxelTs[targetVoxel,:],voxelTs)[0] for targetVoxel in targetVoxels])
                        if targetFunction == 'weighted mean consistency': # TODO: construct a case for targetFunction == 'mean consistency'
                            target = calculateTarget(nVoxels,sourceSize,sourceSum,sourceFlipSum,targetSize,targetSum,targetFlipSum)
                        flips[target] = flip
                    else:
                        rejectedFlips = rejectedFlips + 1
        print(str(rejectedFlips) + ' unvalid flips rejected')
                    
    # optimization starts from here:
    # TODO: update the finding of max flip etc. after deciding the final datatype
    targets = []
    meanWeightedConsistencies = []
    missingSource = 0
    while max(flips.keys()) > 0:
        #import pdb; pdb.set_trace()
        meanWeightedConsistency = sum([ROISize*2*correlationSum/(ROISize*(ROISize-1)) for correlationSum,ROISize in zip(correlationSums,ROISizes)])
        meanWeightedConsistency = meanWeightedConsistency/sum(ROISizes)
        meanWeightedConsistencies.append(meanWeightedConsistency)
        # finding the next flip
        maxTarget = max(flips.keys())
        targets.append(maxTarget)
        voxel, sourceROI, targetROI = flips.pop(maxTarget)
        voxelTs = allVoxelTs[voxel]
        print('Optimizing by flipping, max target value ' +str(maxTarget) + ', flip: (' + str(voxel) + ', ' + str(sourceROI) + ', ' + str(targetROI) + ')')
        print('Mean weighted consistency: ' + str(meanWeightedConsistency))
        totalNVoxels = sum([len(voxels) for voxels in ROIVoxels])
        print('Total number of voxels: ' + str(totalNVoxels))
        if totalNVoxels != 16855 or len(ROIVoxels) != 246:
            import pdb; pdb.set_trace()
        sourceVoxels = list(ROIVoxels[sourceROI])
        sourceVoxels.remove(voxel)
        targetVoxels = list(ROIVoxels[targetROI])
        sourceFlipSum = sum([pearsonr(allVoxelTs[sourceVoxel,:],voxelTs)[0] for sourceVoxel in sourceVoxels])
        targetFlipSum = sum([pearsonr(allVoxelTs[targetVoxel,:],voxelTs)[0] for targetVoxel in targetVoxels])
        # moving voxel from sourceROI to targetROI and updating voxelLabels and ROISizes
        removeVoxel(sourceROI,voxel,ROIInfo,voxelCoordinates)
        addVoxel(targetROI,voxel,ROIInfo,voxelCoordinates)
        voxelLabels[voxel] = targetROI
        ROISizes[sourceROI] = ROISizes[sourceROI] - 1
        ROISizes[targetROI] = ROISizes[targetROI] + 1
        # updating correlationSums
        correlationSums[sourceROI] = correlationSums[sourceROI] - sourceFlipSum
        correlationSums[targetROI] = correlationSums[targetROI] + targetFlipSum
        # updating flip candidate list
        # 1) removing from flips all flips where voxel is flipped from sourceROI
        flips = {target:flip for target, flip in flips.items() if flip[0] != voxel}
        # 2) the flipped voxel itself belong to the boundary of targetROI
        flips[-1*maxTarget] = (voxel,targetROI,sourceROI)
        # 3) updating flips of the flipped voxel from targetROI to other possibly neighboring ROIs
        _,neighborROIs = isBoundary(voxel,voxelCoordinates,voxelLabels)
        if sourceROI in neighborROIs: # TODO: check how a sourceROI not in neighborROIs situation is possible; I think this is possible only if the voxel is a salt-and-pepper like anomaly which should be removed by checking the validity of flips
            neighborROIs.remove(sourceROI)
        else:
            missingSource = missingSource + 1
        for neighborROI in neighborROIs:
            if checkFlipValidity((voxel,targetROI,neighborROI),ROIVoxels,voxelCoordinates,voxelLabels):
                sourceSize = ROISizes[targetROI] # TODO: could the flip construction be a separated function that takes sourceIndex, targetIndex, voxel, and allVoxelTs?
                targetSize = ROISizes[neighborROI]
                sourceSum = correlationSums[targetROI]
                targetSum = correlationSums[neighborROI]
                sourceVoxels = list(ROIVoxels[targetROI])
                sourceVoxels.remove(voxel)
                targetVoxels = ROIVoxels[neighborROI]
                sourceFlipSum = sum([pearsonr(allVoxelTs[sourceVoxel,:],voxelTs)[0] for sourceVoxel in sourceVoxels])
                targetFlipSum = sum([pearsonr(allVoxelTs[targetVoxel,:],voxelTs)[0] for targetVoxel in targetVoxels])
                target = calculateTarget(nVoxels,sourceSize,sourceSum,sourceFlipSum,targetSize,targetSum,targetFlipSum)
                flips[target] = (voxel,targetROI,neighborROI)
        # 4) neighbors of the flipped voxel in the sourceROI, note that these are always in boundary
        # neighbors in source and target ROIs are handled separately to make it easier to later on check flip validity
        neighborVoxels = findInROINeighbors(voxel,sourceROI,voxelCoordinates,voxelLabels)
        for neighborVoxel in neighborVoxels:
            neighborVoxelTs = allVoxelTs[neighborVoxel,:]
            _,neighborROIs = isBoundary(neighborVoxel,voxelCoordinates,voxelLabels)
            for neighborROI in neighborROIs:
                if (neighborVoxel,sourceROI,neighborROI) in flips.values():
                    flips = {target:flip for target,flip in flips.items() if flip!=(neighborVoxel,sourceROI,neighborROI)} # TODO: this probably pretty slow so a better way should be found...
                if checkFlipValidity((neighborVoxel,sourceROI,neighborROI),ROIVoxels,voxelCoordinates,voxelLabels):
                    sourceSize = ROISizes[sourceROI]
                    targetSize = ROISizes[neighborROI]
                    sourceSum = correlationSums[sourceROI]
                    targetSum = correlationSums[neighborROI]
                    sourceVoxels = list(ROIVoxels[sourceROI])
                    sourceVoxels.remove(neighborVoxel)
                    targetVoxels = ROIVoxels[neighborROI]
                    sourceFlipSum = sum([pearsonr(allVoxelTs[sourceVoxel,:],neighborVoxelTs)[0] for sourceVoxel in sourceVoxels])
                    targetFlipSum = sum([pearsonr(allVoxelTs[targetVoxel,:],neighborVoxelTs)[0] for targetVoxel in targetVoxels])
                    target = calculateTarget(nVoxels,sourceSize,sourceSum,sourceFlipSum,targetSize,targetSum,targetFlipSum)
                    flips[target] = (neighborVoxel,sourceROI,neighborROI)
        # 5) neighbors of the flipped voxel in the targetROI
        neighborVoxels = findInROINeighbors(voxel,targetROI,voxelCoordinates,voxelLabels)
        for neighborVoxel in neighborVoxels:
            neighborVoxelTs = allVoxelTs[neighborVoxel,:]
            inBoundary,neighborROIs = isBoundary(neighborVoxel,voxelCoordinates,voxelLabels)
            if inBoundary:
                for neighborROI in neighborROIs:
                    if (neighborVoxel,targetROI,neighborROI) in flips.values():
                        flips = {target:flip for target,flip in flips.items() if flip!=(neighborVoxel,targetROI,neighborROI)} # TODO: this probably pretty slow so a better way should be found...
                    if checkFlipValidity((neighborVoxel,targetROI,neighborROI),ROIVoxels,voxelCoordinates,voxelLabels):
                        sourceSize = ROISizes[targetROI]
                        targetSize = ROISizes[neighborROI]
                        sourceSum = correlationSums[targetROI]
                        targetSum = correlationSums[neighborROI]
                        sourceVoxels = list(ROIVoxels[targetROI])
                        sourceVoxels.remove(neighborVoxel)
                        targetVoxels = ROIVoxels[neighborROI]
                        sourceFlipSum = sum([pearsonr(allVoxelTs[sourceVoxel,:],neighborVoxelTs)[0] for sourceVoxel in sourceVoxels])
                        targetFlipSum = sum([pearsonr(allVoxelTs[targetVoxel,:],neighborVoxelTs)[0] for targetVoxel in targetVoxels])
                        target = calculateTarget(nVoxels,sourceSize,sourceSum,sourceFlipSum,targetSize,targetSum,targetFlipSum)
                        flips[target] = (neighborVoxel,targetROI,neighborROI)
            else:
                flips = {target:flip for target,flip in flips.items() if flip[0] != neighborVoxel} # removing all flips containing this specific voxel
    print(str(missingSource) + 'source ROIs missing from neighboring ROIs')    
    import pdb; pdb.set_trace()            
    return voxelCoordinates, voxelLabels, targets, meanWeightedConsistencies
     
    # some pseudocode for what follows:
    # 0) ensure that there's a ROIVoxels list OK
    # 1) find boundary voxels of all ROIs
    # 2) create a collection of (voxel,fromROI,toROI) tuples OK (excluding the points below)
    #      2.1) find an as efficient as possible datatype for this
    #      2.2) check that the flips are valid (i.e. don't break the ROI into two)
    #           to this end, there need to be a path between each in-ROI neighbor of the voxel after the flip
    #           NOTE: IT'S BEST TO DO THIS WITH A SEPARATE FUNCTION; THIS REQUIRES A BIT MORE INVESTIGATION
    # 3) for each ROI,
    #        calculate sum of correlations inside the ROI OK
    #        calculate size OK
    # 4) for each (voxel,fromROI,toROI)
    #        calculate sum(C(voxel, rest of fromROI)) OK
    #        calculate sum(C(voxel,toROI)) OK
    #        calculate delta(target) (check overleaf for the formula) OK
    #    4.1) find the best datatype for storing these values
    # THE OPTIMIZATION STARTS HERE:
    # while max(delta(target)) > 0:
    #     flip = (voxel,fromROI,toROI) with max(delta(target)) OK
    #     update ROISizes: fromSize = fromSize - 1, toSize = toSize + 1 OK
    #     update ROIVoxels OK
    #     update correlation sums: C(fromROI) = C(fromROI) - C(voxel, rest of fromROI), C(toROI) = C(toROI) + C(voxel, toROI) OK
    #     update boundaries:
    #     A) find neighbors of voxel in fromROI. Use isBoundary() to check if they belong to the boundary. If so, find their neighboring
    #        ROIs and if needed, add the corresponding tuples to the flip list. If not, check that they are not in the flip list.
    #        For tuples remaining in the list, update delta(target). OK
    #     B) find neighbors of voxel in toROI. These are by definition in boundary; check with isBoundary() if they still belong
    #        there after the update and if their neighboring ROIs have changed. Remove/update tuples as needed. For tuples remaining
    #        in the list, update delta(target).
    #     NOTE: BEFORE ADDING ANY TUPLES, CHECK THAT THEY ARE VALID FLIPS (SEE 2.2 ABOVE)
     
def spectralNCutClustering(cfg):
    """
    Clusters voxels to ROIs using the spectral ncut method introduced by Craddock
    et al. 2012 (Hum Brain Mapp. 33(8)). This function is meant to be applied on the
    data of a single subject and therefore lacks the second, group-level clustering
    also presented in the article.
    
    NOTE: Subfunctions of this function makes strong use of pyClusterROI py Cameron 
    Craddock et al. (http://ccraddock.github.io/cluster_roi/). Using this function
    requires the python_ncut_lib distributed with pyClusterROI to be saved in a 
    location accessible through PYTHONPATH.
    
    Parameters:
    -----------
    cfg: dict, contains:
         imgdata: x*y*z*t np.array, fMRI measurement data to be used for the clustering.
                  Three first dimensions correspond to voxel coordinates while the fourth is time.
                  For voxels outside of the gray matter, all values must be set to 0.
         threshold: float, threshold value, correlation coefficients lower than this value
                    will be removed from the matrix (set to zero).
         nROIs: int, number of clusters to construct. Note that this is the aimed number of ROIs,
                and the actual number of produced ROIs may be slightly smaller.
    
    Returns:
    --------
    voxelLabels: nVoxels x 1 np.array, ROI labels of all voxels. Voxels that don't belong to any ROI have label -1.
    voxelCoordinates: list (len = nVoxels) of tuples (len = 3), coordinates (in voxels) of all voxels
    """
    imgdata = cfg['imgdata']
    thres = cfg['threshold']
    nClusters = cfg['nROIs']
    voxelCoordinates = list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    
    localConnectivity = makeLocalConnectivity(imgdata,thres)
    voxelLabels = nCutClustering(localConnectivity,nClusters)
    voxelLabels = np.squeeze(np.array(voxelLabels.todense())).astype(int)
    
    # fixing the gaps in cluster labeling (because of the clustering method, some labels may be missing)
    labels = np.unique(voxelLabels)
    for label, expectedLabel in zip(labels,range(1,len(labels)+1)):
        if not label == expectedLabel:
            voxelLabels[np.where(voxelLabels==label)]=expectedLabel
    
    return voxelLabels,voxelCoordinates

# post-clustering analysis

def findSuccessors():
    """
    For each ROI on layer t, finds successor, or the ROI, to which the majority
    of ROI's voxels belong on layer t+1 and calculates the percentage of voxels
    that moved to other ROIs than the successor.
    
    Parameters:
    -----------
    
    Returns:
    --------
    successors: array-like of ints, indexes of the successor ROIs
    changePercentages: array-like of doubles, percentages of voxels moving to other ROIs
                       than the successor.
    """
    
        
    
    
    
    
