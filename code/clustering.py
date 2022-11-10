import numpy as np
import sklearn.cluster
import scipy.sparse.csgraph
import collections

import corrs_and_mask_calculations
import clustering_by_consistency as cbc
'''
Clustering on brain data.
'''


def calculate_correlation_distance_matrix(R):
    # This function exists for readability and modularity
    return 1-R

def cluster_timewindow_scikit(imgdata, **kwargs):
    # Using precomputed affinity (correlation distances)
    # Connectivity matrix: only voxels that share a face can be connected
    R,unmasked_voxels = corrs_and_mask_calculations.calculate_correlation_matrix(imgdata,exclude_masked=True)
    Rdist = calculate_correlation_distance_matrix(R)
    connectivity_matrix = create_connectivity_matrix(unmasked_voxels)
    n_components,component_labels = get_connected_components(imgdata)
    component_sizes = collections.Counter(component_labels)
    ordered_component_sizes = component_sizes.most_common()
    if n_components > 1:
        if ordered_component_sizes[0][1] == ordered_component_sizes[1][1]:
            raise NotImplementedError("Multiple largest components")
        if ordered_component_sizes[1][1] > 10:
            raise NotImplementedError("Non-largest components of size > 10 detected (not negligible)")
        largest_component_voxels = [unmasked_voxels[i] for i in range(len(unmasked_voxels)) if component_labels[i] == ordered_component_sizes[0][0]]
        R,largest_component_voxels = corrs_and_mask_calculations.calculate_correlation_matrix_for_voxellist(imgdata,largest_component_voxels)
        Rdist = calculate_correlation_distance_matrix(R)
        connectivity_matrix = create_connectivity_matrix(largest_component_voxels)
    model = sklearn.cluster.AgglomerativeClustering(affinity='precomputed',connectivity=connectivity_matrix,linkage='average', **kwargs)
    model.fit(Rdist)
    if n_components > 1:
        return model,largest_component_voxels
    else:
        return model,unmasked_voxels
    # TODO: smarter way to communicate which voxels were used
    
def create_connectivity_matrix(voxellist):
    # voxellist = list of coordinate tuples
    # Neighbor = voxel where exactly one coordinate is changed by +-1
    indices_of_voxels = dict()
    for index,voxel in enumerate(voxellist):
        indices_of_voxels[voxel] = index
    connectivity_matrix = np.zeros((len(voxellist),len(voxellist)))
    for voxel in voxellist:
        voxel_index = indices_of_voxels[voxel]
        for dimension in range(3):
            for dimension_shift in [-1,1]:
                neighbor = list(voxel)
                neighbor[dimension] = neighbor[dimension]+dimension_shift
                neighbor = tuple(neighbor)
                neighbor_index = indices_of_voxels.get(neighbor,None)
                if neighbor_index != None:
                        connectivity_matrix[voxel_index,neighbor_index] = 1
                        connectivity_matrix[neighbor_index,voxel_index] = 1
    return connectivity_matrix
    
def get_number_of_connected_components_each_timewindow(imgdata,timewindow=100,overlap=0):
    # Check number of connected components for each timewindow and return them in a list
    # This number should be the same for each timewindow, so this function exists just for possible future use
    # Returns list of ints
    k = int(1 + np.floor((imgdata.shape[3]-timewindow)/float((timewindow-overlap))))
    n_connected_components = []
    for tw_no in range(k):
        start = tw_no*(timewindow-overlap)
        end = timewindow + tw_no*(timewindow-overlap) - 1
        n_connected_components.append(get_connected_components(imgdata[:,:,:,start:end])[0])
    return n_connected_components
    
def get_connected_components(imgdata):
    # Returns tuple (number_of_components, component_labels)
    # where component_labels is an array
    # Finds connected components among unmasked voxels (neighbor definition from create_connectivity_matrix)
    unmasked_voxels = corrs_and_mask_calculations.find_unmasked_voxels(imgdata)
    connectivity_matrix = create_connectivity_matrix(unmasked_voxels)
    return scipy.sparse.csgraph.connected_components(connectivity_matrix)
