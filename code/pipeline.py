import nibabel as nib
import os
import datetime
import collections
import pickle

import network_construction
import network_io
import subgraph_classification
import corrs_and_mask_calculations
import clustering_by_consistency as cbc

def isomorphism_classes_from_file(filename,data_mask_filename,
                                timewindow,overlap,density_params,
                                clustering_method_params,
                                nlayers,nnodes,isomorphism_allowed_aspects=[0],
                                isomorphism_class_savenames=None,
                                isomorphism_class_examples_savenames=None,
                                layersetwise_networks_savefolder=None,
                                log_savename=None):
    """Create isomorphism class dictionary/ies from brain imaging data.
    
    Arguments:
    filename : str, name of nifti file
    data_mask_filename : str, name of file containing the gray matter mask for the data
        (None = no masking)
    timewindow : int, length of timewindow in data points
    overlap : int, length of overlap between consecutive timewindows in data points
    density_params : dict, with keys
        intralayer_density : float, density of intralayer networks
        interlayer_density : float, density of interlayer networks
        OR
        intra_avg_degree : float, average intralayer degree of nodes
        inter_avg_degree : float, average in- and out-degree of nodes
    clustering_method_params : dict, with keys
        method : str, name of the clustering method
        AND
        key-value pairs giving parameters for chosen method (see section Clustering)
    nlayers : int, the number of layers in graphlets of interest
    nnodes : int or list of ints, the number of nodes in graphlets of interest
        if list, each entry will be combined with nlayers and enumerated
    isomorphism_allowed_aspects : list, define allowed aspects for isomorphism
    isomorphism_class_savenames : str or list of strs, the same length as nnodes
        filenames for saving found isomorphism class dicts (None = no saving)
    isomorphism_class_examples_savenames : str of list of strs, the same length as nnodes
        filenames for saving example networks for each isomorphism class (None = no saving)
    layersetwise_networks_savefolder : str, folder for saving the generated networks
        (None = no saving)
    log_savename : str, appends successful completion info to this file (None = no logging)
    
    Clustering:
    TODO explanation
    
    Returns:
    dict of dicts of dicts, first level of keys is (nnodes,nlayers) pairs as tuples,
    second level of keys is isomorphism classes as tuples (complete invariants),
    and third level of keys is (ordered) layersets as tuples. The number of (nnodes,nlayers)
    pairs depends on how many different nnodes are given.
    return_dict[(nnodes,nlayers)][compinvariant][(layerset)] = frequency
    (dicts are collections.defaultdict objects)
    """
    # convert int nnodes to length-1 list
    nnodes = [nnodes] if isinstance(nnodes,int) else nnodes
    # create container data structures for isomorphism classes
    aggregated_isomclass_dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    aggregated_example_dict = collections.defaultdict(dict)
    # load data
    data = nib.load(filename)
    image_array = data.get_fdata()
    # mask data
    if data_mask_filename:
        maskdata = nib.load(data_mask_filename)
        mask_array = maskdata.get_fdata()
        corrs_and_mask_calculations.gray_mask(image_array,mask_array)
    # get layersetwise network generator
    layersetwise_generator = clustering_method_parser(image_array,timewindow,overlap,nlayers,clustering_method_params)
    for M in layersetwise_generator:
        if layersetwise_networks_savefolder:
            # write full network with all the weights
            network_io.write_layersetwise_network(M,layersetwise_networks_savefolder)
        M = network_construction.threshold_network(M,density_params)
        for i in range(len(nnodes)):
            subgraph_classification.find_isomorphism_classes(M,nnodes[i],nlayers,None,
                                                                     allowed_aspects=isomorphism_allowed_aspects,
                                                                     aggregated_dict=aggregated_isomclass_dict[(nnodes[i],nlayers)],
                                                                     examples_dict=aggregated_example_dict[(nnodes[i],nlayers)])
            if isomorphism_class_savenames:
                network_io.write_pickle_file(dict(aggregated_isomclass_dict[(nnodes[i],nlayers)]),isomorphism_class_savenames[i])
            if isomorphism_class_examples_savenames:
                network_io.write_pickle_file(aggregated_example_dict[(nnodes[i],nlayers)],isomorphism_class_examples_savenames[i])
    return aggregated_isomclass_dict

def clustering_method_parser(image_array,timewindow,overlap,nlayers,clustering_method_params):
    method = clustering_method_params['method']
    if method == None:
        # optional params
        nan_log = clustering_method_params.get('nan_log_savename',None)
        return network_construction.yield_multiplex_network_in_layersets(image_array,nlayers,timewindow,overlap,nanlogfile=nan_log)
    elif method == 'template':
        # required params
        template_filename = clustering_method_params['template_filename']
        template_data = nib.load(template_filename)
        template_array = template_data.get_fdata()
        # optional params
        nan_log = clustering_method_params.get('nan_log_savename',None)
        calculate_consistency_while_clustering = clustering_method_params.get('calculate_consistency',False)
        consistency_save_name = clustering_method_params.get('consistency_save_name','spatial-consistency.pkl')
        n_consistency_CPUs = clustering_method_params.get('n_consistency_CPUs',5)
        return network_construction.yield_clustered_multilayer_network_in_layersets(image_array,nlayers,timewindow,overlap,n_clusters=-1,method=method,template=template_array,nanlogfile=nan_log,calculate_consistency_while_clustering=calculate_consistency_while_clustering,consistency_save_name=consistency_save_name,n_consistency_CPUs=n_consistency_CPUs)
    elif method == 'sklearn' or method == 'HAC':
        # required params
        nclusters = clustering_method_params['nclusters']
        # optional params
        nan_log = clustering_method_params.get('nan_log_savename',None)
        event_time_stamps = clustering_method_params.get('event_time_stamps',None)
        calculate_consistency_while_clustering = clustering_method_params.get('calculate_consistency',False)
        consistency_save_name = clustering_method_params.get('consistency_save_name','spatial-consistency.pkl')
        n_consistency_CPUs = clustering_method_params.get('n_consistency_CPUs',5)
        return network_construction.yield_clustered_multilayer_network_in_layersets(image_array,nlayers,timewindow,overlap,n_clusters=nclusters,method=method,template=None,nanlogfile=nan_log,event_time_stamps=event_time_stamps,calculate_consistency_while_clustering=calculate_consistency_while_clustering,consistency_save_name=consistency_save_name,n_consistency_CPUs=n_consistency_CPUs)
    elif method == 'consistency_optimized':
        # required params
        nclusters = clustering_method_params['nclusters']
        consistency_target_function = clustering_method_params['consistency_target_function']
        # optional params
        centroid_template_filename = clustering_method_params.get('centroid_template_filename',None)
        use_random_seeds = clustering_method_params.get('use_random_seeds',True)
        # choose centroid acquisition method
        if centroid_template_filename and not use_random_seeds:
            centroid_template_data = nib.load(centroid_template_filename)
            centroid_template_array = centroid_template_data.get_fdata()
            ROI_centroids, _,_ = cbc.findROICentroids(centroid_template_array,fixCentroids=True)
        elif use_random_seeds:
            centroid_template_array = None
            ROI_centroids = 'random'
        ROI_names = clustering_method_params.get('ROI_names',[])
        consistency_threshold = clustering_method_params.get('consistency_threshold',-1)
        n_consistency_iters = clustering_method_params.get('n_consistency_iters',100)
        n_consistency_CPUs = clustering_method_params.get('n_consistency_CPUs',5)
        nan_log = clustering_method_params.get('nan_log_savename',None)
        event_time_stamps = clustering_method_params.get('event_time_stamps',None)
        calculate_consistency_while_clustering = clustering_method_params.get('calculate_consistency',False)
        consistency_save_name = clustering_method_params.get('consistency_save_name','spatial-consistency.pkl')
        return network_construction.yield_clustered_multilayer_network_in_layersets(image_array,nlayers,timewindow,overlap,n_clusters=nclusters,method=method,template=centroid_template_array,nanlogfile=nan_log,event_time_stamps=event_time_stamps,ROI_centroids=ROI_centroids,ROI_names=ROI_names,consistency_threshold=consistency_threshold,consistency_target_function=consistency_target_function,f_transform_consistency=False,calculate_consistency_while_clustering=calculate_consistency_while_clustering,n_consistency_iters=n_consistency_iters,n_consistency_CPUs=n_consistency_CPUs,consistency_save_name=consistency_save_name)
    elif method=='random_balls':
        # required params
        ROI_centroids='random'
        nclusters = clustering_method_params['nclusters']
        template_filename = clustering_method_params.get('centroid_template_filename')
        template_data = nib.load(template_filename)
        template_array = template_data.get_fdata()
        # optional params
        calculate_consistency_while_clustering = clustering_method_params.get('calculate_consistency',False)
        consistency_save_name = clustering_method_params.get('consistency_save_name','spatial-consistency.pkl')
        n_consistency_CPUs = clustering_method_params.get('n_consistency_CPUs',5)
        nan_log = clustering_method_params.get('nan_log_savename',None)
        event_time_stamps = clustering_method_params.get('event_time_stamps',None)
        return network_construction.yield_clustered_multilayer_network_in_layersets(image_array,nlayers,timewindow,overlap,nclusters=nclusters,method=method,template=template_array,nanlogfile=nan_log,event_time_stamps=event_time_stamps,ROI_centroids=ROI_centroids,calculate_consistency_while_clustering=calculate_consistency_while_clustering,consistency_save_name=consistency_save_name,n_consistency_CPUs=n_consistency_CPUs)
    elif method=='craddock':
        # required params
        nclusters=clustering_method_params['nclusters']
        consistency_threshold = clustering_method_params['consistency_threshold']
        # optional params
        calculate_consistency_while_clustering = clustering_method_params.get('calculate_consistency',False)
        consistency_save_name = clustering_method_params.get('consistency_save_name','spatial-consistency.pkl')
        n_consistency_CPUs = clustering_method_params.get('n_consistency_CPUs',5)
        nan_log = clustering_method_params.get('nan_log_savename',None)
        event_time_stamps = clustering_method_params.get('event_time_stamps',None)
        return network_construction.yield_clustered_multilayer_network_in_layersets(image_array,nlayers,timewindow,overlap,nclusters=nclusters,method=method,nanlogfile=nan_log,event_time_stamps=event_time_stamps,consistency_threshold=consistency_threshold,calculate_consistency_while_clustering=calculate_consistency_while_clustering,consistency_save_name=consistency_save_name,n_consistency_CPUs=n_consistency_CPUs)
    else:
        raise NotImplementedError('Clustering method not implemented')



def isomorphism_classes_from_existing_network_files(network_folder,subnets_savefolder,
                                                    subgraph_size_dict,
                                                    allowed_aspects=[0],
                                                    intralayer_density=0.05,
                                                    interlayer_density=0.05):
    '''
    Find isomorphism classes from previously constructed and saved networks.
    Subgraph size dict can only contain one n_layers because saved nets have fixed n_layers.
    Aggregated results and examples dict only.
    '''
    sorted_filenames = sorted(os.listdir(network_folder),key=lambda s:[int(l) for l in s.split('_')])
    
    aggregated_dicts_dict = dict()
    examples_dicts_dict = dict()
    for n_layers in subgraph_size_dict:
        for n_nodes in subgraph_size_dict[n_layers]:
            aggregated_dicts_dict[(n_nodes,n_layers)] = collections.defaultdict(dict)
            examples_dicts_dict[(n_nodes,n_layers)] = dict()
    
    for filename in sorted_filenames:
        full_path = network_folder+filename
        M = network_io.read_weighted_network(full_path)
        M = network_construction.threshold_multilayer_network(M,intralayer_density,interlayer_density)
        for nlayers in subgraph_size_dict:
            for nnodes in subgraph_size_dict[nlayers]:
                subgraph_classification.find_isomorphism_classes(M,nnodes,nlayers,filename='this_file_should_not_exist',
                                                         allowed_aspects=allowed_aspects,
                                                         aggregated_dict=aggregated_dicts_dict[(nnodes,nlayers)],
                                                         examples_dict=examples_dicts_dict[(nnodes,nlayers)])
    for n_layers in subgraph_size_dict:
        for n_nodes in subgraph_size_dict[n_layers]:
            aggregated_dict_filename = subnets_savefolder+str(n_nodes)+'_'+str(n_layers)+'_agg.pickle'
            f = open(aggregated_dict_filename,'w')
            pickle.dump(aggregated_dicts_dict[(n_nodes,n_layers)],f)
            f.close()
            del(aggregated_dicts_dict[(n_nodes,n_layers)])
            
            examples_dict_filename = subnets_savefolder+'examples_'+str(n_nodes)+'_'+str(n_layers)+'.pickle'
            f = open(examples_dict_filename,'w')
            pickle.dump(examples_dicts_dict[(n_nodes,n_layers)],f)
            f.close()
            del(examples_dicts_dict[(n_nodes,n_layers)])



#################### Helpers ###############################################################################################################
    
def stringify_density(density):
    density_as_string = str(density).replace('.','')
    if len(density_as_string) < 3:
        density_as_string = '{:.2f}'.format(density).replace('.','')
    return density_as_string



if __name__ == '__main__':
    print('Import pipeline in python to access functions')
