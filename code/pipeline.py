import nibabel as nib
import os
import datetime
import collections
import pickle

import network_construction
import network_io
import subgraph_classification
import corrs_and_mask_calculations

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
	layersetwise_networks_savefolder : str, folder for saving the generated networks
		(None = no saving)
	log_savename : str, appends successful completion info to this file (None = no logging)
	
	Clustering:
	TODO explanation
	
	Returns:
	if nnodes is an int:
		dict of dicts, first level of keys is isomorphism classes as tuples (complete
		invariants) and second level of keys is (ordered) layersets as tuples
		return_dict[compinvariant][layerset] = frequency
	if nnodes is a list:
		dict of dicts of dicts, first level of keys is nnodes,nlayeyers pairs as tuples,
		second and third level as in the nnodes is an int case
		return_dict[(nnodes,nlayers)][compinvariant][layerset] = frequency
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
    		network_io.write_pickle_file(aggregated_isomclass_dict[(nnodes[i],nlayers)],isomorphism_class_savenames[i])
    		network_io.write_pickle_file(aggregated_example_dict[(nnodes[i],nlayers)]isomorphism_class_examples_savenames[i])

def clustering_method_parser(image_array,timewindow,overlap,nlayers,clustering_method_params):
	method = clustering_method_params['method']
	if method == None:
		# voxel-level
		pass
    elif method == "template":
    	pass
    elif method == "sklearn" or method == "HAC":
    	pass
    elif method == "consistency_growth":
    	pass
    else:
    	raise NotImplementedError('Clustering method not implemented')
    

def isomorphism_classes_from_nifti(nii_data_filename, subj_id, run_number,
                       timewindow, overlap, intralayer_density, interlayer_density,
                       subgraph_size_dict,
                       allowed_aspects=[0],
                       use_aggregated_dict=True,
                       create_examples_dict=True,
                       clustering_method=None, mask_or_template_filename=None, mask_or_template_name=None,
                       number_of_clusters = 100,
                       data_folder=None,
                       preprocess_level_folder=None,
                       template_folder=None,
                       relative_nii_path=False, relative_template_path=False,
                       event_time_stamps=None):
    '''
    Usage:
    nii_data_filename : string, filename for nifti file which contains the 4D data matrix (three spatial and one temporal)
    subj_id : string, id for subject for saving (e.g. 'a5n')
    run_number : int, run number for saving (e.g. 2)
    timewindow : int, timewindow size in data points (e.g. 100)
    overlap : int, number of overlapping data points between time windows (e.g. 0 for no overlap)
    intralayer_density : float, edge density (0<density<1) for thresholding correlation networks within layers
    interlayer_density : float, edge density (0<density<1) for thresholding networks between layers, only used if net is
                        _not_ a multiplex network (e.g. template networks and voxel-level networks are multiplex)
    subgraph_size_dict : dict, with number of layers as key and number of nodes as value (in tuple)
                        e.g. {2:(2,3), 3:(2,)} finds all subgraphs with sizes:
                        (2 layers, 2 nodes)
                        (2 layers, 3 nodes)
                        (3 layers, 2 nodes)
    allowed_aspects : list, which aspects are allowed to be permuted when calculating isomorphism classes
                        [0] = vertex-isomorphic classes
                        [0,1] = vertex-layer-isomorphic classes
    use_aggregated_dicts : bool, whether to save results as pickle dict (True) or one-ine-per-subgraph text file (False)
                        USE TRUE!
    create_examples_dict : bool, whether to save an example network from each isomorphism class or not
    clustering_method : string or None, 'template' or 'sklearn' or None
                        'template' = use preconstruted template
                        'sklearn' = use sklearn HAC for each layer individually
                        None = voxel-level analysis
    mask_or_template_filename : string, if clustering_method == 'template', then this will be used as template, otherwise it will
                        be used as a mask (to remove e.g. non-gray matter)
    mask_or_template_name : string, for saving (e.g. 'HarvardOxford')
    number_of_clusters : int, only used if clustering_method == 'sklearn'
    data_folder : string, location of data folder if desired, to be used with relative_nii_path=True
    preprocess_level_folder : string, save location for results (file structure for results will be created under this), IMPORTANT
    template_folder : string, location of template folder if desired, to be used with relative_template_path=True
    relative_nii_path : bool, True if nii_data_filename should be added to data_folder to reach the nifti file (allows nii_data_filename
                        to be given as a relative path starting from data_folder)
    relative_template_path : bool, same as relative_nii_path but for template
    event_time_stamps : list, can contain time stamps where event change happens in the data, to create layers according to them in sklearn
                        clustering (available in sklearn clustering so far, not in template clustering)
    
    Recommendations:
    Do not use data_folder or template_folder and set relative_nii_path=relative_template_path=False.
    There is nothing wrong with using those, but it is simpler to just give nii_data_filename and mask_or_template_filename as complete path,
    e.g. nii_data_filename='/a/b/c/data_file.nii'.
    Preprocess level folder is the location where everything is saved, give a complete path.
    The results will be saved as:
    preprocess_level_folder
        - subj_id
            - run_number
                - clustering_type
                    - mask_or_template_name
                        (- number_of_clusters if using sklearn, else this level does not exist)
                            - net_X
                            - subnets_X
    where X is an identifier containing timewindow, overlap, creation date, and for subnets also densities.
    Net_X will be a folder which contains layersetwise unthresholded networks, subnets_X will be a folder which contains subnets files
    named after nnodes_nlayers. Exact form depends on use_aggregated_dicts. If it is true, these will be pickle files, otherwise text files.
    Subnets_X will also contain example networks in dicts in pickle files, if create_examples_dict==True.
    
    For examples see sections below.
    '''
    assert(0<intralayer_density<1 and 0<interlayer_density<1)
    assert(isinstance(timewindow,int))
    assert(isinstance(overlap,int))
    # masking required for every file, to remove voxels outside of the brain at the very least
    assert(mask_or_template_filename is not None and mask_or_template_name is not None)
    
    if relative_nii_path:
        nii_data_filename = data_folder+nii_data_filename
    if relative_template_path and mask_or_template_filename is not None:
        mask_or_template_filename = template_folder+mask_or_template_filename
    
    if clustering_method == None:
        voxel_level_folder = preprocess_level_folder+subj_id+'/'+str(run_number)+'/voxel_level/'+mask_or_template_name+'/'
        if not os.path.exists(voxel_level_folder):
            os.makedirs(voxel_level_folder)
    elif clustering_method == 'template':
        assert(mask_or_template_filename is not None and mask_or_template_name is not None)
        cluster_level_folder = preprocess_level_folder+subj_id+'/'+str(run_number)+'/template_clustering/'+mask_or_template_name+'/'
        if not os.path.exists(cluster_level_folder):
            os.makedirs(cluster_level_folder)
    elif clustering_method == 'sklearn':
        cluster_level_folder = preprocess_level_folder+subj_id+'/'+str(run_number)+'/sklearn_hac/'+mask_or_template_name+'/'+str(number_of_clusters)+'/'
        if not os.path.exists(cluster_level_folder):
            os.makedirs(cluster_level_folder)
    else:
        raise NotImplementedError('Not implemented')
    
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    network_identifier = str(timewindow)+'_'+str(overlap)+'_'+current_time
    
    intralayer_density_as_string = str(intralayer_density).replace('.','')
    if len(intralayer_density_as_string) < 3:
        intralayer_density_as_string = '{:.2f}'.format(intralayer_density).replace('.','')
        
    if clustering_method == 'sklearn':
        interlayer_density_as_string = str(interlayer_density).replace('.','')
        if len(interlayer_density_as_string) < 3:
            interlayer_density_as_string = '{:.2f}'.format(interlayer_density).replace('.','')
    
    if clustering_method == None:
        subnets_folder = voxel_level_folder+'subnets_'+network_identifier+'_'+intralayer_density_as_string+'/'
    elif clustering_method == 'template':
        subnets_folder = cluster_level_folder+'subnets_'+network_identifier+'_'+intralayer_density_as_string+'/'
    elif clustering_method == 'sklearn':
        subnets_folder = cluster_level_folder+'subnets_'+network_identifier+'_'+intralayer_density_as_string+'_'+interlayer_density_as_string+'/'
    else:
        raise NotImplementedError('Not implemented')
    os.makedirs(subnets_folder)
    
    # load data
    img = nib.load(nii_data_filename)
    imgdata = img.get_fdata()
    # load template if template clustering is used
    if clustering_method == 'template':
        templateimg = nib.load(mask_or_template_filename)
        template = templateimg.get_fdata()
    # apply mask to sklearn clustering, if mask is given
    elif clustering_method == 'sklearn':
        maskimg = nib.load(mask_or_template_filename)
        mask = maskimg.get_fdata()
        corrs_and_mask_calculations.gray_mask(imgdata,mask)
    # apply mask to voxel-level, if mask is given
    elif clustering_method == None:
        maskimg = nib.load(mask_or_template_filename)
        mask = maskimg.get_fdata()
        corrs_and_mask_calculations.gray_mask(imgdata,mask)
        
    # create aggregated dicts and example dicts
    # saved in dicts with key (nnodes,nlayers) - element is correct dict if relevant parameter is True, None otherwise
    aggregated_dicts_dict = dict()
    examples_dicts_dict = dict()
    for n_layers in subgraph_size_dict:
        for n_nodes in subgraph_size_dict[n_layers]:
            if use_aggregated_dict:
                aggregated_dicts_dict[(n_nodes,n_layers)] = collections.defaultdict(dict)
            else:
                aggregated_dicts_dict[(n_nodes,n_layers)] = None
            if create_examples_dict:
                examples_dicts_dict[(n_nodes,n_layers)] = dict()
            else:
                examples_dicts_dict[(n_nodes,n_layers)] = None
        
    for n_layers in subgraph_size_dict:
        if clustering_method == None:
            layersetwise_save_location = voxel_level_folder+'net_'+network_identifier+'/'+str(n_layers)+'_layers/'
        elif clustering_method == 'template' or clustering_method == 'sklearn':
            layersetwise_save_location = cluster_level_folder+'net_'+network_identifier+'/'+str(n_layers)+'_layers/'
        else:
            raise NotImplementedError('Not implemented')
        os.makedirs(layersetwise_save_location)
        
        # Generators for getting layersetwise networks
        if clustering_method == None:
            nanlogfile = voxel_level_folder+'net_'+network_identifier+'/'+str(n_layers)+'_layers_nanlog.txt'
        elif clustering_method == 'template' or clustering_method == 'sklearn':
            nanlogfile = cluster_level_folder+'net_'+network_identifier+'/'+str(n_layers)+'_layers_nanlog.txt'
        else:
            raise NotImplementedError('Not implemented')
        
        if clustering_method == None:
            layersetwise_generator = network_construction.yield_multiplex_network_in_layersets(
            imgdata,n_layers,timewindow,overlap,nanlogfile=nanlogfile)
        elif clustering_method == 'template':
            layersetwise_generator = network_construction.yield_clustered_multilayer_network_in_layersets(
            imgdata,n_layers,timewindow,overlap,n_clusters=-1,method='template',template=template,nanlogfile=nanlogfile)
        elif clustering_method == 'sklearn':
            layersetwise_generator = network_construction.yield_clustered_multilayer_network_in_layersets(
            imgdata,n_layers,timewindow,overlap,number_of_clusters,method='sklearn',template=None,nanlogfile=nanlogfile,
            event_time_stamps=event_time_stamps)
        else:
            raise NotImplementedError('Not implemented')
        
        for M in layersetwise_generator:
            layerset_net_filename = '_'.join([str(l) for l in sorted(M.iter_layers())])
            metadata = 'Origin: '+nii_data_filename+' Layers: '+layerset_net_filename+' Timewindow: '+str(timewindow)+' Overlap: '+str(overlap)+' Created_on: '+current_time
            network_io.write_weighted_network(M,layersetwise_save_location+layerset_net_filename,metadata)
            
            if clustering_method == None:
                M = network_construction.threshold_multiplex_network(M,intralayer_density)
            elif clustering_method == 'template':
                M = network_construction.threshold_multiplex_network(M,intralayer_density)
            elif clustering_method == 'sklearn':
                M = network_construction.threshold_multilayer_network(M,intralayer_density,interlayer_density)
            
            for n_nodes in subgraph_size_dict[n_layers]:
                if use_aggregated_dict:
                    subnets_filename = 'this_file_should_not_exist'
                else:
                    subnets_filename = subnets_folder+str(n_nodes)+'_'+str(n_layers)
                subgraph_classification.find_isomorphism_classes(M,n_nodes,n_layers,subnets_filename,
                                                                     allowed_aspects=allowed_aspects,
                                                                     aggregated_dict=aggregated_dicts_dict[(n_nodes,n_layers)],
                                                                     examples_dict=examples_dicts_dict[(n_nodes,n_layers)])            
            del(M)
            
        if use_aggregated_dict:
            for n_nodes in subgraph_size_dict[n_layers]:
                aggregated_dict_filename = subnets_folder+str(n_nodes)+'_'+str(n_layers)+'_agg.pickle'
                f = open(aggregated_dict_filename,'w')
                pickle.dump(aggregated_dicts_dict[(n_nodes,n_layers)],f)
                f.close()
                del(aggregated_dicts_dict[(n_nodes,n_layers)])
            
        if create_examples_dict:
            for n_nodes in subgraph_size_dict[n_layers]:
                examples_dict_filename = subnets_folder+'examples_'+str(n_nodes)+'_'+str(n_layers)+'.pickle'
                f = open(examples_dict_filename,'w')
                pickle.dump(examples_dicts_dict[(n_nodes,n_layers)],f)
                f.close()
                del(examples_dicts_dict[(n_nodes,n_layers)])
            
        end_time_for_n_layers = datetime.datetime.now().replace(microsecond=0).isoformat()
        
        if clustering_method == None:
            log_file_name = voxel_level_folder+'net_'+network_identifier+'/'+'log.txt'
        elif clustering_method == 'template' or clustering_method == 'sklearn':
            log_file_name = cluster_level_folder+'net_'+network_identifier+'/'+'log.txt'
        else:
            raise NotImplementedError('Not implemented')
        with open(log_file_name,'a+') as f:
            f.write(str(n_layers)+'_layers...Done at '+end_time_for_n_layers+'\n')
    
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
