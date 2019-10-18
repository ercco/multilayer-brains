import pymnet as pn
import numpy as np

import corrs_and_mask_calculations
import clustering



#################### Network creating functions ############################################################################################

def make_multiplex(imgdata,timewindow=100,overlap=0,nanlogfile=None):
    # ignore remainder of timepoints
    # maybe use nan-inclusive adjacency matrix to weed out masked voxels?
    #xdim = imgdata.shape[0]
    #ydim = imgdata.shape[1]
    #zdim = imgdata.shape[2]
    k = get_number_of_layers(imgdata.shape,timewindow,overlap)
    start_times,end_times = get_start_and_end_times(k,timewindow,overlap)
    
    M = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
    
    for tw_no in range(k):
        #start = tw_no*(timewindow-overlap)
        #end = timewindow + tw_no*(timewindow-overlap)
        A,voxellist = corrs_and_mask_calculations.make_adjacency_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],exclude_masked=True)
        for ii in range(A.shape[0]):
            # magic tricks to get true index from linearized form (disclaimer: not actual magic)
            # node1 = str((ii//(ydim*zdim),(ii%(ydim*zdim))//zdim,(ii%(ydim*zdim))%zdim))
            node1 = str(voxellist[ii])
            for jj in range(ii+1,A.shape[1]):
                #node2 = str((jj//(ydim*zdim),(jj%(ydim*zdim))//zdim,(jj%(ydim*zdim))%zdim))
                node2 = str(voxellist[jj])
                # only get unmasked voxels
                if not np.isnan(A[ii,jj]):
                    M[node1,tw_no][node2,tw_no] = A[ii,jj]
                else:
                    if nanlogfile != None:
                        with open(nanlogfile,'a+') as f:
                            f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                    else:
                        print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
    return M
    
def yield_multiplex_network_in_layersets(imgdata,layerset_size,timewindow=100,overlap=0,nanlogfile=None):
    # Yields small networks which contain layerset_size layers each,
    # starting from the beginning of imgdata
    # This is for running subgraph enumeration without having to load entire network at once (only create one relevant subnet at a time)
    assert(layerset_size>0 and isinstance(layerset_size,int))
    k = get_number_of_layers(imgdata.shape,timewindow,overlap)
    start_times,end_times = get_start_and_end_times(k,timewindow,overlap)
    # get all consecutive subsequences of length layerset_size in the range of all layers:
    layersets = zip(*(range(k)[ii:] for ii in range(layerset_size)))
    for layerset in layersets:
        
        M = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
        
        for tw_no in layerset:
            A,voxellist = corrs_and_mask_calculations.make_adjacency_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],exclude_masked=True)
            for ii in range(A.shape[0]):
                node1 = str(voxellist[ii])
                for jj in range(ii+1,A.shape[1]):
                    node2 = str(voxellist[jj])
                    if not np.isnan(A[ii,jj]):
                        M[node1,tw_no][node2,tw_no] = A[ii,jj]
                    else:
                        if nanlogfile != None:
                            with open(nanlogfile,'a+') as f:
                                f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                        else:
                            print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
        yield M
        del(M)
    
def make_clustered_multilayer(imgdata,timewindow=100,overlap=0,n_clusters=100,method='sklearn',template=None,nanlogfile=None):
    '''
    Possible methods:
    'sklearn' : hierarchical clustering from sklearn, different for each time window
    'template' : preconstructed clustering, same for each time window (requires parameter template : 3d ndarray with the same shape
        as imgdata.shape[0:3] where 0 denotes masked voxel and other values denote cluster identity
        Using this, n_clusters is ignored
    '''
    k = get_number_of_layers(imgdata.shape,timewindow,overlap)
    start_times,end_times = get_start_and_end_times(k,timewindow,overlap)
    
    if method == 'sklearn':
        M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
        previous_voxels_in_clusters = dict()
        for tw_no in range(k):
            # Create new object and make voxels_in_clusters refer to it (doesn't change previous_voxels_in_clusters)
            voxels_in_clusters = dict()
            #start = tw_no*(timewindow-overlap)
            #end = timewindow + tw_no*(timewindow-overlap)
            model,voxellist = clustering.cluster_timewindow_scikit(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],n_clusters=n_clusters)
            for ii,label in enumerate(model.labels_):
                voxels_in_clusters.setdefault(label,[]).append(voxellist[ii])
            R = calculate_cluster_correlation_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],voxels_in_clusters)
            for ii in range(R.shape[0]):
                node1 = str(voxels_in_clusters[ii])
                for jj in range(ii+1,R.shape[1]):
                    node2 = str(voxels_in_clusters[jj])
                    if not np.isnan(R[ii,jj]):
                        M[node1,tw_no][node2,tw_no] = R[ii,jj]
                    else:
                        if nanlogfile != None:
                            with open(nanlogfile,'a+') as f:
                                f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                        else:
                            print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
            for cluster_number in voxels_in_clusters:
                for previous_cluster_number in previous_voxels_in_clusters:
                    cluster_overlap = get_overlap(set(voxels_in_clusters[cluster_number]),set(previous_voxels_in_clusters[previous_cluster_number]))
                    M[str(previous_voxels_in_clusters[previous_cluster_number]),tw_no-1][str(voxels_in_clusters[cluster_number]),tw_no] = cluster_overlap
            previous_voxels_in_clusters = voxels_in_clusters # reference to the same object
            
    elif method == 'template':
        M = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
        voxels_in_clusters = get_voxels_in_clusters(template)
        for tw_no in range(k):
            R = calculate_cluster_correlation_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],voxels_in_clusters)
            for ii in range(R.shape[0]):
                node1 = str(voxels_in_clusters[ii])
                for jj in range(ii+1,R.shape[1]):
                    node2 = str(voxels_in_clusters[jj])
                    if not np.isnan(R[ii,jj]):
                        M[node1,tw_no][node2,tw_no] = R[ii,jj]
                    else:
                        if nanlogfile != None:
                            with open(nanlogfile,'a+') as f:
                                f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                        else:
                            print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
    
    else:
        raise NotImplementedError('This clustering not implemented')
    
    return M
    
def yield_clustered_multilayer_network_in_layersets(imgdata,layerset_size,timewindow=100,overlap=0,n_clusters=100,
                                                    method='sklearn',template=None,nanlogfile=None,
                                                    event_time_stamps=None):
    # If event_time_stamps is specified, then they are used to compute start_times, end_times and k (and timewindow and overlap are ignored).
    # Otherwise, timewindow and overlap are used ot compute start_times, end_times and k.
    if event_time_stamps == None:
        k = get_number_of_layers(imgdata.shape,timewindow,overlap)
        start_times,end_times = get_start_and_end_times(k,timewindow,overlap)
    else:
        assert isinstance(event_time_stamps,list)
        k = len(event_time_stamps) + 1
        start_times = [0] + event_time_stamps
        end_times = event_time_stamps + [imgdata.shape[3]]
    layersets = zip(*(range(k)[ii:] for ii in range(layerset_size)))
    
    if method == 'sklearn':
        voxels_in_clusters_by_timewindow = dict()
        for layerset in layersets:
            M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
            previous_voxels_in_clusters = dict()
            for tw_no in layerset:
                if not tw_no in voxels_in_clusters_by_timewindow:
                    voxels_in_clusters = dict()
                    model,voxellist = clustering.cluster_timewindow_scikit(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],n_clusters=n_clusters)
                    for ii,label in enumerate(model.labels_):
                        voxels_in_clusters.setdefault(label,[]).append(voxellist[ii])
                    voxels_in_clusters_by_timewindow[tw_no] = voxels_in_clusters
                else:
                    voxels_in_clusters = voxels_in_clusters_by_timewindow[tw_no]
                R = calculate_cluster_correlation_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],voxels_in_clusters)
                for ii in range(R.shape[0]):
                    node1 = str(voxels_in_clusters[ii])
                    for jj in range(ii+1,R.shape[1]):
                        node2 = str(voxels_in_clusters[jj])
                        if not np.isnan(R[ii,jj]):
                            M[node1,tw_no][node2,tw_no] = R[ii,jj]
                        else:
                            if nanlogfile != None:
                                with open(nanlogfile,'a+') as f:
                                    f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                            else:
                                print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                for cluster_number in voxels_in_clusters:
                    for previous_cluster_number in previous_voxels_in_clusters:
                        cluster_overlap = get_overlap(set(voxels_in_clusters[cluster_number]),set(previous_voxels_in_clusters[previous_cluster_number]))
                        M[str(previous_voxels_in_clusters[previous_cluster_number]),tw_no-1][str(voxels_in_clusters[cluster_number]),tw_no] = cluster_overlap
                previous_voxels_in_clusters = voxels_in_clusters # reference to the same object
            del(voxels_in_clusters_by_timewindow[min(voxels_in_clusters_by_timewindow)])
            yield M
            del(M)
    
    elif method == 'template':
        voxels_in_clusters = get_voxels_in_clusters(template)
        for layerset in layersets:
            M = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
            for tw_no in layerset:
                R = calculate_cluster_correlation_matrix(imgdata[:,:,:,start_times[tw_no]:end_times[tw_no]],voxels_in_clusters)
                for ii in range(R.shape[0]):
                    node1 = str(voxels_in_clusters[ii])
                    for jj in range(ii+1,R.shape[1]):
                        node2 = str(voxels_in_clusters[jj])
                        if not np.isnan(R[ii,jj]):
                            M[node1,tw_no][node2,tw_no] = R[ii,jj]
                        else:
                            if nanlogfile != None:
                                with open(nanlogfile,'a+') as f:
                                    f.write('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
                            else:
                                print('NaN correlation at nodes '+node1+', '+node2+' at timewindow '+str(tw_no)+'\n')
            yield M
            del(M)
    else:
        raise NotImplementedError('Not implemented')
    
        
def make_specific_timewindows_network_sklearn(imgdata,start_times,end_times,layer_labels,n_clusters=100,nanlogfile=None):
    # start_times = vector of timewindow start times
    # end_times = vector of timewindow end times
    # NB! returns also the acquired models and voxellists in a dict with keys (start_time,end_time)
    # (start_time,end_time) = (model,voxellist)
    assert(len(start_times) == len(end_times))
    assert(len(start_times) == len(layer_labels))
    M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    previous_voxels_in_clusters = dict()
    models = dict()
    layer_relabeling = dict()
    for ii in range(len(start_times)):
        layer_relabeling[ii] = layer_labels[ii]
        voxels_in_clusters = dict()
        model,voxellist = clustering.cluster_timewindow_scikit(imgdata[:,:,:,start_times[ii]:end_times[ii]],n_clusters=n_clusters)
        for jj,label in enumerate(model.labels_):
            voxels_in_clusters.setdefault(label,[]).append(voxellist[jj])
        models[(start_times[ii],end_times[ii])] = (model,voxellist)
        R = calculate_cluster_correlation_matrix(imgdata[:,:,:,start_times[ii]:end_times[ii]],voxels_in_clusters)
        for kk in range(R.shape[0]):
            node1 = str(voxels_in_clusters[kk])
            for ll in range(kk+1,R.shape[1]):
                node2 = str(voxels_in_clusters[ll])
                if not np.isnan(R[kk,ll]):
                    M[node1,ii][node2,ii] = R[kk,ll]
                else:
                    if nanlogfile != None:
                        with open(nanlogfile,'a+') as f:
                            f.write('NaN correlation at nodes '+node1+', '+node2+' at time '+str(start_times[ii])+', '+str(end_times[ii])+'\n')
                    else:
                        print('NaN correlation at nodes '+node1+', '+node2+' at time '+str(start_times[ii])+', '+str(end_times[ii])+'\n')
        for cluster_number in voxels_in_clusters:
            for previous_cluster_number in previous_voxels_in_clusters:
                cluster_overlap = get_overlap(set(voxels_in_clusters[cluster_number]),set(previous_voxels_in_clusters[previous_cluster_number]))
                M[str(previous_voxels_in_clusters[previous_cluster_number]),ii-1][str(voxels_in_clusters[cluster_number]),ii] = cluster_overlap
        previous_voxels_in_clusters = voxels_in_clusters # reference to the same object
    # relabel layers (to avoid having every net have labels 0...k)
    M = pn.transforms.relabel(M,layerNames=layer_relabeling)
    return M,models
    
    
    
#################### Helpers ###############################################################################################################
    
def get_number_of_layers(imgdata_shape,timewindow,overlap):
    return int(1 + np.floor((imgdata_shape[3]-timewindow)/float((timewindow-overlap))))
    
def get_start_and_end_times(k,timewindow,overlap):
    # start time = time point where time window starts
    # end time = the first time point after time window ends, i.e. NOT part of time window itself
    # This enables slicing imgdata (numpy ndarray) correctly, e.g.
    # start = 0, end = 100
    # => imgdata[:,:,:,start:end] contains time points 0...99 (100 time points in total)
    start_times = []
    end_times = []
    for tw_no in range(k):
        start_times.append(tw_no*(timewindow-overlap))
        end_times.append(timewindow + tw_no*(timewindow-overlap))
    return start_times,end_times
    
def get_overlap(set1,set2):
    # Overlap between sets of voxels.
    # Defined as cardinality of intersection divided by cardinality of union.
    # Thus is always between 0 and 1, where 1 is complete overlap.
    return float(len(set1 & set2))/float(len(set1 | set2))
    
def get_voxels_in_clusters(template):
    assert(template is not None)
    cluster_ids = np.unique(template)
    cluster_ids.sort()
    if not 0 in cluster_ids:
        raise Exception('Masked space should be denoted with 0')
    cluster_ids = np.delete(cluster_ids,0)
    voxels_in_clusters = dict()
    # new_cluster_id goes 0...k where k = number of clusters - 1
    for new_cluster_id,cluster_id in enumerate(cluster_ids):
        voxels_in_clusters[new_cluster_id] = list(zip(*np.where(template==cluster_id)))
    return voxels_in_clusters
        
def calculate_cluster_correlation_matrix(imgdata_tw,voxels_in_clusters):
    # voxels_in_clusters : dict with keys 0...k (assuming labels are ints starting from 0)
    # Returns correlation between average time series, NOT average correlation between all pairwise time series!!
    R = np.ndarray((len(voxels_in_clusters),len(voxels_in_clusters)))
    cluster_timeseries = dict()
    for cluster_number in voxels_in_clusters:
        for voxel in voxels_in_clusters[cluster_number]:
            voxel_timeseries = imgdata_tw[voxel]
            cluster_timeseries[cluster_number] = cluster_timeseries.get(cluster_number,np.zeros(voxel_timeseries.shape)) + voxel_timeseries
    for cluster_number in cluster_timeseries:
        cluster_timeseries[cluster_number] = cluster_timeseries[cluster_number] / float(len(voxels_in_clusters[cluster_number]))
    # loops twice over 0...k, maybe think of a better way?
    for cluster_number_1 in voxels_in_clusters:
        for cluster_number_2 in voxels_in_clusters:
            R[cluster_number_1,cluster_number_2] = np.corrcoef(cluster_timeseries[cluster_number_1],cluster_timeseries[cluster_number_2])[0][1]
    return R



#################### Thresholding ##########################################################################################################

def threshold_network(M,density_params):
    # threshold network according to params in density_params dict
    if 'intralayer_density' in density_params and 'interlayer_density' in density_params:
        return threshold_multilayer_network(M,density_params['intralayer_density'],density_params['interlayer_density'],density_params.get('replace_interlayer_weights_with_ones',True))
    elif 'intra_avg_degree' in density_params and 'inter_avg_degree' in density_params:
        raise NotImplementedError('Thresholding method not implemented')
    else:
        raise NotImplementedError('Thresholding method not implemented')

def threshold_multiplex_network(M,density=0.05):
    # works only for multiplex networks
    # assumes that edge weights are NUMBERS not NANs!!!
    # assumes multiplex network (1 aspect)
    # modifies original M! FIX !!!
    for layer in M.iter_layers():
        ordered_edges = sorted(M.A[layer].edges,key=lambda w:w[2],reverse=True)
        assert(all([w[2] is not np.nan for w in ordered_edges]))
        max_edges = (len(M.A[layer])*(len(M.A[layer])-1))/2.0
        thresholded_edges = ordered_edges[0:int(np.floor(max_edges*density))]
        for edge in ordered_edges:
            if edge in thresholded_edges:
                M.A[layer][edge[0],edge[1]] = 1
            else:
                M.A[layer][edge[0],edge[1]] = 0
    return M

def threshold_multilayer_network(M,intralayer_density=0.05,interlayer_density=0.05,replace_interlayer_weights_with_ones=True):
    layers = sorted(list(M.get_layers()))
    assert(layers == list(range(min(layers),max(layers)+1)))
    #pairwise_layersets = zip(*(range(len(layers))[ii:] for ii in range(2)))
    
    # threshold first layer in network:
    first_layer = layers[0]
    ordered_edges = sorted(pn.subnet(M,M.iter_nodes(),[first_layer]).edges,key=lambda w:w[4],reverse=True)
    assert(all([w[4] is not np.nan for w in ordered_edges]))
    max_edges = (len(list(M.iter_nodes(first_layer)))*(len(list(M.iter_nodes(first_layer)))-1))/2.0
    thresholded_edges = ordered_edges[0:int(np.floor(max_edges*intralayer_density))]
    for edge in ordered_edges:
        if edge in thresholded_edges:
            M[edge[0],edge[1],edge[2],edge[3]] = 1
        else:
            M[edge[0],edge[1],edge[2],edge[3]] = 0
            
    # threshold subsequent layers and intralayer networks between current layer and the previous one
    del(layers[0])
    for layer in layers:
        ordered_edges = sorted(pn.subnet(M,M.iter_nodes(),[layer]).edges,key=lambda w:w[4],reverse=True)
        assert(all([w[4] is not np.nan for w in ordered_edges]))
        max_edges = (len(list(M.iter_nodes(layer)))*(len(list(M.iter_nodes(layer)))-1))/2.0
        thresholded_edges = ordered_edges[0:int(np.floor(max_edges*intralayer_density))]
        for edge in ordered_edges:
            if edge in thresholded_edges:
                M[edge[0],edge[1],edge[2],edge[3]] = 1
            else:
                M[edge[0],edge[1],edge[2],edge[3]] = 0
        interlayer_ordered_edges = sorted([edge for edge in pn.subnet(M,M.iter_nodes(),[layer,layer-1]).edges if edge[2] != edge[3]],key=lambda w:w[4],reverse=True)
        interlayer_max_edges = len(list(M.iter_nodes(layer)))*len(list(M.iter_nodes(layer-1)))
        interlayer_thresholded_edges = interlayer_ordered_edges[0:int(np.floor(interlayer_max_edges*interlayer_density))]
        for edge in interlayer_ordered_edges:
            if edge in interlayer_thresholded_edges:
                if replace_interlayer_weights_with_ones:
                    M[edge[0],edge[1],edge[2],edge[3]] = 1
            else:
                M[edge[0],edge[1],edge[2],edge[3]] = 0
                
    return M
    
    
    
#################### Multiplex check #######################################################################################################
    
def is_multiplex(M):
    if isinstance(M,pn.MultiplexNetwork):
        return True
    for edge in list(M.edges):
        if edge[0] != edge[1] and edge[2] != edge[3]:
            return False
    nodes_on_layers = dict()
    for nl in list(M.iter_node_layers()):
        nodes_on_layers[nl[1]] = nodes_on_layers.get(nl[1],set()).union([nl[0]])
    layers = list(M.iter_layers())
    for layer1 in layers:
        shared_node_found = False
        for layer2 in layers:
            if layer1 != layer2:
                if len(nodes_on_layers[layer1].intersection(nodes_on_layers[layer2])) > 0:
                    shared_node_found = True
        if not shared_node_found and len(layers) > 1:
            return False
    return True
