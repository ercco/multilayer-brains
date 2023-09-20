import pymnet as pn
import pickle
import os

def write_weighted_network(M,filename,metadata):
    # First line: Multiplex or Multilayer, depending on network type
    # Second line: free-form metadata as single line, as string
    # Writes an edgelist
    # Assuming node names are strings
    with open(filename,'w') as f:
        if isinstance(M,pn.net.MultilayerNetwork):
            if isinstance(M,pn.net.MultiplexNetwork):
                f.write('# Multiplex\n')
            else:
                f.write('# Multilayer\n')
        else:
            raise TypeError('M has to be multiplex or general multilayer')
        f.write('# '+metadata+'\n')
        f.write('nodes:\n')
        for i,node in enumerate(M.iter_nodes()):
            if i == 0:
                f.write(node)
            else:
                f.write(';'+node)
        f.write('\n')
        f.write('layers:\n')
        for i,layer in enumerate(M.iter_layers()):
            if i == 0:
                f.write(str(layer))
            else:
                f.write(';'+str(layer))
        f.write('\n')
        f.write('edges:\n')
        for edge in M.edges:
            f.write(edge[0]+';'+edge[1]+';'+str(edge[2])+';'+str(edge[3])+';'+str(edge[4])+'\n')
    return
    
def read_weighted_network(filename):
    # There's no EOF in Python so we need to use clumsy flags (must iterate using 'for line in f'...)
    # The closest thing to EOF is the empty string '', but if there are empty lines in the file there's no way to know if it's the end or
    # a stripped newline
    # Edges should be the last entry in the file
    # Assuming layer names are integers (ordinally coupled multiplex)
    nodeflag = 0
    layerflag = 0
    edgeflag = 0
    with open(filename,'r') as f:
        # Read first line which should contain the type of the network (multilayer/multiplex)
        nettype = f.readline()
        if nettype[0] != '#':
            raise Exception('The first line should start with # and contain network type')
        nettype = nettype.strip(' #\n').lower()
        if nettype == 'multiplex':
            M = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
        else:
            # If network is not multiplex, we use the general multilayer class
            M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
        
        if nettype == 'multiplex':
            for line in f:
                line = line.strip()
                if line[0] == '#':
                    pass
                else:
                    if nodeflag == 1:
                        for node in line.split(';'):
                            M.add_node(node)
                        nodeflag = 0
                    if layerflag == 1:
                        for layer in line.split(';'):
                            M.add_layer(int(layer),1)
                        layerflag = 0
                    if edgeflag == 1:
                        edge = line.split(';')
                        if edge[2] == edge[3]:
                            M[edge[0],int(edge[2])][edge[1],int(edge[3])] = float(edge[4])
                        elif edge[0] == edge[1]:
                            pass
                        else:
                            raise Exception('Illegal inter-layer edges')
                    if line == 'nodes:':
                        nodeflag = 1
                    if line == 'layers:':
                        layerflag = 1
                    if line == 'edges:':
                        edgeflag = 1
        else:
            # If the network is a general multilayer one, we just need to set all the edges
            for line in f:
                line = line.strip()
                if edgeflag == 1:
                    edge = line.split(';')
                    M[edge[0],int(edge[2])][edge[1],int(edge[3])] = float(edge[4])
                if line == 'edges:':
                    edgeflag = 1
    
    return M

def write_pickle_file(object,filename):
    f = open(filename,'w')
    pickle.dump(object,f)
    f.close()
    return

def read_pickle_file(filename):
    f = open(filename,'r')
    obj = pickle.load(f)
    f.close()
    return obj

def write_layersetwise_network(M,layersetwise_networks_savefolder):
    # uses sorted layer names as the savename
    net_name = '_'.join([str(l) for l in sorted(M.iter_layers())])
    if not layersetwise_networks_savefolder[-1] == os.path.sep:
        layersetwise_networks_savefolder = layersetwise_networks_savefolder+os.path.sep
    filename = layersetwise_networks_savefolder+net_name
    write_pickle_file(M,filename)
    return

def read_consistency_data(consistency_save_stem,subject_id,run_number,clustering_method,nlayers,clustering_method_specifier=''):
    """
    Reads the ROI consistency and size data per subject, run, and clustering method.
    
    Parameters:
    -----------
    consistency_save_stem: str, parths of the consistency save path that are common to all subjects, runs, and clustering methods
    subject_id: str, subject identifier
    run_number: int
    clustering_method: str, name of the clustering method applied
    nlayers: int, the number of graphlet layers used in network construction
    clustering_method_specifier: str, optional additional token with more info on the clustering method, e.g. threshold; 
                                 used to separate different variants of the same method
                                 
    Returns:
    --------
    consistencies: dic, keys: time window indices, values: list of ROI consistencies in the given window
    sizes: dic, keys: time window indices, values: list of ROI sizes in the given window
    """
    if clustering_method_specifier == '':
        consistency_save_path = consistency_save_stem + '/' + subject_id + '/' + str(run_number) + '/' + clustering_method_specifier + '/' + str(nlayers) + '_layers' + '/spatial_consistency.pkl'
    else:
        consistency_save_path = consistency_save_stem + '/' + subject_id + '/' + str(run_number) + '/' + clustering_method_specifier + '/' + str(nlayers) + '_layers' + '/spatial_consistency_' + clustering_method_specifier + '.pkl'
    f = open(consistency_save_path,'r')
    consistency_data = pickle.load(f)
    f.close()
    consistencies = {}
    sizes = {}
    for window_number, layer in consistency_data.items():
        consistencies[window_number] = layer['consistencies']
        sizes[window_number] = layer['ROI_sizes']
    return consistencies, sizes

def pool_consistency_data(consistency_save_stem,subject_ids,run_numbers,clustering_methods,nlayers,clustering_method_specifiers=''):
    """
    Reads the ROI consistency and size of given set of subjects, runs, and clustering methods and pools them over subjects, runs,
    and time windows (layers).
    
    Parameters:
    -----------
    consistency_save_stem: str, parths of the consistency save path that are common to all subjects, runs, and clustering methods
    subject_ids: iterable of str, subject identifiers
    run_numbers: iterable of int
    clustering_methods: iterable of str, names of the clustering method applied
    nlayers: int, the number of graphlet layers used in network construction
    clustering_method_specifiers: iterable of str, optional additional token with more info on the clustering method, e.g. threshold; 
                                 used to separate different variants of the same method
                                 
    Returns:
    --------
    pooled_consistencies: list of lists, consistencies per each clustering method pooled over other parameters
    pooled_sizes: list of lists, sizes per each clustering method pooled over other parameters
    """
    if clustering_method_specifiers == '':
        clustering_method_specifiers = ['' for clustering_method in clustering_methods]
    pooled_consistencies = [[] for clustering_method in clustering_methods] # this is a clustering methods x ROIs list (ROIs pooled over subjects, layers, and runs)
    pooled_sizes = [[] for clustering_method in clustering_methods]
    for i, (clustering_method,clustering_method_specifier) in enumerate(zip(clustering_methods,clustering_method_specifiers)):
        for subject_id in subject_ids:
            for run_number in run_numbers:
                consistencies, sizes = read_consistency_data(consistency_save_stem,subject_id,run_number,clustering_method,nlayers,clustering_method_specifier='')
                pooled_consistencies[i].extend(consistencies.values())
                pooled_sizes[i].extend(sizes.values())
    return pooled_consistencies, pooled_values


    
        

    