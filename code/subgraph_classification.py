import pymnet as pn
import collections

def find_isomorphism_classes(M,nnodes,nlayers,filename,
                             allowed_aspects='all',
                             aggregated_dict=None,
                             examples_dict=None):
    # aggregated_dict needs to be collections.defaultdict with default value dict
    if isinstance(aggregated_dict,collections.defaultdict):
        pn.sampling.esu.sample_multilayer_subgraphs_esu(M,lambda subnet: add_subnet_to_aggregated_dict(M,subnet[0],subnet[1],aggregated_dict,examples_dict,allowed_aspects),nnodes=nnodes,nlayers=nlayers)
    else:
        # Appends each subnet to file, if file exists will append to existing data
        writefile = open(filename,'a+')
        pn.sampling.esu.sample_multilayer_subgraphs_esu(M,lambda subnet: save_subnet_with_complete_invariant(M,subnet[0],subnet[1],writefile,examples_dict,allowed_aspects),nnodes=nnodes,nlayers=nlayers)
        writefile.close()
    return
    
def save_subnet_with_complete_invariant(M,nodes,layers,writefile,examples_dict,allowed_aspects):
    if allowed_aspects != 'all':
        # relabel layers to 0...k
        sorted_layers = tuple(sorted(layers))
        relabeling = dict()
        for ii,orig_layer in enumerate(sorted_layers):
            relabeling[orig_layer] = ii
        subnet = pn.transforms.relabel(pn.subnet(M,nodes,layers),layerNames=relabeling)
        complete_invariant = pn.get_complete_invariant(subnet,allowed_aspects=allowed_aspects)
    else:
        subnet = pn.subnet(M,nodes,layers)
        complete_invariant = pn.get_complete_invariant(subnet,allowed_aspects=allowed_aspects)
    writefile.write(str(nodes)+'\t'+str(layers)+'\t'+str(complete_invariant)+'\n')
    if isinstance(examples_dict,dict):
        if complete_invariant not in examples_dict:
            examples_dict[complete_invariant] = subnet
    return
    
def add_subnet_to_aggregated_dict(M,nodes,layers,aggregated_dict,examples_dict,allowed_aspects):
    # modifies aggregated_dict directly
    sorted_layers = tuple(sorted(layers))
    if allowed_aspects != 'all':
        # relabel layers to 0...k
        relabeling = dict()
        for ii,orig_layer in enumerate(sorted_layers):
            relabeling[orig_layer] = ii
        subnet = pn.transforms.relabel(pn.subnet(M,nodes,layers),layerNames=relabeling)
        complete_invariant = pn.get_complete_invariant(subnet,allowed_aspects=allowed_aspects)
    else:
        subnet = pn.subnet(M,nodes,layers)
        complete_invariant = pn.get_complete_invariant(subnet,allowed_aspects=allowed_aspects)
    aggregated_dict[complete_invariant][sorted_layers] = aggregated_dict[complete_invariant].get(sorted_layers,0) + 1
    if isinstance(examples_dict,dict):
        if complete_invariant not in examples_dict:
            examples_dict[complete_invariant] = subnet
    return
    
def read_subnet_with_complete_invariant(filename,return_subnets=True):
    # Uses eval, only read files where you know the contents
    complete_invariants = dict()
    subnets = []
    with open(filename,'r') as f:
        for line in f:
            try:
                node_string,layer_string,comp_invariant_string = line.rstrip().split('\t')
            except ValueError:
                raise ValueError('Number of entries per line is not three')
            # Casting into list done mostly using string manipulation but eval used for complete invariant
            # Eval warning
            complete_invariant = eval(comp_invariant_string)
            complete_invariants[complete_invariant] = complete_invariants.get(complete_invariant,0) + 1
            if return_subnets:
                nodes = node_string.strip('[]').split("', '")
                nodes[0] = nodes[0].strip("'")
                nodes[-1] = nodes[-1].strip("'")
                layers = [int(l) for l in layer_string.strip('[]').split(', ')]
                subnet_data = [nodes,layers,complete_invariant]
                subnets.append(subnet_data)
    if return_subnets:
        return complete_invariants,subnets
    else:
        return complete_invariants
        
def yield_subnet_with_complete_invariant(filename):
    with open(filename,'r') as f:
        for line in f:
            try:
                node_string,layer_string,comp_invariant_string = line.rstrip().split('\t')
            except ValueError:
                raise ValueError('Number of entries per line is not three')
            # Casting into list done mostly using string manipulation but eval used for complete invariant
            # Eval warning
            complete_invariant = eval(comp_invariant_string)
            nodes = node_string.strip('[]').split("', '")
            nodes[0] = nodes[0].strip("'")
            nodes[-1] = nodes[-1].strip("'")
            layers = [int(l) for l in layer_string.strip('[]').split(', ')]
            subnet_data = [nodes,layers,complete_invariant]
            yield subnet_data
