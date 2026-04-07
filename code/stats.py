import pickle
import scipy.stats
import os
import collections
import numpy as np

import network_io
import complete_invariant_dicts
import plotting
import pipeline
import network_construction

def find_numbers_of_isomorphism_classes():
    example_nets_folder = '???'
    sizes = [(2,2),(3,2),(4,2),(2,3),(3,3)]
    counts = dict()
    for size in sizes:
        with open(example_nets_folder+str(size[0])+'_'+str(size[1])+'.pickle','r') as f:
            d = pickle.load(f)
        counts[size] = len(d)
        del d
    return counts
    
def t_tests_for_two_dict_lists(list1,list2,equal_var=False,map_ni_to_nli=False,nnodes=-1,nlayers=-1):
    invariants = set()
    layersets = set()
    
    if map_ni_to_nli:
        example_nets_filename = '???'
        invdicts = complete_invariant_dicts.load_example_nets_file(example_nets_filename)
        list1,invariants1,mapped_invdicts1 = plotting.ni_to_nli(list1,invdicts)
        list2,invariants2,mapped_invdicts2 = plotting.ni_to_nli(list2,invdicts)
        invariants = invariants1.union(invariants2)
        
    
    for compinv_dict in list1:
        for compinv in compinv_dict:
            if not map_ni_to_nli:
                invariants.add(compinv)
            layersets.update([layerset for layerset in compinv_dict[compinv]])
    for compinv_dict in list2:
        for compinv in compinv_dict:
            if not map_ni_to_nli:
                invariants.add(compinv)
            layersets.update([layerset for layerset in compinv_dict[compinv]])
            
    layersets = list(layersets)
    layersets.sort()
    
    results = collections.defaultdict(dict)
    
    for compinv in invariants:
        for layerset in layersets:
            temp_layerset_vals_1 = []
            for compinv_dict in list1:
                if compinv in compinv_dict:
                    temp_layerset_vals_1.append(compinv_dict[compinv].get(layerset,0))
                else:
                    temp_layerset_vals_1.append(0)
            temp_layerset_vals_2 = []
            for compinv_dict in list2:
                if compinv in compinv_dict:
                    temp_layerset_vals_2.append(compinv_dict[compinv].get(layerset,0))
                else:
                    temp_layerset_vals_2.append(0)
                    
            results[compinv][layerset] = scipy.stats.ttest_ind(temp_layerset_vals_1,temp_layerset_vals_2,equal_var=equal_var)
            
    return results



#################### Basic properties ######################################################################################################

def find_number_and_weights_of_interlayer_edges(network):
    number_of_interlayer_edges = 0
    weights = []
    for edge in list(network.edges):
        if edge[2] != edge[3]:
            number_of_interlayer_edges = number_of_interlayer_edges + 1
            weights.append(edge[4])
    return number_of_interlayer_edges,weights
    
def find_all_interlayer_information(network):
    number_of_interlayer_edges = 0
    weights = []
    intersections = []
    unions = []
    sizes = []
    for edge in list(network.edges):
        if edge[2] != edge[3]:
            number_of_interlayer_edges = number_of_interlayer_edges + 1
            weights.append(edge[4])
            if edge[2] < edge[3]:
                A = set(eval(edge[0]))
                B = set(eval(edge[1]))
            else:
                A = set(eval(edge[1]))
                B = set(eval(edge[0]))
            intersections.append(A.intersection(B))
            unions.append(A.union(B))
            sizes.append((len(A),len(B)))
    return number_of_interlayer_edges,weights,intersections,unions,sizes

def find_in_and_out_degrees(M,layer):
    # Assumes layers are numbered with consecutive integers and considers only edges from layer+-1 to layer
    in_degrees = []
    out_degrees = []
    for node in M.iter_nodes(layer=layer):
        in_degree = 0
        out_degree = 0
        for neighbor in list(M[node,layer]):
            if neighbor[1] == layer - 1:
                in_degree = in_degree + 1
            elif neighbor[1] == layer + 1:
                out_degree = out_degree + 1
        in_degrees.append(in_degree)
        out_degrees.append(out_degree)
    return in_degrees,out_degrees

def find_cluster_sizes(M,layer):
    cluster_sizes = []
    for node in M.iter_nodes(layer=layer):
        voxellist = eval(node)
        cluster_sizes.append(len(voxellist))
    return cluster_sizes



#################### Aggregation and null model p values ###################################################################################

def t_tests_for_aggregated_dicts(compinvdict1,compinvdict2):
    assert set(compinvdict1.keys()) == set(compinvdict2.keys())
    results = dict()
    for compinv in compinvdict1:
        results[compinv] = scipy.stats.ttest_ind(compinvdict1[compinv],compinvdict2[compinv],equal_var=False)
    return results

def aggregated_ni_to_nli(compinvdict,examples_dict):
    nli_dict = collections.defaultdict(list)
    for compinv in compinvdict:
        nl_complete_invariant = network_io.pn.get_complete_invariant(examples_dict[compinv],allowed_aspects='all')
        nli_dict[nl_complete_invariant].extend(compinvdict[compinv])
    return nli_dict
