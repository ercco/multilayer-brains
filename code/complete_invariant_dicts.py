import pymnet as pn
import pickle
import os

'''
Brute force calculation of isomorphism classes. Very slow!
'''
    
#################### Multilayer ############################################################################################################
    
def generate_example_nets(nnodes,nlayers):
    invdict = dict()
    full = pn.full_multilayer(nnodes,nlayers)
    for net in pn.transforms.subnet_iter(full):
        compinv = pn.get_complete_invariant(net)
        if compinv not in invdict:
            invdict[compinv] = net
    return invdict
    
def generate_example_nets_file(nnodes,nlayers,filename):
    f = open(filename,'w')
    invdict = generate_example_nets(nnodes,nlayers)
    pickle.dump(invdict,f)
    f.close()
    
def load_example_nets_file(filename):
    f = open(filename,'r')
    invdict = pickle.load(f)
    f.close()
    return invdict
    
def generate_example_net_files_for_plotting():
    # creates the example net files that plotting functions use, created into folder example_nets
    folder = 'example_nets/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for nnodes in [2,3,4]:
        for nlayers in [2,3]:
            filename = folder+str(nnodes)+'_'+str(nlayers)+'.pickle'
            generate_example_nets_file(nnodes,nlayers,filename)
