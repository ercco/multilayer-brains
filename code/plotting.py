import matplotlib.pyplot as plt
import numpy as np
import itertools
import pymnet as pn
import pickle

import subgraph_classification
import complete_invariant_dicts

cm_black = plt.cm.colors.LinearSegmentedColormap.from_list('allblack',[(0,0,0),(0,0,0)]) # black colormap for black line

#################### Nifti file plotting ###################################################################################################

def plot_time_series(xplotrange, yplotrange, zplotrange, imgdata, colormap=plt.cm.ocean_r, **kwargs):
    '''
    Plot time series for voxels contained in x-, y-, and z-ranges in one figure.
    
    Arguments:
    xplotrange, yplotrange, zplotrange -- iterables containing integers
    imgdata -- get_fdata() from nib.load(...) object
    colormap -- colormap used in color cycler for plotting
    kwargs -- passed to plt.plot(...)
    
    Returns:
    fig -- figure
    '''
    colorcycler = plt.cycler('color',colormap(np.linspace(0,1,len(xplotrange)*len(yplotrange)*len(zplotrange))))
    fig = plt.figure()
    fig.gca().set_prop_cycle(colorcycler)
    fig.gca().margins(0,0.05)
    for x in xplotrange:
        for y in yplotrange:
            for z in zplotrange:
                if not all(imgdata[x,y,z,:]==0):
                    #plt.plot(range(0,len(imgdata[x,y,z,:])),imgdata[x,y,z,:],alpha=0.3,linewidth=0.1) # remove color def to use cycler
                    plt.plot(range(0,len(imgdata[x,y,z,:])),imgdata[x,y,z,:],**kwargs)
    return fig
    
def plot_brain_slices(x,y,z,t,imgdata,subj_id='unknown'):
    slice_0 = imgdata[x, :, :, t]
    slice_1 = imgdata[:, y, :, t]
    slice_2 = imgdata[:, :, z, t]
    return show_slices([slice_0, slice_1, slice_2],x,y,z,t,subj_id)
    
def show_slices(slices,x,y,z,t,subj_id):
    # NB! hardcoded dimensions 42,42,24
    fig, axes = plt.subplots(2,2,sharex='col',sharey='row',figsize=((84.0/100.0)*7,(66.0/100.0)*7),gridspec_kw={'height_ratios':[24,42]})
    coordinates = ['x','y','z']
    pos1 = axes[0,1].get_position()
    pos2 = axes[1,0].get_position()
    for i, slice in enumerate(slices):
        im = axes[i//2,i%2].imshow(slice.T, cmap='gray', origin='lower',interpolation='none')
        axes[i//2,i%2].text(1,1,coordinates[i],color='red',ha='left',va='bottom',size='large')
        for spine in axes[i//2,i%2].spines.values():
            spine.set_visible(False)
        axes[i//2,i%2].tick_params(axis='y',direction='out')
        #for tick in axes[i//2,i%2].xaxis.get_major_ticks():
        #    tick.tick10n = tick.tick20n = False
        cbaxes = fig.add_axes([pos1.x0+i*0.13,pos2.y0,0.03,pos2.y1-pos2.y0])
        fig.colorbar(im,cax=cbaxes)
        cbaxes.text(0.15,0,coordinates[i],color='red',ha='left',va='bottom',size='large')
    #fig.colorbar(im)
    axes[1,1].set_frame_on(False)
    labels_for_11 = ['']*len([item.get_text() for item in axes[1,1].get_xticklabels()])
    axes[1,1].set_xticklabels(labels_for_11)
    #axes[1,1].set_xticks([])
    #axes[1,1].set_yticks([])
    axes[1,1].tick_params(axis='y',colors='white')
    axes[1,1].tick_params(axis='x',width=0.0)
    fig.suptitle('ID = {}, x = {}, y = {}, z = {}, t = {}'.format(subj_id,x,y,z,t))
    return fig



#################### Subgraph plotting #####################################################################################################

def plot_isomorphism_class_histogram(complete_invariants):
    # complete_invariants : dict from read_subnet_with_complete_invariant
    inv_numbers = []
    inv_identities = []
    for invariant in complete_invariants:
        inv_numbers.append(complete_invariants[invariant])
        inv_identities.append(invariant)
    plt.bar(range(len(inv_numbers)),inv_numbers,align='center')
    plt.xticks(range(len(inv_numbers)),inv_identities,rotation=90)
    
def plot_complete_invariant_time_series(subnet_filename,title,xlabel,ylabel,
                                        savename=None,examples_filename=None,map_ni_to_nli=False,
                                        colormap=plt.cm.jet,**kwargs):
    
    M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    M['a','b',0,0]=1
    M['a','a',0,1]=1
    
    #layersetwise_complete_invariants = dict()
    complete_invariant_timeseries_dict = dict()
    layersets = set()
    invariants = set()
    
    if subnet_filename.strip()[-10:] == 'agg.pickle':
        f = open(subnet_filename,'r')
        complete_invariant_timeseries_dict = pickle.load(f)
        f.close()
        for key in complete_invariant_timeseries_dict:
            invariants.add(key)
            layersets.update([key2 for key2 in complete_invariant_timeseries_dict[key]])
        
    else:
        for subnet_data in subgraph_classification.yield_subnet_with_complete_invariant(subnet_filename):
            nodes = tuple(subnet_data[0])
            layers = tuple(sorted(subnet_data[1]))
            complete_invariant = subnet_data[2]
            layersets.add(layers)
            invariants.add(complete_invariant)
            #layersetwise_complete_invariants[layers] = layersetwise_complete_invariants.get(layers,dict())
            #layersetwise_complete_invariants[layers][complete_invariant] = layersetwise_complete_invariants[layers].get(complete_invariant,0) + 1
            complete_invariant_timeseries_dict[complete_invariant] = complete_invariant_timeseries_dict.get(complete_invariant,dict())
            complete_invariant_timeseries_dict[complete_invariant][layers] = complete_invariant_timeseries_dict[complete_invariant].get(
                layers,0) + 1
                
    layersets = list(layersets)
    layersets.sort()
    full_range = zip(*(range(layersets[0][0],layersets[-1][-1]+1)[ii:] for ii in range(len(layersets[0]))))
    if layersets != full_range:
        layersets = full_range
    x_axis = list(range(layersets[0][0],layersets[-1][1]))
    
    # load example networks
    if examples_filename is not None:
        f = open(examples_filename,'r')
        invdicts = pickle.load(f)
        f.close()
    else:
        nnodes = int(subnet_filename.split('/')[-1].strip()[0])
        nlayers = int(subnet_filename.split('/')[-1].strip()[2])
        example_nets_filename = 'example_nets/'+str(nnodes)+'_'+str(nlayers)+'.pickle'
        invdicts = complete_invariant_dicts.load_example_nets_file(example_nets_filename)
        
    # map node isomorphisms to nodelayer isomorphisms:
    if map_ni_to_nli:
        invariants = set()
        mapped_invdicts = dict()
        mapped_complete_invariant_timeseries_dict = dict()
        for complete_invariant in complete_invariant_timeseries_dict:
            nl_complete_invariant = pn.get_complete_invariant(invdicts[complete_invariant])
            if nl_complete_invariant not in mapped_invdicts:
                mapped_invdicts[nl_complete_invariant] = invdicts[complete_invariant]
            invariants.add(nl_complete_invariant)
            mapped_complete_invariant_timeseries_dict[nl_complete_invariant] = mapped_complete_invariant_timeseries_dict.get(nl_complete_invariant,dict())
            for tw in complete_invariant_timeseries_dict[complete_invariant]:
                mapped_complete_invariant_timeseries_dict[nl_complete_invariant][tw] = mapped_complete_invariant_timeseries_dict[nl_complete_invariant].get(tw,0) + complete_invariant_timeseries_dict[complete_invariant][tw]
        complete_invariant_timeseries_dict = mapped_complete_invariant_timeseries_dict
        invdicts = mapped_invdicts
        
    # needed numbers of example figures
    number_of_invariants = len(invariants)
    grid_side_length = 0
    while grid_side_length**2 < number_of_invariants:
        grid_side_length = grid_side_length + 1
    ax_locs = list(itertools.product(range(grid_side_length),range(grid_side_length,2*grid_side_length)))
    
    # figure
    fig = plt.figure(figsize=(12,6))
    main_ax = plt.subplot2grid((grid_side_length,2*grid_side_length),(0,0),rowspan=grid_side_length,colspan=grid_side_length)
    colorcycler = plt.cycler('color',colormap(np.linspace(0,1,number_of_invariants)))
    main_ax.set_prop_cycle(colorcycler)
#    side_ax = plt.subplot2grid((2,4),(0,2),projection='3d')
#    side_ax2 = plt.subplot2grid((2,4),(0,3),projection='3d')
    #plt.hold(True)
    
    # identifiers for legend
    ids = range(len(invariants))
    
    for ii,compinv in enumerate(sorted(complete_invariant_timeseries_dict.iterkeys())):
        y_values = []
        for layerset in layersets:
            y_values.append(complete_invariant_timeseries_dict[compinv].get(layerset,0))
        #main_ax = plt.subplot2grid((2,4),(0,0),rowspan=2,colspan=2)
        #main_ax.plot(x_axis,y_values,label=str(compinv))
        line = main_ax.plot(x_axis,y_values,label=str(ii))
        compinv_ax = plt.subplot2grid((grid_side_length,2*grid_side_length),ax_locs[ii],projection='3d')
        #example_ax = plt.gcf().add_axes((1.1,0,0.5,0.5),projection='3d')
        M = invdicts[compinv]
        pn.draw(M,layout='shell',alignedNodes=True,ax=compinv_ax,layerLabelRule={},nodeLabelRule={})
        #compinv_ax.text2D(0,0,str(ii),None,False,transform=compinv_ax.transAxes)
        if grid_side_length < 3:
            legend_loc = 'lower left'
        else:
            legend_loc = (0,0)
        leg = compinv_ax.legend(labels=[''],loc=legend_loc,handles=line,frameon=False)
        plt.setp(leg.get_lines(),linewidth=4)
#        if ii==0:
#            pn.draw(M,ax=side_ax)
#        else:
#            pn.draw(M,ax=side_ax2)
    #plt.sca(fig.gca())
    
#    plt.xticks(x_axis,layersets,rotation=90,fontsize='small')
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.title(title)
#    plt.legend(bbox_to_anchor=(1.05,1.),loc=0,fontsize='xx-small')
#    plt.yscale('log')
#    plt.margins(y=0.2)
#    plt.xlim([x_axis[0],x_axis[-1]])
#    plt.tight_layout()
#    plt.show()
    main_ax.set_xticks(x_axis)
    main_ax.set_xticklabels(layersets,rotation=90,fontsize='small')
    main_ax.set_yscale('log')
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)
    main_ax.set_title(title)
    #main_ax.legend(bbox_to_anchor=(1.05,0),loc='upper left',ncol=number_of_invariants,fontsize='small')
    #main_ax.yscale('log')
    main_ax.margins(y=0.2)
    main_ax.set_xlim([x_axis[0],x_axis[-1]])
    plt.tight_layout()
    
    if savename == None:
        plt.show()
    else:
        plt.savefig(savename,format='pdf')
        
def plot_complete_invariant_average_time_series(subnet_filename_list,title,xlabel,ylabel,
                                        savename=None,examples_filename=None,map_ni_to_nli=False,
                                        yscale='log',
                                        colormap=plt.cm.jet,**kwargs):
    compinv_timeseries_dict_list = []
    layersets = set()
    invariants = set()
    for filename in subnet_filename_list:
        f = open(filename,'r')
        complete_invariant_timeseries_dict = pickle.load(f)
        f.close()
        for key in complete_invariant_timeseries_dict:
            invariants.add(key)
            layersets.update([key2 for key2 in complete_invariant_timeseries_dict[key]])
        compinv_timeseries_dict_list.append(complete_invariant_timeseries_dict)
    
    if examples_filename is not None:
        invdicts = complete_invariant_dicts.load_example_nets_file(examples_filename)
    else:
        nnodes = int(subnet_filename_list[0].split('/')[-1].strip()[0])
        nlayers = int(subnet_filename_list[0].split('/')[-1].strip()[2])
        example_nets_filename = 'example_nets/'+str(nnodes)+'_'+str(nlayers)+'.pickle'
        invdicts = complete_invariant_dicts.load_example_nets_file(example_nets_filename)
        
    layersets = list(layersets)
    layersets.sort()
    full_range = zip(*(range(layersets[0][0],layersets[-1][-1]+1)[ii:] for ii in range(len(layersets[0]))))
    if layersets != full_range:
        layersets = full_range
    x_axis = list(range(layersets[0][0],layersets[-1][1]))
    
    if map_ni_to_nli:
        invariants = set()
        mapped_invdicts = dict()
        mapped_compinv_timeseries_dict_list = []
        for complete_invariant_timeseries_dict in compinv_timeseries_dict_list:
            mapped_complete_invariant_timeseries_dict = dict()
            for complete_invariant in complete_invariant_timeseries_dict:
                nl_complete_invariant = pn.get_complete_invariant(invdicts[complete_invariant])
                if nl_complete_invariant not in mapped_invdicts:
                    mapped_invdicts[nl_complete_invariant] = invdicts[complete_invariant]
                invariants.add(nl_complete_invariant)
                mapped_complete_invariant_timeseries_dict[nl_complete_invariant] = mapped_complete_invariant_timeseries_dict.get(nl_complete_invariant,dict())
                for tw in complete_invariant_timeseries_dict[complete_invariant]:
                    mapped_complete_invariant_timeseries_dict[nl_complete_invariant][tw] = mapped_complete_invariant_timeseries_dict[nl_complete_invariant].get(tw,0) + complete_invariant_timeseries_dict[complete_invariant][tw]
            mapped_compinv_timeseries_dict_list.append(mapped_complete_invariant_timeseries_dict)
        compinv_timeseries_dict_list = mapped_compinv_timeseries_dict_list
        invdicts = mapped_invdicts
        
    number_of_invariants = len(invariants)
    grid_side_length = 0
    while grid_side_length**2 < number_of_invariants:
        grid_side_length = grid_side_length + 1
    ax_locs = list(itertools.product(range(grid_side_length),range(grid_side_length,2*grid_side_length)))
    
    fig = plt.figure(figsize=(12,6))
    main_ax = plt.subplot2grid((grid_side_length,2*grid_side_length),(0,0),rowspan=grid_side_length,colspan=grid_side_length)
    colorcycler = plt.cycler('color',colormap(np.linspace(0,1,number_of_invariants)))
    main_ax.set_prop_cycle(colorcycler)
    
    for ii,compinv in enumerate(sorted(invariants)):
        y_values = []
        std_values = []
        for layerset in layersets:
            temp_layerset_vals = []
            for compinv_timeseries_dict in compinv_timeseries_dict_list:
                if compinv in compinv_timeseries_dict:
                    temp_layerset_vals.append(compinv_timeseries_dict[compinv].get(layerset,0))
                else:
                    temp_layerset_vals.append(0)
            y_values.append(np.mean(temp_layerset_vals))
            std_values.append(np.std(temp_layerset_vals))

        line = main_ax.plot(x_axis,y_values,label=str(ii))
        main_ax.fill_between(x_axis,np.subtract(y_values,std_values),np.add(y_values,std_values),facecolor=line[0].get_color(),alpha=0.2)
        compinv_ax = plt.subplot2grid((grid_side_length,2*grid_side_length),ax_locs[ii],projection='3d')
        M = invdicts[compinv]
        pn.draw(M,layout='shell',alignedNodes=True,ax=compinv_ax,layerLabelRule={},nodeLabelRule={})
        if grid_side_length < 3:
            legend_loc = 'lower left'
        else:
            legend_loc = (0,0)
        leg = compinv_ax.legend(labels=[''],loc=legend_loc,handles=line,frameon=False)
        plt.setp(leg.get_lines(),linewidth=4)
        
    main_ax.set_xticks(x_axis)
    main_ax.set_xticklabels(layersets,rotation=90,fontsize='small')
    main_ax.set_yscale(yscale)
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)
    main_ax.set_title(title)
    main_ax.margins(y=0.2)
    main_ax.set_xlim([x_axis[0],x_axis[-1]])
    plt.tight_layout()
    
    if savename == None:
        plt.show()
    else:
        plt.savefig(savename,format='pdf')



#################### Mapping nodeisomorphisms to nodelayerisomorphisms #####################################################################
        
def ni_to_nli(compinv_timeseries_dict_list,invdicts):
    invariants = set()
    mapped_invdicts = dict()
    mapped_compinv_timeseries_dict_list = []
    for complete_invariant_timeseries_dict in compinv_timeseries_dict_list:
        mapped_complete_invariant_timeseries_dict = dict()
        for complete_invariant in complete_invariant_timeseries_dict:
            nl_complete_invariant = pn.get_complete_invariant(invdicts[complete_invariant],allowed_aspects='all')
            if nl_complete_invariant not in mapped_invdicts:
                mapped_invdicts[nl_complete_invariant] = invdicts[complete_invariant]
            invariants.add(nl_complete_invariant)
            mapped_complete_invariant_timeseries_dict[nl_complete_invariant] = mapped_complete_invariant_timeseries_dict.get(nl_complete_invariant,dict())
            for tw in complete_invariant_timeseries_dict[complete_invariant]:
                mapped_complete_invariant_timeseries_dict[nl_complete_invariant][tw] = mapped_complete_invariant_timeseries_dict[nl_complete_invariant].get(tw,0) + complete_invariant_timeseries_dict[complete_invariant][tw]
        mapped_compinv_timeseries_dict_list.append(mapped_complete_invariant_timeseries_dict)
    return mapped_compinv_timeseries_dict_list,invariants,mapped_invdicts

#################### Visualization of voxel-level correlation matrices #####################################################################
    
def visualize_roi_ordered_correlation_matrix(correlations, n_voxels, ROI_onsets, save_path, ROI_boundary_color='r'):
    """
    Visualizes as a heatmap the voxel-level correlation where voxels are ordered
    by their ROI identity.
    
    Parameters:
    -----------
    correlations: 1D np.array, the upper-triangle values of the correlation matrix
    n_voxels: int, the number of voxels
    ROI_onsets: 1D np.array, the row/column index of the first voxel of each ROI in the correlation matrix
    save_path: str, path to which to save the visualization
    ROI_boundary_color: str, color for drawind the lines showing ROI boundaries
    
    Output:
    -------
    saves the visualization as .pdf
    """
    plt.figure()
    correlation_matrix = np.eye(n_voxels)
    triu_indices = np.triu_indices(n_voxels, k=1)
    for correlation, triu_x, triu_y in zip(correlations, triu_indices[0], triu_indices[1]):
        correlation_matrix[triu_x, triu_y] = correlation
        correlation_matrix[triu_y, triu_x] = correlation
    plt.imshow(correlation_matrix)
    y = np.arange(-0.5, n_voxels + 0.5)
    for i, ROI_onset in enumerate(ROI_onsets):
        x = [ROI_onset - 0.5] * (n_voxels + 1)
        plt.plot(x, y, color=ROI_boundary_color)
        plt.plot(y, x, color=ROI_boundary_color)
    plt.tight_layout()
    plt.savefig(save_path,format='pdf',bbox_inches='tight')

