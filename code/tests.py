import numpy as np
import unittest
import pymnet as pn
import itertools
import os
import collections

import clustering
import corrs_and_mask_calculations
import network_construction
import subgraph_classification
import network_io
import plotting
import statistics
import null_models

class test_network_construction(unittest.TestCase):
    
    timeseries0 = np.array(range(-10,11),dtype=np.float64)
    timeseries1 = np.array([-0.46362334, -0.56982772,  0.06455791, -0.44209878, -0.31068497,
        0.05360425, -0.41299186, -0.29082169, -0.07190158, -0.12474256,
       -0.24997589,  0.01267206,  0.03601663,  0.29330202, -0.12646342,
        0.13130587,  0.57496159,  0.77851974,  0.12816724,  0.63563011,
        0.35058168],dtype=np.float64)
    timeseries2 = np.array(range(100,121),dtype=np.float64)
    timeseries3 = np.zeros((21,))
    timeseries4 = np.array([1,2,3,4,5,6,7,8,9,10,11,10,9,8,7,6,5,4,3,2,1],dtype=np.float64)
    timeseries5 = np.array(range(0,-21,-1))
    
    timeseries6 = np.copy(timeseries0)
    timeseries7 = np.copy(timeseries1)
    timeseries8 = np.copy(timeseries2)
    timeseries9 = np.copy(timeseries3)
    timeseries10 = np.copy(timeseries4)
    timeseries11 = np.copy(timeseries5)
    
    imgdata = np.block([[[[timeseries0],[timeseries1]],[[timeseries2],
                          [timeseries3]],[[timeseries4],[timeseries5]]],
                        [[[timeseries6],[timeseries7]],[[timeseries8],
                          [timeseries9]],[[timeseries10],[timeseries11]]]])
                          
    # two more imgdatas for clustering test                         
    imgdata2 = np.block([[[[timeseries4],[timeseries0]],[[timeseries5],
                          [timeseries1]],[[timeseries6],[timeseries2]]],
                        [[[timeseries9],[timeseries3]],[[timeseries11],
                          [timeseries10]],[[timeseries7],[timeseries8]]]])
                          
    imgdata3 = np.block([[[[timeseries9],[timeseries0]],[[timeseries1],
                          [timeseries2]],[[timeseries4],[timeseries11]]],
                        [[[timeseries3],[timeseries6]],[[timeseries7],
                          [timeseries10]],[[timeseries8],[timeseries5]]]])
                          
    imgdata_large = np.concatenate((imgdata2,imgdata,imgdata3),axis=3)
                          
    R_true = np.array([[1,0.8,1,np.nan,0,-1,1,0.8,1,np.nan,0,-1],[0.8,1,0.8,np.nan,-0.15319,-0.8,0.8,1,0.8,np.nan,-0.15319,-0.8],
                       [1,0.8,1,np.nan,0,-1,1,0.8,1,np.nan,0,-1],[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                        [0,-0.15319,0,np.nan,1,0,0,-0.15319,0,np.nan,1,0],[-1,-0.8,-1,np.nan,0,1,-1,-0.8,-1,np.nan,0,1],
                        [1,0.8,1,np.nan,0,-1,1,0.8,1,np.nan,0,-1],[0.8,1,0.8,np.nan,-0.15319,-0.8,0.8,1,0.8,np.nan,-0.15319,-0.8],
                        [1,0.8,1,np.nan,0,-1,1,0.8,1,np.nan,0,-1],[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                        [0,-0.15319,0,np.nan,1,0,0,-0.15319,0,np.nan,1,0],[-1,-0.8,-1,np.nan,0,1,-1,-0.8,-1,np.nan,0,1]])
                        
    template = np.zeros((2,3,2))
    template[0,0,0] = 1
    template[0,1,1] = 1
    template[0,0,1] = 1.5
    template[1,1,0] = 1.5
    
    def generate_full_random_weighted_network(self,nnodes,nlayers):
        # random node names and random sequential layer numbers
        M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
        letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        # start from layer 9 to check that there are no indexing issues with 9->10
        l0 = 9
        # have one node name constant over all layers
        c = 'fixed'
        for l in range(l0,l0+nlayers):
            M.add_layer(l)
            nodes_l = [c]+[''.join(np.random.choice(letters,5)) for i in range(nnodes-1)]
            for i1 in range(len(nodes_l)):
                for i2 in range(i1+1,len(nodes_l)):
                    M[nodes_l[i1],l][nodes_l[i2],l] = 1.0 - np.random.random()
            if l != l0:
                for n1 in nodes_l:
                    for n2 in M.iter_nodes(l-1):
                        M[n1,l][n2,l-1] = 1.0 - np.random.random()
        return M
                        
    def get_R_true_index(self,voxel):
        return self.imgdata.shape[1]*self.imgdata.shape[2]*voxel[0] + self.imgdata.shape[2]*voxel[1] + voxel[2]
        
    def test_find_unmasked_voxels(self):
        unmasked_voxels = set(corrs_and_mask_calculations.find_unmasked_voxels(self.imgdata))
        true_unmasked_voxels = set(list(itertools.product([0,1],[0,1,2],[0,1])))-set([(0,1,1),(1,1,1)])
        self.assertEqual(unmasked_voxels,true_unmasked_voxels)
        
    def test_find_masked_voxels(self):
        masked_voxels = set(corrs_and_mask_calculations.find_masked_voxels(self.imgdata))
        true_masked_voxels = set([(0,1,1),(1,1,1)])
        self.assertEqual(masked_voxels,true_masked_voxels)
        
    def test_gray_mask(self):
        temp_imgdata = self.imgdata.copy()
        gray_matter_mask = np.zeros(self.imgdata.shape[0:3])
        gray_matter_mask[0,1,0] = 0.5
        gray_matter_mask[1,2,1] = 0.9
        corrs_and_mask_calculations.gray_mask(temp_imgdata,gray_matter_mask)
        unmasked_voxels = set(corrs_and_mask_calculations.find_unmasked_voxels(temp_imgdata))
        true_unmasked_voxels = set([(0,1,0),(1,2,1)])
        self.assertEqual(unmasked_voxels,true_unmasked_voxels)
        self.assertTrue(all(temp_imgdata[0,1,0]==self.timeseries2))
        self.assertTrue(all(temp_imgdata[1,2,1]==self.timeseries5))
        for coord in set(list(itertools.product([0,1],[0,1,2],[0,1])))-set([(0,1,0),(1,2,1)]):
            self.assertTrue(all(temp_imgdata[coord]==np.zeros(self.imgdata.shape[3])))
    
    def test_calculate_correlation_matrix(self):
        # calculate all
        R = corrs_and_mask_calculations.calculate_correlation_matrix(self.imgdata,exclude_masked=False)
        self.assertEqual(R.shape,self.R_true.shape)
        self.assertTrue(np.allclose(R,self.R_true,rtol=0,atol=0.001,equal_nan=True))
        # exclude masked
        R,unmasked_voxels = corrs_and_mask_calculations.calculate_correlation_matrix(self.imgdata,exclude_masked=True)
        for ii,voxel1 in enumerate(unmasked_voxels):
            R_true_index_voxel1 = self.imgdata.shape[1]*self.imgdata.shape[2]*voxel1[0] + self.imgdata.shape[2]*voxel1[1] + voxel1[2]
            for jj,voxel2 in enumerate(unmasked_voxels):
                R_true_index_voxel2 = self.imgdata.shape[1]*self.imgdata.shape[2]*voxel2[0] + self.imgdata.shape[2]*voxel2[1] + voxel2[2]
                self.assertTrue(np.isclose(R[ii,jj],self.R_true[R_true_index_voxel1,R_true_index_voxel2],rtol=0,atol=0.001,equal_nan=False))
                
    def test_get_voxels_in_clusters(self):
        voxels_in_clusters = network_construction.get_voxels_in_clusters(self.template)
        self.assertEqual(voxels_in_clusters,{0:[(0,0,0),(0,1,1)],1:[(0,0,1),(1,1,0)]})
                
    def test_calculate_cluster_correlation_matrix(self):
        voxels_in_clusters = network_construction.get_voxels_in_clusters(self.template)
        R = network_construction.calculate_cluster_correlation_matrix(self.imgdata,voxels_in_clusters)
        self.assertTrue(np.isclose(R,np.corrcoef(self.timeseries0,self.timeseries1+self.timeseries8)).all())
        
    def test_threshold_multilayer_network(self):
        testnet = pn.full_multilayer(10,[1,2,3])
        for edge in list(testnet.edges):
            if abs(edge[2]-edge[3]) > 1:
                testnet[edge[0],edge[1],edge[2],edge[3]] = 0
            else:
                testnet[edge[0],edge[1],edge[2],edge[3]] = -0.8
        testnet[2,3,1,1] = 4
        testnet[4,5,1,1] = 0.19
        testnet[3,4,1,1] = 0.14
        testnet[5,6,2,2] = 0.9
        testnet[7,8,2,2] = 0.18
        testnet[6,7,2,2] = -0.02
        testnet[4,5,3,3] = 0.95
        testnet[2,3,3,3] = 0.17
        testnet[3,4,3,3] = 0.02
        testnet[3,4,1,2] = 1.2
        testnet[6,7,1,2] = 0.15
        testnet[9,8,1,2] = 0.02
        testnet[5,4,2,3] = 0.78
        testnet[8,7,2,3] = 0.15
        testnet[1,8,2,3] = -0.02
        thresholded_net = network_construction.threshold_multilayer_network(testnet,0.05,0.02)
        truenet = pn.MultilayerNetwork(aspects=1,fullyInterconnected=True)
        for ii in [1,2,3]:
            truenet.add_layer(ii)
        for ii in range(10):
            truenet.add_node(ii)
        truenet[2,3,1,1] = 1
        truenet[4,5,1,1] = 1
        truenet[5,6,2,2] = 1
        truenet[7,8,2,2] = 1
        truenet[4,5,3,3] = 1
        truenet[2,3,3,3] = 1
        truenet[3,4,1,2] = 1
        truenet[6,7,1,2] = 1
        truenet[5,4,2,3] = 1
        truenet[8,7,2,3] = 1
        self.assertEqual(thresholded_net,truenet)
        
    def test_threshold_network(self):
        # test different parameter sets
        M = self.generate_full_random_weighted_network(10,3)
        w1 = sorted([e[4] for e in M.edges if e[2] == 9 and e[3] == 9],reverse=True)
        w2 = sorted([e[4] for e in M.edges if e[2] == 10 and e[3] == 10],reverse=True)
        w3 = sorted([e[4] for e in M.edges if e[2] == 11 and e[3] == 11],reverse=True)
        w12 = sorted([e[4] for e in M.edges if (e[2] == 9 and e[3] == 10) or (e[2] == 10 and e[3] == 9)],reverse=True)
        w23 = sorted([e[4] for e in M.edges if (e[2] == 10 and e[3] == 11) or (e[2] == 11 and e[3] == 10)],reverse=True)
        M = network_construction.threshold_network(M,{'intra_avg_degree':1.6,'inter_avg_degree':0.5,'replace_intralayer_weights_with_ones':False,'replace_interlayer_weights_with_ones':False})
        self.assertEqual(sorted([e[4] for e in M.edges if e[2] == 9 and e[3] == 9],reverse=True),w1[0:8])
        self.assertEqual(sorted([e[4] for e in M.edges if e[2] == 10 and e[3] == 10],reverse=True),w2[0:8])
        self.assertEqual(sorted([e[4] for e in M.edges if e[2] == 11 and e[3] == 11],reverse=True),w3[0:8])
        self.assertEqual(sorted([e[4] for e in M.edges if (e[2] == 9 and e[3] == 10) or (e[2] == 10 and e[3] == 9)],reverse=True),w12[0:5])
        self.assertEqual(sorted([e[4] for e in M.edges if (e[2] == 10 and e[3] == 11) or (e[2] == 11 and e[3] == 10)],reverse=True),w23[0:5])
        M = self.generate_full_random_weighted_network(10,3)
        M = network_construction.threshold_network(M,{'intra_avg_degree':2.1,'inter_avg_degree':2.06,'replace_intralayer_weights_with_ones':True,'replace_interlayer_weights_with_ones':True})
        self.assertEqual(len([e[4] for e in M.edges if e[2] == 9 and e[3] == 9]),10)
        self.assertEqual(len([e[4] for e in M.edges if e[2] == 10 and e[3] == 10]),10)
        self.assertEqual(len([e[4] for e in M.edges if e[2] == 11 and e[3] == 11]),10)
        self.assertEqual(len([e[4] for e in M.edges if (e[2] == 9 and e[3] == 10) or (e[2] == 10 and e[3] == 9)]),20)
        self.assertEqual(len([e[4] for e in M.edges if (e[2] == 10 and e[3] == 11) or (e[2] == 11 and e[3] == 10)]),20)
        self.assertTrue(all([e[4]==1 for e in M.edges]))
        
    def test_make_multiplex(self):
        # voxel-level = multiplex network
        # timewindow = 21 (all data)
        M = network_construction.make_multiplex(self.imgdata,timewindow=21,overlap=0)
        self.assertTrue(M.couplings == [('ordinal',1.0)])
        self.assertTrue(set(M.iter_layers()) == set((0,)))
        true_nodes = set([str(voxel) for voxel in itertools.product([0,1],[0,1,2],[0,1])]) - set([str((0,1,1)),str((1,1,1))])
        self.assertTrue(set(M.iter_nodes()) == true_nodes)
        for edge in M.edges:
            voxel1 = eval(edge[0])
            R_true_index_voxel1 = self.get_R_true_index(voxel1)
            voxel2 = eval(edge[1])
            R_true_index_voxel2 = self.get_R_true_index(voxel2)
            layer1 = edge[2]
            layer2 = edge[3]
            weight = edge[4]
            if voxel1 != voxel2:
                if layer1 == 0 and layer2 == 0:
                    self.assertTrue(np.isclose(weight,self.R_true[R_true_index_voxel1,R_true_index_voxel2],rtol=0,atol=0.001,equal_nan=False))
                else:
                    raise Exception('Illegal inter-layer links')
            else:
                raise Exception('Inter-layer links present in one-layer case')
        
        # timewindow = 10
        M = network_construction.make_multiplex(self.imgdata,timewindow=10,overlap=0)
        self.assertTrue(M.couplings == [('ordinal',1.0)])
        self.assertTrue(set(M.iter_layers()) == set((0,1)))
        true_nodes = set([str(voxel) for voxel in itertools.product([0,1],[0,1,2],[0,1])]) - set([str((0,1,1)),str((1,1,1))])
        self.assertTrue(set(M.iter_nodes()) == true_nodes)
        for edge in M.edges:
            voxel1 = eval(edge[0])
            voxel2 = eval(edge[1])
            layer1 = edge[2]
            layer2 = edge[3]
            weight = edge[4]
            if voxel1 != voxel2:
                if layer1 == 0 and layer2 == 0:
                    self.assertTrue(np.isclose(weight,np.corrcoef(self.imgdata[voxel1[0],voxel1[1],voxel1[2],0:10],self.imgdata[voxel2[0],voxel2[1],voxel2[2],0:10])[0][1],rtol=0,atol=0.001,equal_nan=False))
                elif layer1 == 1 and layer2 == 1:
                    self.assertTrue(np.isclose(weight,np.corrcoef(self.imgdata[voxel1[0],voxel1[1],voxel1[2],10:20],self.imgdata[voxel2[0],voxel2[1],voxel2[2],10:20])[0][1],rtol=0,atol=0.001,equal_nan=False))
                else:
                    raise Exception('Illegal inter-layer links')
            else:
                self.assertTrue(weight == 1)
                self.assertTrue(layer1 == layer2-1 or layer1 == layer2+1)
        
        # timewindow = 8, overlap = 2
        M = network_construction.make_multiplex(self.imgdata,timewindow=8,overlap=2)
        self.assertTrue(M.couplings == [('ordinal',1.0)])
        self.assertTrue(set(M.iter_layers()) == set((0,1,2)))
        true_nodes = set([str(voxel) for voxel in itertools.product([0,1],[0,1,2],[0,1])]) - set([str((0,1,1)),str((1,1,1))])
        self.assertTrue(set(M.iter_nodes()) == true_nodes)
        for edge in M.edges:
            voxel1 = eval(edge[0])
            voxel2 = eval(edge[1])
            layer1 = edge[2]
            layer2 = edge[3]
            weight = edge[4]
            if voxel1 != voxel2:
                if layer1 == 0 and layer2 == 0:
                    self.assertTrue(np.isclose(weight,np.corrcoef(self.imgdata[voxel1[0],voxel1[1],voxel1[2],0:8],self.imgdata[voxel2[0],voxel2[1],voxel2[2],0:8])[0][1],rtol=0,atol=0.001,equal_nan=False))
                elif layer1 == 1 and layer2 == 1:
                    self.assertTrue(np.isclose(weight,np.corrcoef(self.imgdata[voxel1[0],voxel1[1],voxel1[2],6:14],self.imgdata[voxel2[0],voxel2[1],voxel2[2],6:14])[0][1],rtol=0,atol=0.001,equal_nan=False))
                elif layer1 == 2 and layer2 == 2:
                    self.assertTrue(np.isclose(weight,np.corrcoef(self.imgdata[voxel1[0],voxel1[1],voxel1[2],12:20],self.imgdata[voxel2[0],voxel2[1],voxel2[2],12:20])[0][1],rtol=0,atol=0.001,equal_nan=False))
                else:
                    raise Exception('Illegal inter-layer links')
            else:
                self.assertTrue(weight == 1)
                self.assertTrue(layer1 == layer2-1 or layer1 == layer2+1)
    
    def test_make_clustered_multilayer(self):
        # TODO: link weights
        # One layer
        # 4 clusters
        M = network_construction.make_clustered_multilayer(self.imgdata,timewindow=21,overlap=0,n_clusters=4)
        self.assertTrue(set(M.iter_layers()) == set((0,)))
        self.assertTrue(all(isinstance(nodelabel,str) for nodelabel in M.iter_nodes(layer=0)))
        clusters = set(frozenset(eval(nodelabel)) for nodelabel in M.iter_nodes(layer=0))
        true_clusters = set(frozenset(clust) for clust in [[(0,0,1),(1,0,1)],[(0,2,0),(1,2,0)],[(1,2,1),(0,2,1)],[(0,0,0),(1,0,0),(0,1,0),(1,1,0)]])
        self.assertEqual(clusters,true_clusters)
        
        # 3 clusters
        M = network_construction.make_clustered_multilayer(self.imgdata,timewindow=21,overlap=0,n_clusters=3)
        self.assertTrue(set(M.iter_layers()) == set((0,)))
        self.assertTrue(all(isinstance(nodelabel,str) for nodelabel in M.iter_nodes(layer=0)))
        clusters = set(frozenset(eval(nodelabel)) for nodelabel in M.iter_nodes(layer=0))
        true_clusters = set(frozenset(clust) for clust in [[(0,2,0),(1,2,0)],[(1,2,1),(0,2,1)],[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1)]])
        self.assertEqual(clusters,true_clusters)
        
        # 2 clusters
        M = network_construction.make_clustered_multilayer(self.imgdata,timewindow=21,overlap=0,n_clusters=2)
        self.assertTrue(set(M.iter_layers()) == set((0,)))
        self.assertTrue(all(isinstance(nodelabel,str) for nodelabel in M.iter_nodes(layer=0)))
        clusters = set(frozenset(eval(nodelabel)) for nodelabel in M.iter_nodes(layer=0))
        true_clusters = set(frozenset(clust) for clust in [[(0,2,0),(1,2,0),(1,2,1),(0,2,1)],[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1)]])
        self.assertEqual(clusters,true_clusters)
        
        # Multiple layers
        # TODO: 
        
        # Testing clustering using predetermined templates
        template = np.array([[[0.5,1.5],[0.5,0],[1.5,1.5]],[[1.5,1.5],[3,0],[3,1.5]]])
        M = network_construction.make_clustered_multilayer(self.imgdata,timewindow=7,overlap=0,n_clusters=999,method='template',template=template,nanlogfile='TEST_nanlogfile.txt')
        self.assertTrue(set(M.iter_layers()) == set((0,1,2)))
        self.assertTrue(all(isinstance(nodelabel,str) for nodelabel in M.iter_nodes(layer=0)))
        clusters = set(frozenset(eval(nodelabel)) for nodelabel in M.iter_nodes(layer=0))
        true_clusters = set(frozenset(clust) for clust in [[(0,0,0),(0,1,0)],[(0,0,1),(0,2,0),(0,2,1),(1,0,0),(1,0,1),(1,2,1)],[(1,1,0),(1,2,0)]])
        self.assertEqual(clusters,true_clusters)
        self.assertTrue(all([not np.isnan(e[4]) for e in M.edges]))
        self.assertEqual(len(M.edges),13)
        try:
            with open('TEST_nanlogfile.txt','r') as f:
                self.assertEqual(f.readline(),'NaN correlation at nodes [(0, 0, 0), (0, 1, 0)], [(1, 1, 0), (1, 2, 0)] at timewindow 2\n')
                self.assertEqual(f.readline(),'NaN correlation at nodes [(0, 0, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 2, 1)], [(1, 1, 0), (1, 2, 0)] at timewindow 2\n')
        finally:
            os.remove('TEST_nanlogfile.txt')
            
    def test_make_specific_timewindows_network_sklearn(self):
        # Does not test if clustering is correct, only that network is created properly
        start_times = [0,5,10]
        end_times = [10,21,21]
        labels = [0,1,2]
        M,models = network_construction.make_specific_timewindows_network_sklearn(self.imgdata,start_times,end_times,labels,4)
        for edge in M.edges:
            nodes1 = set(eval(edge[0]))
            nodes2 = set(eval(edge[1]))
            if edge[2] != edge[3]:
                self.assertTrue(float(len(nodes1.intersection(nodes2)))/len(nodes1.union(nodes2)) == edge[4])
            else:
                start_time = start_times[edge[2]]
                end_time = end_times[edge[2]]
                sum_ts1 = np.zeros(end_time-start_time)
                sum_ts2 = np.zeros(end_time-start_time)
                for node in nodes1:
                    sum_ts1 = sum_ts1 + self.imgdata[node[0],node[1],node[2],start_time:end_time]
                for node in nodes2:
                    sum_ts2 = sum_ts2 + self.imgdata[node[0],node[1],node[2],start_time:end_time]
                corr = np.corrcoef(sum_ts1,sum_ts2)[0][1]
                self.assertTrue(np.isclose(corr,edge[4],rtol=0,atol=1e-08))
        # Very small data just to see that the principle is right
        imgdata_verysmall = np.block([[[[self.timeseries0],[self.timeseries1]],[[self.timeseries4],[self.timeseries3]]]])
        M2,models = network_construction.make_specific_timewindows_network_sklearn(imgdata_verysmall,[0],[21],[0],2)
        M3,models = network_construction.make_specific_timewindows_network_sklearn(imgdata_verysmall,[0],[21],[0],3)
        true_clusters2 = set(frozenset(clust) for clust in [[(0,0,0),(0,0,1)],[(0,1,0)]])
        true_clusters3 = set(frozenset(clust) for clust in [[(0,0,0)],[(0,0,1)],[(0,1,0)]])
        clusters2 = set(frozenset(eval(nodelabel)) for nodelabel in M2.iter_nodes(layer=0))
        clusters3 = set(frozenset(eval(nodelabel)) for nodelabel in M3.iter_nodes(layer=0))
        self.assertEqual(clusters2,true_clusters2)
        self.assertEqual(clusters3,true_clusters3)
        for edge in M2.edges:
            self.assertTrue(np.isclose(edge[4],-0.00877422,rtol=0,atol=1e-05))
        for edge in M3.edges:
            w = edge[4]
            self.assertTrue(np.isclose(w,self.R_true[0,1],rtol=0,atol=0.001) or np.isclose(w,self.R_true[0,4],rtol=0,atol=0.001) or np.isclose(w,self.R_true[1,4],rtol=0,atol=0.001))



class test_subgraph_classification(unittest.TestCase):
    
    testnet = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    testnet[1,2,0,0] = 1
    testnet[1,1,0,1] = 1
    testnet[3,3,1,2] = 1
    testnet[3,4,2,2] = 1
    testnet[1,5,0,0] = 1
    testnet[6,7,2,2] = 1
    testnet[6,6,2,3] = 1
    testnet[7,7,2,3] = 1
    testnet[8,9,2,2] = 1
    testnet[9,9,2,3] = 1
    
    subnet1 = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    subnet1['a','b',0,0] = 1
    subnet1['a','a',0,1] = 1
    
    subnet2 = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    subnet2['a','b',0,0] = 1
    subnet2['a','a',0,1] = 1
    subnet2['b','b',0,1] = 1
    
    subnet3 = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    subnet3['a','b',1,1] = 1
    subnet3['a','a',0,1] = 1
    
    # nodelayer isomorphisms
    compinv1 = pn.get_complete_invariant(subnet1)
    compinv2 = pn.get_complete_invariant(subnet2)
    
    # node isomorphisms
    node_compinv1 = pn.get_complete_invariant(subnet1,[0])
    node_compinv2 = pn.get_complete_invariant(subnet2,[0])
    node_compinv3 = pn.get_complete_invariant(subnet3,[0])
    
    def test_find_isomorphism_classes_with_aggregated_dict(self):
        dd = collections.defaultdict(dict)
        subgraph_classification.find_isomorphism_classes(self.testnet,2,2,'this_file_should_not_exist','all',dd,None)
        self.assertEqual(len(dd),2)
        self.assertEqual(dd[self.compinv1],{(0,1):2,(1,2):1,(2,3):1})
        self.assertEqual(dd[self.compinv2],{(2,3):1})
        
    def test_find_isomorphism_classes_agg_dict_node_isomorphism_example_dicts(self):
        # test a) aggregated dict, b) node isomorphism, c) example generation
        dd = collections.defaultdict(dict)
        dd_e = dict()
        subgraph_classification.find_isomorphism_classes(self.testnet,2,2,'this_file_should_not_exist',[0],dd,dd_e)
        self.assertEqual(len(dd),3)
        self.assertEqual(dd[self.node_compinv1],{(0,1):2,(2,3):1})
        self.assertEqual(dd[self.node_compinv2],{(2,3):1})
        self.assertEqual(dd[self.node_compinv3],{(1,2):1})
        self.assertEqual(len(dd_e),3)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv1],[0]),self.node_compinv1)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv2],[0]),self.node_compinv2)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv3],[0]),self.node_compinv3)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv1]),self.compinv1)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv2]),self.compinv2)
        self.assertEqual(pn.get_complete_invariant(dd_e[self.node_compinv3]),self.compinv1)



class test_network_io(unittest.TestCase):
    
    timeseries0 = np.array(range(-10,11),dtype=np.float64)
    timeseries1 = np.array([-0.46362334, -0.56982772,  0.06455791, -0.44209878, -0.31068497,
        0.05360425, -0.41299186, -0.29082169, -0.07190158, -0.12474256,
       -0.24997589,  0.01267206,  0.03601663,  0.29330202, -0.12646342,
        0.13130587,  0.57496159,  0.77851974,  0.12816724,  0.63563011,
        0.35058168],dtype=np.float64)
    timeseries2 = np.array(range(100,121),dtype=np.float64)
    timeseries3 = np.zeros((21,))
    timeseries4 = np.array([1,2,3,4,5,6,7,8,9,10,11,10,9,8,7,6,5,4,3,2,1],dtype=np.float64)
    timeseries5 = np.array(range(0,-21,-1))
    
    timeseries6 = np.copy(timeseries0)
    timeseries7 = np.copy(timeseries1)
    timeseries8 = np.copy(timeseries2)
    timeseries9 = np.copy(timeseries3)
    timeseries10 = np.copy(timeseries4)
    timeseries11 = np.copy(timeseries5)
    
    imgdata = np.block([[[[timeseries0],[timeseries1]],[[timeseries2],
                          [timeseries3]],[[timeseries4],[timeseries5]]],
                        [[[timeseries6],[timeseries7]],[[timeseries8],
                          [timeseries9]],[[timeseries10],[timeseries11]]]])
                          
    pickle_test_mplex = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
    pickle_test_mplex['(1, 2, 3)',0]['(2, 3, 4)',0] = 0.5
    pickle_test_mplex['(2, 3, 4)',1]['(3, 4, 5)',1] = 0.999
    pickle_test_mplex['(1, 2, 3)',1]['(2, 3, 4)',1] = 0.001
    pickle_test_mplex['(1, 2, 3)',1]['(3, 4, 5)',1] = 0.234
    pickle_test_mplex['(3, 4, 5)',2]['(1, 2, 3)',2] = 1
    
    pickle_test_mlayer = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    pickle_test_mlayer['[(1, 2, 3),(2, 3, 4)]',9]['[(3, 4, 5)]',9] = 0.123
    pickle_test_mlayer['[(1, 2, 3)]',10]['[(2, 3, 4),(3, 4, 5)]',10] = 0.456
    pickle_test_mlayer['[(1, 2, 3),(2, 3, 4)]',9]['[(1, 2, 3)]',10] = 0.5
    pickle_test_mlayer['[(1, 2, 3),(2, 3, 4)]',9]['[(2, 3, 4),(3, 4, 5)]',10] = 0.333
    pickle_test_mlayer['[(3, 4, 5)]',9]['[(1, 2, 3)]',10] = 0
    pickle_test_mlayer['[(3, 4, 5)]',9]['[(2, 3, 4),(3, 4, 5)]',10] = 0.5
    pickle_test_mlayer['[(4,5,6)]',10]['[(2, 3, 4),(3, 4, 5)]',10] = 0.01
    pickle_test_mlayer['[(2, 3, 4)]',11]['[(2, 3, 4),(3, 4, 5)]',10] = 0.999
    
    def round_edge_weights(self,M):
        # rounds edge weights to 10 decimals
        if isinstance(M,pn.net.MultiplexNetwork):
            for edge in M.edges:
                if edge[2] == edge[3]:
                    M[edge[0],edge[2]][edge[1],edge[3]] = round(edge[4],10)
            return M
        else:
            for edge in M.edges:
                M[edge[0],edge[2]][edge[1],edge[3]] = round(edge[4],10)
            return M
    
    def test_write_weighted_network(self):
        M_multiplex = pn.MultiplexNetwork(couplings='ordinal',fullyInterconnected=True)
        M_multiplex['(1, 2, 3)',0]['(2, 3, 4)',0] = 0.5
        M_multiplex['(2, 3, 4)',1]['(3, 4, 5)',1] = 0.999
        M_multiplex['(3, 4, 5)',2]['(1, 2, 3)',2] = 1
        possible_nodelines = set(['(1, 2, 3);(2, 3, 4);(3, 4, 5)\n',
                                  '(1, 2, 3);(3, 4, 5);(2, 3, 4)\n',
                                  '(2, 3, 4);(1, 2, 3);(3, 4, 5)\n',
                                  '(2, 3, 4);(3, 4, 5);(1, 2, 3)\n',
                                  '(3, 4, 5);(1, 2, 3);(2, 3, 4)\n',
                                  '(3, 4, 5);(2, 3, 4);(1, 2, 3)\n'])
        possible_layerlines = set(['0;1;2\n','0;2;1\n','1;0;2\n','1;2;0\n','2;0;1\n','2;1;0\n'])
        edgeset = set(['(1, 2, 3);(2, 3, 4);0;0;0.5\n',
                    '(1, 2, 3);(1, 2, 3);0;1;1.0\n',
                    '(1, 2, 3);(1, 2, 3);1;2;1.0\n',
                    '(1, 2, 3);(3, 4, 5);2;2;1\n',
                    '(2, 3, 4);(2, 3, 4);0;1;1.0\n',
                    '(2, 3, 4);(3, 4, 5);1;1;0.999\n',
                    '(2, 3, 4);(2, 3, 4);1;2;1.0\n',
                    '(3, 4, 5);(3, 4, 5);0;1;1.0\n',
                    '(3, 4, 5);(3, 4, 5);1;2;1.0\n'])
        network_io.write_weighted_network(M_multiplex,'test_for_network_writing_WILL_BE_REMOVED.txt','Created by test_write_weighted_network')
        try:
            with open('test_for_network_writing_WILL_BE_REMOVED.txt','r') as f:
                self.assertEqual(f.readline(),'# Multiplex\n')
                self.assertEqual(f.readline(),'# Created by test_write_weighted_network\n')
                self.assertEqual(f.readline(),'nodes:\n')
                self.assertTrue(f.readline() in possible_nodelines)
                self.assertEqual(f.readline(),'layers:\n')
                self.assertTrue(f.readline() in possible_layerlines)
                self.assertEqual(f.readline(),'edges:\n')
                for line in f:
                    self.assertTrue(line in edgeset)
                    edgeset.remove(line)
                self.assertEqual(len(edgeset),0)
                self.assertEqual(f.readline(),'')
        finally:
            os.remove('test_for_network_writing_WILL_BE_REMOVED.txt')
        
        M_multilayer = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
        M_multilayer['[(1, 2, 3),(2, 3, 4)]',0]['[(3, 4, 5)]',0] = 0.123
        M_multilayer['[(1, 2, 3)]',1]['[(2, 3, 4),(3, 4, 5)]',1] = 0.456
        M_multilayer['[(1, 2, 3),(2, 3, 4)]',0]['[(1, 2, 3)]',1] = 0.5
        M_multilayer['[(1, 2, 3),(2, 3, 4)]',0]['[(2, 3, 4),(3, 4, 5)]',1] = 0.333
        M_multilayer['[(3, 4, 5)]',0]['[(1, 2, 3)]',1] = 0
        M_multilayer['[(3, 4, 5)]',0]['[(2, 3, 4),(3, 4, 5)]',1] = 0.5
        possible_nodelines = set([x[0]+';'+x[1]+';'+x[2]+';'+x[3]+'\n' for x in itertools.permutations(
                    ['[(1, 2, 3)]','[(1, 2, 3),(2, 3, 4)]','[(2, 3, 4),(3, 4, 5)]','[(3, 4, 5)]'])])
        possible_layerlines = set(['0;1\n','1;0\n'])
        edgeset = set(['[(1, 2, 3),(2, 3, 4)];[(2, 3, 4),(3, 4, 5)];0;1;0.333\n',
                    '[(1, 2, 3),(2, 3, 4)];[(3, 4, 5)];0;0;0.123\n',
                    '[(1, 2, 3),(2, 3, 4)];[(1, 2, 3)];0;1;0.5\n',
                    '[(3, 4, 5)];[(2, 3, 4),(3, 4, 5)];0;1;0.5\n',
                    '[(1, 2, 3)];[(2, 3, 4),(3, 4, 5)];1;1;0.456\n'])
        network_io.write_weighted_network(M_multilayer,'test_for_network_writing_WILL_BE_REMOVED.txt','Created by test_write_weighted_network')
        try:
            with open('test_for_network_writing_WILL_BE_REMOVED.txt','r') as f:
                self.assertEqual(f.readline(),'# Multilayer\n')
                self.assertEqual(f.readline(),'# Created by test_write_weighted_network\n')
                self.assertEqual(f.readline(),'nodes:\n')
                self.assertTrue(f.readline() in possible_nodelines)
                self.assertEqual(f.readline(),'layers:\n')
                self.assertTrue(f.readline() in possible_layerlines)
                self.assertEqual(f.readline(),'edges:\n')
                for line in f:
                    self.assertTrue(line in edgeset)
                    edgeset.remove(line)
                self.assertEqual(len(edgeset),0)
                self.assertEqual(f.readline(),'')
        finally:
            os.remove('test_for_network_writing_WILL_BE_REMOVED.txt')
    
    def test_read_weighted_network(self):
        M_multiplex = network_construction.make_multiplex(self.imgdata,timewindow=7,overlap=2)
        network_io.write_weighted_network(M_multiplex,'test_for_network_reading_WILL_BE_REMOVED.txt','Created by test_read_weighted_network')
        M_multiplex_read = network_io.read_weighted_network('test_for_network_reading_WILL_BE_REMOVED.txt')
        try:
            self.assertEqual(self.round_edge_weights(M_multiplex),self.round_edge_weights(M_multiplex_read))
        finally:
            os.remove('test_for_network_reading_WILL_BE_REMOVED.txt')
        
        M_multilayer = network_construction.make_clustered_multilayer(self.imgdata,timewindow=7,overlap=2,n_clusters=3)
        network_io.write_weighted_network(M_multilayer,'test_for_network_reading_WILL_BE_REMOVED.txt','Created by test_read_weighted_network')
        M_multilayer_read = network_io.read_weighted_network('test_for_network_reading_WILL_BE_REMOVED.txt')
        try:
            self.assertEqual(self.round_edge_weights(M_multilayer),self.round_edge_weights(M_multilayer_read))
        finally:
            os.remove('test_for_network_reading_WILL_BE_REMOVED.txt')
            
    def test_pickle_file_io_for_networks(self):
        try:
            network_io.write_pickle_file(self.pickle_test_mplex,'test_for_pickle_io_mplex_network_WILL_BE_REMOVED.pkl')
            network_io.write_pickle_file(self.pickle_test_mlayer,'test_for_pickle_io_mlayer_network_WILL_BE_REMOVED.pkl')
            pickle_test_mplex_read = network_io.read_pickle_file('test_for_pickle_io_mplex_network_WILL_BE_REMOVED.pkl')
            pickle_test_mlayer_read = network_io.read_pickle_file('test_for_pickle_io_mlayer_network_WILL_BE_REMOVED.pkl')
            self.assertEqual(self.pickle_test_mplex,pickle_test_mplex_read)
            self.assertEqual(self.pickle_test_mlayer,pickle_test_mlayer_read)
        finally:
            if os.path.exists('test_for_pickle_io_mplex_network_WILL_BE_REMOVED.pkl'):
                os.remove('test_for_pickle_io_mplex_network_WILL_BE_REMOVED.pkl')
            if os.path.exists('test_for_pickle_io_mlayer_network_WILL_BE_REMOVED.pkl'):
                os.remove('test_for_pickle_io_mlayer_network_WILL_BE_REMOVED.pkl')
    
    def test_write_layersetwise_network(self):
        try:
            os.mkdir('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED')
            network_io.write_layersetwise_network(self.pickle_test_mplex,'dir_for_test_write_layersetwise_network_WILL_BE_REMOVED')
            network_io.write_layersetwise_network(self.pickle_test_mlayer,'dir_for_test_write_layersetwise_network_WILL_BE_REMOVED')
            self.assertEqual(sorted(os.listdir('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED')),['0_1_2','9_10_11'])
            mplex = network_io.read_pickle_file('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/0_1_2')
            mlayer = network_io.read_pickle_file('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/9_10_11')
            self.assertEqual(self.pickle_test_mplex,mplex)
            self.assertEqual(self.pickle_test_mlayer,mlayer)
        finally:
            if os.path.exists('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/0_1_2'):
                os.remove('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/0_1_2')
            if os.path.exists('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/9_10_11'):
                os.remove('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED/9_10_11')
            if os.path.exists('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED'):
                os.rmdir('dir_for_test_write_layersetwise_network_WILL_BE_REMOVED')



class test_statistics(unittest.TestCase):
    
    def test_t_tests_for_two_dict_lists(self):
        l1 = [{'a':{(0,1):5,(1,2):6}},{'a':{(0,1):6,(1,2):7}}]
        l2 = [{'a':{(0,1):0,(1,2):1}},{'a':{(0,1):4,(1,2):4}},{'b':{(0,1):6}}]
        _,pa0 = statistics.scipy.stats.ttest_ind([5,6],[0,4,0],equal_var=False)
        _,pa1 = statistics.scipy.stats.ttest_ind([6,7],[1,4,0],equal_var=False)
        _,pb0 = statistics.scipy.stats.ttest_ind([0,0],[0,0,6],equal_var=False)
        res = statistics.t_tests_for_two_dict_lists(l1,l2,equal_var=False)
        self.assertEqual(pa0,res['a'][(0,1)][1])
        self.assertEqual(pa1,res['a'][(1,2)][1])
        self.assertEqual(pb0,res['b'][(0,1)][1])
        self.assertTrue(np.isnan(res['b'][(1,2)][1]))
        
        
        
class test_null_models(unittest.TestCase):
    M = pn.MultilayerNetwork(aspects=1,fullyInterconnected=False)
    M.add_layer(0)
    M.add_layer(1)
    M.add_layer(2)
    M.add_node('a',0)
    M.add_node('b',0)
    M.add_node('c',0)
    M.add_node('c',1)
    M.add_node('d',1)
    M.add_node('e',1)
    M.add_node('f',1)
    M.add_node('g',2)
    M.add_node('h',2)
    M['a','b',0] = M['a','c',0] = M['c','d',1] = M['c','e',1] = M['g','h',2] = 1
    M['c','c',0,1] = M['c','d',0,1] = M['c','g',1,2] = M['d','g',1,2] = M['e','g',1,2] = 1
    
    def test_ER_from_net(self):
        M_null = null_models.ER_from_net(self.M)
        self.assertEqual(list(self.M.iter_node_layers()),list(M_null.iter_node_layers()))
        self.assertEqual(list(self.M.iter_layers()),list(M_null.iter_layers()))
        intra_0 = 0
        intra_1 = 0
        intra_2 = 0
        inter_01 = 0
        inter_12 = 0
        for e in M_null.edges:
            if e[2] == e[3]:
                if e[2] == 0:
                    intra_0 = intra_0 + 1
                elif e[2] == 1:
                    intra_1 = intra_1 + 1
                elif e[2] == 2:
                    intra_2 = intra_2 + 1
            else:
                if (e[2] == 0 and e[3] == 1) or (e[2] == 1 and e[3] == 0):
                    inter_01 = inter_01 + 1
                elif (e[2] == 1 and e[3] == 2) or (e[2] == 2 and e[3] == 1):
                    inter_12 = inter_12 + 1
        self.assertEqual(intra_0,2)
        self.assertEqual(intra_1,2)
        self.assertEqual(intra_2,1)
        self.assertEqual(inter_01,2)
        self.assertEqual(inter_12,3)
        


if __name__ == '__main__':
    loader = unittest.TestLoader()
    fullsuite = unittest.TestSuite([loader.loadTestsFromTestCase(test_network_construction),
                                    loader.loadTestsFromTestCase(test_network_io),
                                    loader.loadTestsFromTestCase(test_subgraph_classification),
                                    loader.loadTestsFromTestCase(test_statistics),
                                    loader.loadTestsFromTestCase(test_null_models)])
    unittest.TextTestRunner(verbosity=2).run(fullsuite)
