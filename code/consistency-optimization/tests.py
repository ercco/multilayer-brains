# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:18:58 2018

@author: aokorhon

A script for testing functions of the consistency as onion project
"""
import clustering_by_consistency as cbc

import numpy as np
import nibabel as nib

testDefSphericalROIs = False
testFindROIlessVoxels = False
testFindROIlessNeighbors = False
testUpdateROI = False
testGrowROIs = False
testFindROICentroids = False
testRemoveVoxel = False
testFindVoxelLabels = False
testFindROIBoundary = False
testVoxelLabelsToROIInfo = False
testVoxelsInClustersToROIInfo = False
testFindInROINeighbors = False
testIsBoundary = False
testGetKendallW = True

# testing defSphericalROIs
if testDefSphericalROIs: 
    ROIPath = 'atlases/brainnetome/brainnetome_thr25_4mm_rois.mat'
    centroids, _, voxelCoords, _ = cbc.readROICentroids(ROIPath, readVoxels=True)
    
    # testcase 1: ROIs with radius < 1 should contain only ROI centroids
    spheres = cbc.defineSphericalROIs(centroids, voxelCoords, 0)
    sphereMaps = []
    for sphereMap in spheres['ROIMaps']:
        sphereMaps.append(sphereMap)
    sphereMaps = np.array(sphereMaps)
    
    diff = np.sum(np.abs(sphereMaps - centroids))
    
    if diff == 0:
        print("defSphericalROIs: testcase 1/3 OK")
        
    # testcase 2: small example space
    space = np.zeros((4,4,4))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((64,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    centroids = np.array([[2,2,2],[1,2,3]])
    map1 = np.array([[2,2,2],[1,2,2],[2,1,2],[2,2,1],[3,2,2],[2,3,2],[2,2,3]])
    map1 = np.sort(map1, axis=0)
    map2 = np.array([[1,2,3],[0,2,3],[1,1,3],[1,2,2],[2,2,3],[1,3,3]]) #[1,2,4] is outside of the space and thus not included
    map2 = np.sort(map2, axis=0)
    
    spheres = cbc.defineSphericalROIs(centroids, voxels, 1, resolution=1)
    diff = 0
    for sphereMap,trueMap in zip(spheres['ROIMaps'],[map1,map2]):
        sphereMap = np.sort(sphereMap, axis=0) # sorts the array columnwise; distorts the original rows but gives same output for all arrays with the same contents (independent of the row order)
        diff = diff + np.sum(np.abs(sphereMap - trueMap))
    
    if diff == 0:
        print("defSphericalROIs: testcase 2/3 OK")
        
    # testcase 3: changing sphere size:
    map3 = np.array([[2,2,2],[1,2,2],[2,1,2],[2,2,1],[3,2,2],[2,3,2],[2,2,3], # centroid + at distance 1
                     [0,2,2],[2,0,2],[2,2,0],[1,1,2],[2,1,1],[1,2,1], # at distance 2
                     [3,3,2],[2,3,3],[3,2,3],[1,3,2],[3,1,2],[2,1,3],[2,3,1],[1,2,3],[3,2,1], # at distance sqrt(2)
                     [1,1,1],[1,3,1],[1,1,3],[3,1,1],[3,3,1],[3,1,3],[1,3,3],[3,3,3]]) #at distance sqrt(3)
    map3 = np.sort(map3, axis=0)    

    spheres = cbc.defineSphericalROIs(centroids[0], voxels, 2, resolution=1)
    sphereMap = np.sort(spheres['ROIMaps'][0], axis=0)
    diff = np.sum(np.abs(sphereMap - map3))
    
    if diff == 0:
        print('defSphericalROIs: testcase 3/3 OK')
        
    # testcase 4: changing resolution
    # NOTE: I've removed the old testcase 4 on 13 Oct 2018: It assumed that voxel coordinates
    # are saved in millimeters while they are actually saved in voxels. So, the testcase didn't
    # match the reality and lead to wrong conclusions.
        
# testing getROIlessVoxels
        
if testFindROIlessVoxels:
    # testcase 1: a very small space with two small ROIs
    space = np.zeros((2,2,2))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((8,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROI1 = np.array([[0,0,0],[0,1,0]])
    ROI2 = np.array([[0,0,1],[1,0,0],[0,1,1]])
    ROIMaps = [ROI1,ROI2]
    ROIVoxels = [np.array([0,1]),np.array([2,3,4])]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels}
    
    trueROIless = np.array([[1,0,1],[1,1,0],[1,1,1]])
    trueROIlessIndices = [5,6,7]
    
    ROIlessVoxels = cbc.findROIlessVoxels(voxels,ROIInfo)
    testROIless = ROIlessVoxels['ROIlessMap']
    testROIlessIndices = ROIlessVoxels['ROIlessIndices']
    
    mapDif = np.sum(np.abs(trueROIless - testROIless))
    indDif = np.sum(np.abs(np.array(trueROIlessIndices)-np.array(testROIlessIndices)))
    
    if mapDif == 0:
        print('findROIlessVoxels: testcase 1/1: maps of ROIless voxels OK')
    if indDif == 0:
        print('findROIlessVoxels: testcase 1/1: indices of ROIless voxels OK')
        
if testFindROIlessNeighbors:
    # testcase 1: a very small space with two small ROIs    
    space = np.zeros((2,2,2))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((8,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
        
    ROI1 = np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])
    ROI2 = np.array([[1,0,1]])
    ROIMaps = [ROI1,ROI2]
    ROIVoxels = [np.array([0,2,6,4]),np.array([5])]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels}
    
    trueROIless1 = np.array([[0,0,1],[0,1,1],[1,1,1],])
    trueROIlessIndices1 = np.array([1,3,7])
    trueROIless2 = np.array([[0,0,1],[1,1,1]])
    trueROIlessIndices2 = np.array([1,7])
    
    testROIlessNeighbors1 = cbc.findROIlessNeighbors(0,voxels,ROIInfo)
    testROIless1 = testROIlessNeighbors1['ROIlessMap']
    testROIlessIndices1 = testROIlessNeighbors1['ROIlessIndices']
    
    mapDif1 = np.sum(np.abs(trueROIless1 - testROIless1))
    indDif1 = np.sum(np.abs(trueROIlessIndices1 - testROIlessIndices1))
    
    testROIlessNeighbors2 = cbc.findROIlessNeighbors(1,voxels,ROIInfo)
    testROIless2 = testROIlessNeighbors2['ROIlessMap']
    testROIlessIndices2 = testROIlessNeighbors2['ROIlessIndices']
    
    mapDif2 = np.sum(np.abs(trueROIless2 - testROIless2))
    indDif2 = np.sum(np.abs(trueROIlessIndices2 - testROIlessIndices2))
    
    if max(mapDif1,mapDif2) == 0:
        print('findROIlessNeighbors: testcase 1/1: maps of ROIless neighbors OK')
        
    if max(indDif1,indDif2) == 0:
        print('findROIlessNeighbors: testcase 1/1: indices of ROIless neighbors OK')

if testUpdateROI:
    # testcase 1: one small ROI in a small space
    space = np.zeros((2,2,2))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((8,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    ROIMaps = [np.array([[0,0,0]])]
    ROIVoxels = [np.array([0])]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels,'ROISizes':np.array([1])}
    
    trueROIMaps = [np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])]
    trueROIVoxels = [np.array([0,4,2,1])]
    
    testROIInfo = cbc.updateROI(0,voxels,ROIInfo)
    testROIMaps = testROIInfo['ROIMaps']
    testROIVoxels = testROIInfo['ROIVoxels']
    
    mapCorrect = []
    indicesCorrect = []
    
    for ROI, testROI, ROIVoxel, testROIVoxel in zip(trueROIMaps,testROIMaps,trueROIVoxels,testROIVoxels):
        if not (len(ROI) == len(testROI)):
            mapCorrect.append(False)
        else:
            tempCorrect = np.zeros(ROI.shape[0])
            for i,testVoxel in enumerate(testROI):
                tempCorrect[i] = np.any((ROI == testVoxel).all(axis=1))
            mapCorrect.append(all(tempCorrect))
        if not (len(ROIVoxel) == len(testROIVoxel)):
            indicesCorrect.append(False)
        else:
            tempCorrect = np.zeros(len(ROIVoxel))
            for i, index in enumerate(testROIVoxel):
                tempCorrect[i] = np.any(ROIVoxel == index)
            indicesCorrect.append(all(tempCorrect))
    
    if all(mapCorrect):
        print('updateROI: testcase 1/2: ROI map OK')
        
    if all(indicesCorrect):
        print('updateROI: testcase 1/2: ROI voxel indices OK')
        
    if testROIInfo['ROISizes'] == np.array([4]):
        print('updateROI: testcase 1/2: ROI sizes OK')
        
    # testcase 2: two small ROIs in a small space:
    ROI1 = np.array([[0,0,0]])
    ROI2 = np.array([[1,0,1]])
    ROIMaps = [ROI1,ROI2]
    ROIVoxels = [np.array([0]),np.array([5])]
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels,'ROISizes':np.array([1,1])}
    
    trueROI1Map = np.array([[0,0,0],[1,0,0],[0,0,1],[0,1,0]])
    trueROI1Voxels = np.array([0,1,2,4])
    trueROIMaps = [trueROI1Map,np.array([[1,0,1]])]
    trueROIVoxels = [trueROI1Voxels,np.array([5])]
    
    testROIInfo = cbc.updateROI(0,voxels,ROIInfo)
    testROIMaps = testROIInfo['ROIMaps']
    testROIVoxels = testROIInfo['ROIVoxels']
    testROISizes = testROIInfo['ROISizes']
    
    mapCorrect = []
    indicesCorrect = []
    
    for ROI, testROI, ROIVoxel, testROIVoxel in zip(trueROIMaps,testROIMaps,trueROIVoxels,testROIVoxels):
        if not (len(ROI) == len(testROI)):
            mapCorrect.append(False)
        else:
            tempCorrect = np.zeros(ROI.shape[0])
            for i,testVoxel in enumerate(testROI):
                tempCorrect[i] = np.any((ROI == testVoxel).all(axis=1))
            mapCorrect.append(all(tempCorrect))
        if not (len(ROIVoxel) == len(testROIVoxel)):
            indicesCorrect.append(False)
        else:
            tempCorrect = np.zeros(len(ROIVoxel))
            for i, index in enumerate(testROIVoxel):
                tempCorrect[i] = np.any(ROIVoxel == index)
            indicesCorrect.append(all(tempCorrect))
            
    if all(mapCorrect) and len(testROIMaps) == len(trueROIMaps):
        print('updateROI: testcase 2/2: ROI maps OK')
        
    if all(indicesCorrect) and len(testROIVoxels) == len(trueROIVoxels):
        print('updateROI: testcase 2/2: ROI voxel indices OK')
        
    if np.all(testROISizes == np.array([7,1])):
        print('updateROI: testcase 2/2: ROI sizes OK')
        
if testGrowROIs:
    # testcase 1: two small ROIs in a small space
    space = np.zeros((2,2,2))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((8,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROI1 = np.array([[0,0,0]])
    ROI2 = np.array([[1,0,1]])
    ROIMaps = [ROI1,ROI2]
    
    trueROI1 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]])
    trueROI2 = np.array([[1,0,1],[1,1,1]])
    trueROI1Voxels = np.array([0,1,2,4,6,3])
    trueROI2Voxels = np.array([5,7])
    
    trueROIMaps = [trueROI1,trueROI2]
    trueROIVoxels = [trueROI1Voxels,trueROI2Voxels]
    
    testROIInfo = cbc.growROIs(ROIMaps,voxels,names='')
    testROIMaps = testROIInfo['ROIMaps']
    testROIVoxels = testROIInfo['ROIVoxels']
    testROISizes = testROIInfo['ROISizes']
    
    mapCorrect = []
    indicesCorrect = []
    
    for ROI, testROI, ROIVoxel, testROIVoxel in zip(trueROIMaps,testROIMaps,trueROIVoxels,testROIVoxels):
        if not (len(ROI) == len(testROI)):
            mapCorrect.append(False)
        else:
            tempCorrect = np.zeros(ROI.shape[0])
            for i,testVoxel in enumerate(testROI):
                tempCorrect[i] = np.any((ROI == testVoxel).all(axis=1))
            mapCorrect.append(all(tempCorrect))
        if not (len(ROIVoxel) == len(testROIVoxel)):
            indicesCorrect.append(False)
        else:
            tempCorrect = np.zeros(len(ROIVoxel))
            for i, index in enumerate(testROIVoxel):
                tempCorrect[i] = np.any(ROIVoxel == index)
            indicesCorrect.append(all(tempCorrect))
            
    if all(mapCorrect) and len(testROIMaps) == len(trueROIMaps):
        print('growROIs: testcase 1/1: ROI maps OK')
        
    if all(indicesCorrect) and len(testROIVoxels) == len(trueROIVoxels):
        print('growROI: testcase 1/1: ROI voxel indices OK')
        
    if np.all(testROISizes == np.array([6,2])):
        print('growROI: testcase 1/1: ROI sizes OK')
    
if testFindROICentroids:
    # Testcase 1: comparing against reading from Matlab file (that's assumed to be correct) without fixing the centroids
    templateMatFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.mat'
    trueROICentroids,_,trueVoxelCoordinates,trueROIMaps = cbc.readROICentroids(templateMatFile,readVoxels=True)
    
    templateNiiFile = '/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii'
    templateimg = nib.load(templateNiiFile)
    template = templateimg.get_fdata()
    testROICentroids,testVoxelCoordinates,testROIMaps = cbc.findROICentroids(template,fixCentroids=False)
    
    if (testROICentroids == trueROICentroids).all(axis=1).all():
        print('findROICentroids: testcase 1/2: ROI centroids OK')
    else:
        print('findROICentroids: testcase 1/2: ERROR IN ROI CENTROIDS!')
        
    if testVoxelCoordinates.shape == trueVoxelCoordinates.shape and np.array([coord in trueVoxelCoordinates for coord in testVoxelCoordinates]).all():
        print('findROICentroids: testcase 1/2: voxel coordinates OK')
    else:
        print('findROICentroids: testcase 1/2: ERROR IN VOXEL COORDINATES!')
        
    ROIMapBools = np.zeros(len(trueROIMaps))
    
    for i, (testROIMap, trueROIMap) in enumerate(zip(testROIMaps,trueROIMaps)):
        if testROIMap.shape == trueROIMap.shape and np.array(testVoxel in trueROIMap for testVoxel in testROIMap).all:
            ROIMapBools[i] = True
    
    if ROIMapBools.all():
        print('findROICentroids: testcase 1/2: ROI maps OK')
    else:
        print('findROICentroids: testcase 1/2: ERROR IN ROI MAPS!')
    
    # Testcase 2: fixing the centroids
    # This case doesn't pass as the voxels are in different orders in mat and nii files and if multiple
    # voxels are at the same distance from the centroid, the first one is selected
    # TODO: consider if this could be improved somehow
    
    trueROICentroids,_,_,_ = cbc.readROICentroids(templateMatFile,fixCentroids=True)
    testROICentroids,_,_ = cbc.findROICentroids(template)
    
    if (testROICentroids == trueROICentroids).all(axis=1).all():
        print('findROICentroids: testcase 2/2: fixed ROI centroids OK')
    else:
        print('findROICentroids: testcase 2/2: ERROR IN FIXED ROI CENTROIDS!')
        
# testing removeVoxels
# this is not a proper test that would print OK when passed; this is rather a piece of code for debugging purposes
# before running, check template file path
if testRemoveVoxel:
    template = nib.load('/media/onerva/KINGSTON/test-data/group_roi_mask-30-4mm_with_subcortl_and_cerebellum.nii').get_fdata()
    ROICentroids,voxelCoordinates,ROIMaps = cbc.findROICentroids(template)
    ROIVoxels = []
    for ROI in ROIMaps:
        voxelIndices = []#np.zeros(len(ROI),dtype=int)
        for i, voxel in enumerate(ROI):
            voxelIndices.append(np.where((voxelCoordinates == voxel).all(axis=1))[0][0])
        ROIVoxels.append(voxelIndices)
    ROIInfo = {'ROIMaps':ROIMaps,'ROIVoxels':ROIVoxels}
    cbc.removeVoxel(0,ROIVoxels[0][15],ROIInfo,voxelCoordinates)
    
# testing findVoxelLabels
if testFindVoxelLabels:
    # testcase 1: a very small space with two small ROIs
    space = np.zeros((2,2,2))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((8,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROI1 = np.array([[0,0,0],[0,1,0]])
    ROI2 = np.array([[0,0,1],[1,0,0],[0,1,1]])
    ROIMaps = [ROI1,ROI2]
    ROIInfo = {'ROIMaps':ROIMaps}
    
    trueLabels = np.array([0,1,0,1,1,-1,-1,-1])
    testLabels = cbc.findVoxelLabels(voxels,ROIInfo)
    
    dif = np.sum(np.abs(trueLabels - testLabels))
    
    if dif == 0:
        print('findVoxelLabels: testcase 1/1: labels OK for two small ROIs')
    else:
        print('findVoxelLabels: testcase 1/1: ERROR IN VOXEL LABELS!!!')
    
    
# testing findROIBoundary
if testFindROIBoundary:    
    # testcase 1: a very small space with two small ROIs
    space = np.zeros((3,3,3))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((27,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROI1 = np.array([[1,1,1],[0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]])
    ROI2 = np.array([[0,0,1],[1,0,0]])
    ROIMaps = [ROI1,ROI2]
    ROIInfo = {'ROIMaps':ROIMaps}
    
    trueBoundary1 = np.array([[0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]])
    trueIndices1 = np.array([4,10,12,22,16,14])
    trueLabels1 = [[1],[1],[1],[],[],[]]
    trueBoundary2 = np.array([[0,0,1],[1,0,0]])
    trueIndices2 = np.array([1,9])
    trueLabels2 = [[0],[0]]
    voxelLabels = cbc.findVoxelLabels(voxels,ROIInfo)
    
    testIndices1,testBoundary1,testLabels1 = cbc.findROIBoundary(0,ROIInfo,voxels,voxelLabels)
    testIndices1 = np.array(testIndices1)
    testBoundary1 = np.array(testBoundary1)
    testIndices2,testBoundary2,testLabels2 = cbc.findROIBoundary(1,ROIInfo,voxels,voxelLabels)
    testIndices2 = np.array(testIndices2)
    testBoundary2 = np.array(testBoundary2)
    
    mapDif1 = np.sum(np.abs(trueBoundary1 - testBoundary1))
    indDif1 = np.sum(np.abs(trueIndices1 - testIndices1))
    mapDif2 = np.sum(np.abs(trueBoundary2 - testBoundary2))
    indDif2 = np.sum(np.abs(trueIndices2 - testIndices2))
    
    labelBoolean1 = [testLabel in trueLabels1 for testLabel in testLabels1]
    labelBoolean2 = [testLabel in trueLabels2 for testLabel in testLabels2]

        
    if mapDif1 == 0 and mapDif2 == 0:
        print('findROIBoundary: testcase 1/1: boundary maps of two small ROIs OK')
    else:
        print('findROIBoundary: testcase 1/1: ERROR IN ROI BOUNDARY MAPS!!!')
    if indDif1 == 0 and indDif2 == 0:
        print('findROIBoundary: testcase 1/1: boundary indices of two small ROIs OK')
    else:
        print('findROIBoundary: testcase 1/1: ERROR IN ROI BOUNDARY INDICES!!!')
    if len(testLabels1) == len(trueLabels1) and all(labelBoolean1) and len(testLabels2) == len(trueLabels2) and all(labelBoolean2):
        print('findROIBoundary: testcase 1/1: neighbor ROI labels of two small ROIs OK')
    else:
        print('findROIBoundary: testcase 1/1: ERROR IN NEIGHBOR ROI LABELS')
        
# testing voxelLabelsToROIInfo:
if testVoxelLabelsToROIInfo:
    space = np.zeros((3,3,3))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((27,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROIMap1 = np.array([[1,1,1],[0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]])
    ROIVoxels1 = np.array([13,4,10,12,22,16,14])
    ROIMap2 = np.array([[0,0,1],[1,0,0]])
    ROIVoxels2 = np.array([1,9])
    
    ROIMaps = [ROIMap1, ROIMap2]
    ROIVoxels = [ROIVoxels1, ROIVoxels2]
    
    voxelLabels = np.array([-1,1,-1,-1,0,-1,-1,-1,-1,1,0,-1,0,0,0,-1,0,-1,-1,-1,-1,-1,0,-1,-1,-1,-1])
    
    ROIInfo = cbc.voxelLabelsToROIInfo(voxelLabels,voxels)
    
    if len(ROIInfo['ROIMaps']) != len(ROIMaps):
        mapVar = False
    else:
        mapVar = []
        for testROIMap,trueROIMap in zip(ROIInfo['ROIMaps'], ROIMaps):
            if len(testROIMap) != len(trueROIMap):
                mapVar.append(False)
            else:
                mapVar.append(all([voxel in trueROIMap for voxel in testROIMap]))
        mapVar = all(mapVar)
        
    if len(ROIInfo['ROIVoxels']) != len(ROIVoxels):
        voxelVar = False
    else:
        voxelVar = []
        for testROIVoxels,trueROIVoxels in zip(ROIInfo['ROIVoxels'], ROIVoxels):
            if len(testROIVoxels) != len(trueROIVoxels):
                voxelVar.append(False)
            else:
                voxelVar.append(all([voxel in trueROIVoxels for voxel in testROIVoxels]))
        voxelVar = all(voxelVar)
        
    if mapVar:
        print('voxelLabelsToROIInfo: testcase 1/1: ROIMaps OK')
    else:
        print('voxelLabelsToROIInfo: testcase 1/1: ERROR IN ROIMAPS!!!')
        
    if voxelVar:
        print('voxelLabelsToROIInfo: testcase 1/1: ROIVoxels OK')
    else:
        print('voxelLabelsToROIInfo: testcase 1/1: ERROR IN ROIVOXELS!!!')
        
# testing voxelsInClustersToROIInfo
if testVoxelsInClustersToROIInfo:
    space = np.zeros((3,3,3))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((27,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROIMap1 = np.array([[1,1,1],[0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]])
    ROIVoxels1 = np.array([13,4,10,12,22,16,14])
    ROIMap2 = np.array([[0,0,1],[1,0,0]])
    ROIVoxels2 = np.array([1,9])
    
    ROIMaps = [ROIMap1, ROIMap2]
    ROIVoxels = [ROIVoxels1, ROIVoxels2]
    
    voxelsInClusters = {0:[(1,1,1),(0,1,1),(1,0,1),(1,1,0),(2,1,1),(1,2,1),(1,1,2)],
                        1:[(0,0,1),(1,0,0)],
                        -1:[(0,0,0),(0,0,2),(0,1,0),(0,1,2),(0,2,0),(0,2,1),(0,2,2),(1,0,2),(1,2,0),(1,2,2),(2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]}
    trueVoxelLabels = np.array([0,0,0,0,0,0,0,1,1])
    
    ROIInfo, voxelCoordinates, voxelLabels = cbc.voxelsInClustersToROIInfo(voxelsInClusters)
    
    if len(ROIInfo['ROIMaps']) != len(ROIMaps):
        mapVar = False
    else:
        mapVar = []
        for testROIMap,trueROIMap in zip(ROIInfo['ROIMaps'], ROIMaps):
            if len(testROIMap) != len(trueROIMap):
                mapVar.append(False)
            else:
                mapVar.append(all([voxel in trueROIMap for voxel in testROIMap]))
        mapVar = all(mapVar)
        
    if len(ROIInfo['ROIVoxels']) != len(ROIVoxels):
        voxelVar = False
    else:
        voxelVar = []
        for testROIVoxels,trueROIVoxels in zip(ROIInfo['ROIVoxels'], ROIVoxels):
            if len(testROIVoxels) != len(trueROIVoxels):
                voxelVar.append(False)
            else:
                trueCoordinates = voxels[trueROIVoxels]
                testCoordinates = voxelCoordinates[testROIVoxels]
                voxelVar.append(np.sum(np.abs(trueCoordinates - testCoordinates))==0)
                
        voxelVar = all(voxelVar)

    if mapVar:
        print('voxelsInClustersToROIInfo: testcase 1/1: ROIMaps OK')
    else:
        print('voxelsInClustersToROIInfo: testcase 1/1: ERROR IN ROIMAPS!!!')
        
    if voxelVar:
        print('voxelsInClustersToROIInfo: testcase 1/1: ROIVoxels OK')
    else:
        print('voxelsInClustersToROIInfo: testcase 1/1: ERROR IN ROIVOXELS!!!')
    if all(voxelLabels == trueVoxelLabels):
        print('voxelsInClustersToROIInfo: testcase 1/1: voxel labels OK')
    else:
        print('voxelsInClustersToROIInfo: testcase 1/1: ERROR IN VOXEL LABELS!!!')
        
# testing findInROINeighbors
if testFindInROINeighbors:
    # testcase 1: a small ROI in a very small space
    space = np.zeros((3,3,3))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((27,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
    
    ROIMap = np.array([[1,1,1],[0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]])
    voxelLabels = np.array([-1,-1,-1,-1,0,-1,-1,-1,-1,-1,0,-1,0,0,0,-1,0,-1,-1,-1,-1,-1,0,-1,-1,-1,-1])
    voxelIndex1 = 13 # voxel [1,1,1]
    trueInROINeighborIndices1 = [4,10,12,22,16,14] # voxels [0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2]
    voxelIndex2 = 4 # voxel [0,1,1]
    trueInROINeighborIndices2 = [13] # voxel [1,1,1]
    
    testInROINeighborIndices1 = cbc.findInROINeighbors(voxelIndex1, 0, voxels, voxelLabels)
    testInROINeighborIndices2 = cbc.findInROINeighbors(voxelIndex2, 0, voxels, voxelLabels)
    
    if len(testInROINeighborIndices1) != len(trueInROINeighborIndices1):
        testVar1 = False
    else:
        testVar1 = all([index in trueInROINeighborIndices1 for index in testInROINeighborIndices1])
    if len(testInROINeighborIndices2) != len(trueInROINeighborIndices2):
        testVar2 = False
    else:
        testVar2 = all([index in trueInROINeighborIndices2 for index in testInROINeighborIndices2])
    if testVar1 and testVar2:
        print('findInROINeighbors: testcase 1/1: in-ROI neighbor indices OK')
    else:
        print('findInROINeighbors: testcase 1/1: ERROR IN IN-ROI NEIGHBOR INDICES!!!')
        
# testing isBoundary
if testIsBoundary:
    # testcase 1: small space with two ROIs
    space = np.zeros((3,3,3))
    w1, w2, w3 = np.where(space==0)
    voxels = np.zeros((27,3))
    for i, (x, y, z) in enumerate(zip(w1,w2,w3)):
        voxels[i,:] = [x,y,z]
        
    voxelLabels = np.array([-1,1,-1,-1,0,-1,-1,-1,-1,1,0,-1,0,0,0,-1,0,-1,-1,-1,-1,-1,0,-1,-1,-1,-1])
    
    trueResult = [False, True, True]
    testResult = [cbc.isBoundary(13,voxels,voxelLabels)[0], cbc.isBoundary(4,voxels,voxelLabels)[0],cbc.isBoundary(1,voxels,voxelLabels)[0]]
    
    trueNeighborROIs = [[],[1],[0]]
    testNeighborROIs = [cbc.isBoundary(13,voxels,voxelLabels)[1], cbc.isBoundary(4,voxels,voxelLabels)[1],cbc.isBoundary(1,voxels,voxelLabels)[1]]
    
    if trueResult == testResult:
        print('isBoundary: testcase 1/1: Boundary identified correctly')
    else:
        print('isBoundary: testcase 1/1: ERROR IN BOUNDARY IDENTIFICATION!!!')
        
    neighborVar = []
    for trueNeighborList, testNeighborList in zip(trueNeighborROIs,testNeighborROIs):
        neighborVar.append(testNeighborList == trueNeighborList)
    neighborVar = all(neighborVar)
    if neighborVar:
        print('isBoundary: testcase 1/1: Neighbor ROI indices OK')
    else:
        print('isBoundary: testcase 1/1: ERROR IN NEIGHBOR ROI INDICES!!!')
        
# testing getKendallsW
if testGetKendallW:
    # testcase 1: a bunch of identical time series
    x = np.sin(np.linspace(-np.pi,np.pi,100))
    y = np.sin(np.linspace(-np.pi,np.pi,100))
    z = np.sin(np.linspace(-np.pi,np.pi,100))
    W1 = cbc.getKendallW(np.array([x,y]))
    W2 = cbc.getKendallW(np.array([x,y,z]))
    if W1 == W2 == 1:
        print('getKendallW: testcase 1/2: Kendall W of identical time series OK')
    else:
        print('getKendallW: testcase 1/2: ERROR IN KENDALL W OF IDENTICAL TIME SERIES!!!')
    # testcase 2: time series in antisych
    x2 = np.sin(np.linspace(0,2*np.pi,100))
    W3 = cbc.getKendallW(np.array([x,x2]))
    if W3 == 0:
        print('getKendallW: testcase 2/2: Kendall W of antisynchronized time series OK')
    else:
        print('getKendallW: testcase 2/2: ERROR IN KENDALL W OF ANTISYNCHRONIZED TIME SERIES!!!')
    
    
    

    
    
    
        
    
    
    
    
    
    



