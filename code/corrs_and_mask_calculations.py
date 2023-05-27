import numpy as np

def calculate_correlation_matrix(imgdata,exclude_masked=True):
    '''
    If exclude_masked == False:
        Calculate correlation between all voxels.
        All 0 series (masked voxels) gives NaN!
    If exclude_masked == True:
        Calculate correlation between non-masked voxels.
        Creates pxp matrix, where p is the number of non-masked voxels.
        Creates also a list where non-masked voxels are listed in the same order as in the correlation matrix.
    In either case, diagonal is set to 1 (correlation with itself)
    '''
    if exclude_masked:
        unmasked_voxels = find_unmasked_voxels(imgdata)
        return calculate_correlation_matrix_for_voxellist(imgdata,unmasked_voxels)
    else:
        flattened_data = imgdata.reshape((-1,imgdata.shape[3]),order='C')
        R = np.corrcoef(flattened_data,rowvar=True)
        return R
        
def calculate_correlation_matrix_for_voxellist(imgdata,voxellist):
    R = np.zeros((len(voxellist),len(voxellist)))
    for ii in range(len(voxellist)):
        for jj in range(ii,len(voxellist)):
            corr = np.corrcoef(imgdata[voxellist[ii]],imgdata[voxellist[jj]])[0][1]
            R[ii,jj] = corr
            R[jj,ii] = corr
    return R,voxellist
        
def find_masked_voxels(imgdata):
    # Returns list of masked voxels (i.e. coordinate tuples)
    # E.g. [(2, 1, 0), (3, 0, 0)]
    return list(zip(*np.where(np.any(imgdata != 0, 3) == False)))
    
def find_unmasked_voxels(imgdata):
    # Returns list of unmasked voxels (i.e. coordinate tuples)
    # E.g. [(1, 1, 0), (2, 0, 0)]
    return list(zip(*np.where(np.any(imgdata != 0, 3) == True)))
    
def calculate_number_of_nan_voxels(R):
    '''
    Calculate how many voxels make the NaN values in correlation matrix R.
    Formula:
    n = number of NaN voxels
    nNaN = number of NaN entries in R
    n(2*nvoxels-1)-(n*n-n) = nNaN
    <=> n(2*nvoxels-n)-nNaN = 0
    '''
    nNaN = np.argwhere(np.isnan(R)).shape[0]
    nvoxels = R.shape[0]
    return np.min(np.roots([-1,2*nvoxels,-1*nNaN]))
    
def check_symmetry(R,tol=1e-8):
    return np.allclose(R,R.T,atol=tol)
    
def nan_and_diag_to_zero(R):
    # NB! fill_diagonal operates in-place, modifies R
    np.fill_diagonal(R,0.0)
    return np.nan_to_num(R,copy=False)
    
def make_adjacency_matrix(imgdata,exclude_masked=True):
    if exclude_masked:
        R,voxellist = calculate_correlation_matrix(imgdata,exclude_masked)
        np.fill_diagonal(R,0.0)
        return R,voxellist
    else:
        R = calculate_correlation_matrix(imgdata,exclude_masked)
        return nan_and_diag_to_zero(R)

#################### Masking out non-gray-matter, for clustering ###########################################################################

def gray_mask(imgdata,gray_matter_mask):
    # Modifies imgdata in-place
    assert imgdata.shape[0:3] == gray_matter_mask.shape
    timeseries_length = imgdata.shape[3]
    for ii in range(gray_matter_mask.shape[0]):
        for jj in range(gray_matter_mask.shape[1]):
            for kk in range(gray_matter_mask.shape[2]):
                if gray_matter_mask[ii,jj,kk] == 0:
                    imgdata[ii,jj,kk] = np.zeros(timeseries_length)
