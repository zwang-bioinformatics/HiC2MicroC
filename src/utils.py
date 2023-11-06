import math
import numpy as np
import scipy.sparse as sp
import cooler

def diag_indices(n, k):
        rows, cols = np.diag_indices(n)
        rows, cols = list(rows), list(cols)
        rows2, cols2 = rows.copy(), cols.copy()
        for i in range(1, (k+1)):
                rows2 += rows[:-i]
                cols2 += cols[i:]
        return np.array(rows2), np.array(cols2)


def get_submat_idx(chr_len, res=5000, step=50, image_size=256):
    
    n_bins = math.ceil(chr_len / res)
    allInds = np.arange(0, n_bins-image_size, step)
    
    lastInd = allInds[len(allInds)-1]
    if (lastInd + image_size) < n_bins:
        allInds = np.append(allInds, n_bins - image_size)
    
    extend_steps = 5
    if res == 1000:
        extend_steps = 20
    
    idxes = []
    for j in allInds: 
        idx_sj = j
        idx_ej = j + image_size - 1
        psj = idx_sj * res
        pej = idx_ej * res + res
        
        if pej > chr_len:
            pej = chr_len
        
        for k in range(extend_steps):
            idx_sk = j + k*step
            idx_ek = idx_sk + image_size - 1
            psk = idx_sk * res
            pek = idx_ek * res + res
            
            if idx_ek >= n_bins:
                continue
            
            if pek > chr_len:
                pek = chr_len
            
            idxes.append([idx_sj, idx_ej+1, idx_sk, idx_ek+1])
    return idxes


def sparse_divide_nonzero(a, b):
        inv_b = b.copy()
        inv_b.data = 1 / inv_b.data
        return a.multiply(inv_b)
    

def pred_2_raw_score(vals, maxV):
    
    # step 1: [-1, 1] to [0, 1]
    vals2 = (vals + 1) / 2
    
    # mat_micro = np.log10((9 / maxV) * mat_micro + 1)
    # step 2: [0, 1] to [1, 10]
    factor = maxV / 9
    vals2 = (10**vals2 -1) * factor
    
    return vals2


def get_predict_mat(matin, maxValue, chr_len, idxes, res=5000, max_bin=456, step=50, image_size=256):
    
    bins = math.ceil(chr_len / res)

    rows, cols = diag_indices(bins, max_bin - 1)

    # matrix for predicted probabilities
    mp = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
    mp = mp + mp.T - sp.diags(mp.diagonal())

    # matrix for overlapping numbers
    mn = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
    mn = mn + mn.T - sp.diags(mn.diagonal())

    for i, idx in enumerate(idxes):
        i1, i2, j1, j2 = idx[0], idx[1], idx[2], idx[3]
    
        mp[i1:i2, j1:j2] += matin[i,:,:]
        mn[i1:i2, j1:j2] += np.ones((image_size, image_size))

    mp.data -= 1
    mn.data -= 1
    mp.eliminate_zeros()
    mn.eliminate_zeros()
    mpn = sparse_divide_nonzero(mp, mn)

    mpn_2 = sp.coo_matrix(mpn) #
    #mpn_2.sum_duplicates()

    pred_converted = pred_2_raw_score(mpn_2.data, maxValue)
    #pred_converted += abs(np.amin(pred_converted)) # make it >=0

    mpn_2.data = pred_converted
    
    return mpn_2


def get_started_index_for_cooler(chrids, chromsizes, res=5000):
    '''
    index used for generating COO matrices
    '''
    chr_start_index = {}
    cump_index = 0
    for i, chrid in enumerate(chrids):
        chr_len = chromsizes[i]
        num_bins = math.ceil(chr_len / res)
        chr_start_index[chrid] = cump_index
        cump_index += num_bins
    
    return chr_start_index


def add_weight_column(fcool):
    '''
    all bins' bias set to 1.0
    '''
    clr = cooler.Cooler(fcool)
    n_bins = clr.bins().shape[0]

    if 'weight' not in clr.bins().columns:
        h5opts = dict(compression='gzip', compression_opts=6)
        with clr.open('r+') as f:
            # Create a weight column
            f['bins'].create_dataset('weight', data=np.ones(n_bins), **h5opts)

