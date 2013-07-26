"""
This module contains some functions for creating feature vectors. 
"""
import numpy as np
from parameters import Parameters
from IPython import embed

def compute_pcs(X_ns):
    """Compute principal components of X_ns
    Parameters
    ----------------
    X_ns : array. each row is a data point
    
    Returns
    -----------------
    PC_ss : array. Each row is a principal component. Sorted in increasing order
                   of eigenvalues
    """
    Cov_ss = np.cov(X_ns.T.astype(np.float64))
    Vals, Vecs = np.linalg.eigh(Cov_ss)
    return Vecs.astype(np.float32).T[np.argsort(Vals)[::-1]]

def reget_features(X_nsc):
    FPC = Parameters['FPC']
   # PC_3s = compute_pcs(X_nsc[:,:,0])[:FPC]  # FPC x Parameters['S_TOTAL']
    n_ch = Parameters['N_CH']
    s_tot = Parameters['S_TOTAL']
    PC_3s = np.zeros((FPC,s_tot,n_ch))
    for j in xrange(n_ch):
        PC_3s[:,:,j] = compute_pcs(X_nsc[:,:,j])[:FPC]  # FPC x Parameters['S_TOTAL'] x Parameters['N_CH']
    print 'PC_3s', PC_3s.shape
    print 'X_nsc', X_nsc.shape # Number of spikes x Number of samples per Spike x Parameters['N_CH']
    if Parameters['SHOW_PCS']:
        import matplotlib.pyplot as plt
        for i in xrange(FPC):
            plt.plot(PC_3s[i])
        plt.show()
     #embed()
    return PC_3s

def project_features(PC_3s, X_sc):
   # return 100.*np.dot(PC_3s, X_sc).T
    return 100.*np.einsum('ijk,jk->ki',PC_3s, X_sc) #Notice the transposition
