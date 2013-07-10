"""
This module contains some functions for creating feature vectors. 
"""
import numpy as np
from parameters import Parameters

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
    PC_3s = compute_pcs(X_nsc[:,:,0])[:FPC]  # FPC x nchannels 
    if Parameters['SHOW_PCS']:
        import matplotlib.pyplot as plt
        for i in xrange(FPC):
            plt.plot(PC_3s[i])
        plt.show()
    return PC_3s

def project_features(PC_3s, X_sc):
    return 100.*np.dot(PC_3s, X_sc).T
