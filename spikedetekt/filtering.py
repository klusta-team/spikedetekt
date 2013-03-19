'''
Routines and algorithms for filtering the data
'''

import numpy as np
from scipy import signal
from parameters import Parameters

def get_filter_params():
    '''
    Get the filter coefficients for the high-pass
    '''
    BUTTER_ORDER = Parameters['BUTTER_ORDER']
    SAMPLE_RATE = Parameters['SAMPLE_RATE']
    F_LOW = Parameters['F_LOW']
    F_HIGH = Parameters['F_HIGH']
    b, a = signal.butter(BUTTER_ORDER,
                        (F_LOW/(SAMPLE_RATE/2), F_HIGH/(SAMPLE_RATE/2)),
                        'pass')
    return b, a

def apply_filtering((b, a), x):
    out_arr = np.zeros_like(x)
    #FilteredChunk = signal.filtfilt(b, a, DatChunk.astype(np.int32), axis=0)
    #FilteredChunk = signal.filtfilt(b, a, DatChunk.astype(np.int32).T).T
    for i_ch in xrange(x.shape[1]):
        out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch]) 
    return out_arr
