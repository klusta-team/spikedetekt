'''
Default values for global parameters
'''

DTYPE = "i2" # ">i2" (> means big-endian), "i4", "f2"
# see http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing

#SHANK_NUM = 1 #Number of shanks in a multishank probe, each with separate numbers e.g. Probename.probe.3

# Options for computing in chunks
CHUNK_SIZE = 20000   # number of time samples used in chunk for filtering and detection
CHUNK_OVERLAP = 200 # number of samples that chunks overlap in time

# Options for filtering
F_LOW = 500. # low pass frequency (Hz)
BUTTER_ORDER = 3 # Order of butterworth filter
WRITE_FIL_FILE = True # write filtered output to .fil file

# Thresholding
USE_SINGLE_THRESHOLD = False # use a single threshold for all channels
CHUNKS_FOR_THRESH = 5 # number of chunks used to determine threshold for detection
THRESH_SD = 4.5 # threshold for detection. standard deviations of signal
DETECT_POSITIVE = False # detect spikes with positive threshold crossing

# Options for spike detection
T_BEFORE = .0005 # time before peak in extracted spike
T_AFTER = .0005 # time after peak in extracted spike
T_JOIN_CC = .0005 # maximum time between two samples for them to be "contiguous" in detection step
PENUMBRA_SIZE = 0 # mask penumbra size (0 no penumbra, 1 first neighbours, etc.)

# Options for alignment
USE_WEIGHTED_MEAN_PEAK_SAMPLE = True # used for aligning waves
UPSAMPLING_FACTOR = 10 # used for aligning waves

# Options for features
FPC = 3 # Features per channel
SHOW_PCS = False # show principal components

# Options for masking
USE_FLOAT_MASKS = True
USE_INTERPOLATION = False
ADDITIONAL_FLOAT_PENUMBRA = 2 # adds some more penumbra
FLOAT_MASK_THRESH_SD = (2, 4.5) # (min, max), mask 0 at min, 1 at max
FLOAT_MASK_INTERPOLATION = 'x' # f(x) for x in [0,1], f(0)=0, f(1)=1

# Experimental options
DO_GLOBAL_CLUSTERING = False
SORT_CLUS_BY_CHANNEL = False # Sort clusters by the channel where the peak occurs
