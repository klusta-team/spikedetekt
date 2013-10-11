'''
Default values for global parameters
'''

DTYPE = "i2" # ">i2" (> means big-endian), "i4", "f2"
# see http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing

# Probe file (no default value provided)
#PROBE_FILE = 'probe_filename.probe'

# Raw data files (no default values provided)
#RAW_DATA_FILES = ['file1.dat', 'file2.dat']
#NCHANNELS = 32
#SAMPLERATE = 20000 # in Hertz

# Output directory, files are inserted in OUTPUT_DIR/OUTPUT_NAME
OUTPUT_DIR = None # the output directory, use params directory if None
OUTPUT_NAME = None # the filename for created directories, use params filename if None

# Thresholding
USE_SINGLE_THRESHOLD = False # use a single threshold for all channels
CHUNKS_FOR_THRESH = 5 # number of chunks used to determine threshold for detection
THRESH_SD = 4.5 # threshold for detection. standard deviations of signal
DETECT_POSITIVE = False # detect spikes with positive threshold crossing

# Recording data in HDF5 file
RECORD_RAW = True      # raw data
RECORD_HIGH = True     # high pass filtered data
RECORD_LOW = True      # low pass filtered data

# Options for filtering
F_LOW = 500. # low pass frequency (Hz)
F_HIGH_FACTOR = 0.95 # high pass frequency as a proportion of the Nyquist freq, used to derive F_HIGH, i.e. F_HIGH = 0.95*SAMPLERATE/2 here
BUTTER_ORDER = 3 # Order of butterworth filter
WRITE_FIL_FILE = True # write filtered output to .fil file
WRITE_BINFIL_FILE = True # write filtered output to .fil file

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
PCA_MAXWAVES = 10000 # number of waves to use to extract principal components
SHOW_PCS = False # show principal components

# Options for masking
USE_FLOAT_MASKS = True
USE_INTERPOLATION = True
ADDITIONAL_FLOAT_PENUMBRA = 2 # adds some more penumbra
FLOAT_MASK_THRESH_SD = (0, 4.5) # (min, max), mask 0 at min, 1 at max
FLOAT_MASK_INTERPOLATION = 'x' # f(x) for x in [0,1], f(0)=0, f(1)=1

# Options for computing in chunks
CHUNK_SIZE = 20000   # number of time samples used in chunk for filtering and detection
CHUNK_OVERLAP_SECONDS = 0.01 # overlap time (in seconds) of chunks, should be wider than spike width

# Maximum number of spikes to process
MAX_SPIKES = None # None for all spikes, or an int

#Options for writing .xml file
WRITE_XML_FILE = True #If you already have an .xml file for use with the Neuroscope, Klusters suite, you can set this to False and it will not be overwritten
VOLTAGE_RANGE = 20
AMPLIFICATION = 1000
OFFSET = 2048


# Experimental options
DO_GLOBAL_CLUSTERING = False
SORT_CLUS_BY_CHANNEL = False # Sort clusters by the channel where the peak occurs
KEEP_OLD_HDF5_FILES = False
DEBUG = False




