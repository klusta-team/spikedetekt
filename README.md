SpikeDetekt
-----------

This is a program for spike detection. It is in development.

Contact: Kenneth Harris (firstname at cortexlab.net), Shabnam Kadir (firstname at cortexlab.net)

Quick Start Guide for electrophysiologists (will become more comprehensive with time):
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

0) Installation
----------------

We recommend you use Python 2.6 or 2.7, e.g. a free academic version can be obtained from [Enthought Python](http://enthought.com/products/epd.php).



Once you have set up Python on your system, go to the SpikeDetekt folder and type (on the command line):

    python setup.py install

This will install SpikeDetekt.

1) Probefiles:
---------------

Below are the instructions for a multi-shank probe (I hope this is clear from my example probe - otherwise do ask):

Construct a probe object from a .probe file with:

   probe = Probe(filename)

The file should have a form something like:

    probes = {
        1: [
            (0, 1), (0, 2),
            (1, 2), (1, 3),
            ...
            ],
        2: [...],
        ...
        }

The file is a Python file which should define a dictionary variable probes,
with keys the shank number, and values a list of channel pairs defining the
edges of the graph.

The Probe object has the following attributes:

* num_channels  
The number of channels used (1+the maximum channel number referred to)

* channel_graph 
 A dictionary with keys the shank number, and values being graphs. A graph being a dictionary with keys the nodes (channel number) and values the set of all connected nodes. (So each channel in an edge is referred to twice in this data structure.)

* shanks_set
       The set of shank numbers
   
* channel_set
       A dictionary with keys the shank numbers, and values the 
set of channels for that shank

* channel_to_shank
       A dictionary with keys the channel numbers and values the corresponding shank number.

* probes
       The raw probes dictionary definition in the file.



I have included some examples of probe files:

* buzsaki32.probe
* linear16.probe
* multishankslinear32.probe (an 8 shank example)


2) Parameters to adjust
----------------------------

Please alter the following two lines in your parameters.py file (I have included the file example_parameters.py)

    # Options for computing in chunks
    CHUNK_SIZE = 20000   # number of time samples used in chunk for filtering and detection
    CHUNK_OVERLAP = 200 # number of samples that chunks overlap in time

The higher your sampling rate, the higher you should set these two (maintain the ratio of 10:1, it seems good). Set CHUNK_SIZE =  sampling rate.

The default parameters are as follows (see /spikedetekt/defaultparameters.py and change as desired):

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
    BUTTER_ORDER = 3 # Order of butterworth filter
    WRITE_FIL_FILE = True # write filtered output to .fil file

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

    # Experimental options
    DO_GLOBAL_CLUSTERING = False
    SORT_CLUS_BY_CHANNEL = False # Sort clusters by the channel where the peak occurs
    
    



3) Running
----------------------------

Finally to run the program type:

    python SpiKeDeteKt/scripts/detektspikes.py filename.params



4) Output
---------------

SpiKeDeteKt will output the following files, where n is your shank number:

1. .fet.n (feature file - required for all versions of KlustaKwik)

+ .mask.n (needed for using the new (masked) KlustaKwik)

+ .clu.n (a trivial clue file where everything is put into a single cluster, enabling the user to peruse the data using Klusters if desired)

+ .fmask.n (trial - float masks instead of binary, we are using this for testing masked KlustaKwik)

+ .spk.n (spike file)

+ .upsk.n (unfiltered spike waveform)

+ .res.n (list of spike times)

+ .xml (an xml file with all the parameters that can subsequently be used by neuroscope or klusters)

+ .fil (highpass filtered data)

+ .h5 (an [.h5](http://en.wikipedia.org/wiki/Hierarchical_Data_Format) file duplicating a lot of the above data, which may later be eliminated). It contains:
    *  raw, unfiltered, unaligned wave
    * filtered, aligned wave
    * channel mask
    * float channel mask
