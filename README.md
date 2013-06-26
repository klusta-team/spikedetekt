SpikeDetekt
-----------

This is a program for spike detection, that is optimized for high-channel count silicon probes.

This software was developed at the [Cortical Processing Laboratory](http://www.ucl.ac.uk/cortexlab) at UCL.

Please send feedback to Kenneth Harris (firstname at cortexlab.net), Shabnam Kadir (firstname at cortexlab.net)

Here is a quick Start Guide (will become more comprehensive with time):
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

0) Installation:
----------------

SpikeDetekt is written in Python and should work on any OS. To install it, make sure you first have Python on your computer.

We recommend you use Python 2.6 or 2.7 (don't use python 3.X!). A free academic distribution can be obtained from [Enthought Python](http://enthought.com/products/epd.php).

SpikeDetekt requires the following Python packages to be installed:

*NumPy
*SciPy
*PyTables
*Cjson
*h5py

Once you have set up Python on your system, download and unzip/tar either the .zip file or the tarball,
go to the `spikedetekt' folder and type (on the command line):

    python setup.py install

In the above,  'python' is the necessary command on your system for calling python. The file 'setup.py' is to be found in the unzipped folder spikedetekt. 

This will install SpikeDetekt. 

1) Usage:
----------

To perform spike detection, you need:

* a .dat file (which contains your raw unfiltered electrode data), 
* a .probe file, which contains information about the electrode,
* a .params file, which contains all other parameters.

The above may have any combination of names. The name of your .params file will be the name of the folder where all the output will be stored. The simplest case is when you have the same name for all three files:

    myexperiment.dat
    myexperiment.probe
    myexperiment.params
    
This will result in output files contained in a folder with the local name: myexperiment.     

2) Probefiles:
---------------
The probe file is a text file that contains information about the spatial arrangement of electrodes on the recording probe. SpikeDetekt needs this information because it detects spikes on high-count probes as spatiotemporally continuous threshold crossings. 

The probe file must contain an "adjancy graph", the specifies all pairs of channels that are nearest neighbors. (Sometimes you will want to also include second-nearest neighbors in the graph too). 

This information is presented in the following form:

    probes = {
        1: [
            (0, 1), (0, 2),
            (1, 2), (1, 3),
            ...
            ],
        2: [
           (13,15),(13,14),...
            ],
            
        shank number: [neighbouring channel pairs
            ],
        ...
        }

The probe file is actually formatted as a python command defining dictionary variable *probes*, with one key for each shank number, and values a list of neighboring channel pairs on that shank. If you don't know Python, don't worry, just think of it as a text file with the above format. 

As a more concrete example, for the following 32 channel zig-zag probe the adjacency graph is defined by the black lines:

<img src="docs/images/adjacency.png" height="750px" width="200px" />



If there are odd channels on one edge, even channels on the other, the .probe file corresponding to the above probe would like something this: 

    probes = {
        # Probe 1
	    1:[
		    (0, 1), (0, 2),
		    (1, 2), (1, 3),
	    	(2, 3), (2, 4),
	    	(3, 4), (3, 5),
    		(4, 5), (4, 6),
    		(5, 6), (5, 7),
    		(6, 7), (6, 8),
	    	(7, 8), (7, 9),
		    (8, 9), (8, 10),
    		(9, 10), (9, 11),
    		(10, 11), (10, 12),
    		(11, 12), (11, 13),
    		(12, 13), (12, 14),
    		(13, 14), (13, 15),
    		(14, 15), (14, 16),
    		(15, 16), (15, 17),
    		(16, 17), (16, 18),
    		(17, 18), (17, 19),
	    	(18, 19), (18, 20),
    		(19, 20), (19, 21),
	    	(20, 21), (20, 22),
	    	(21, 22), (21, 23),
	    	(22, 23), (22, 24),
	    	(23, 24), (23, 25),
	    	(24, 25), (24, 26),
	    	(25, 26), (25, 27),
	    	(26, 27), (26, 28),
	     	(27, 28), (27, 29),
	    	(28, 29), (28, 30),
	    	(29, 30), (29, 31),
	    	(30, 31),
		    ]
	    }

Further examples of probe files can be found in the distribution: 

* buzsaki32.probe
* linear16.probe
* multishankslinear32.probe (an 8 shank example)


3) Parameters to adjust
----------------------------
The parameter file (with a name something like myexperiment.params) contains further information about how to detect spikes. The following parameters, specifying the probe file name, the raw data files, number of recording channels and sample rate, are compulsory:
    
    RAW_DATA_FILES = ['file1.dat', 'file2.dat','file3.dat']
    SAMPLERATE = 20000 # in Hertz
    NCHANNELS = 32
    PROBE_FILE = 'probe_filename.probe'

Note that you can specify an ordered list of .dat files, which will be concatenated before spikes are detected.

There are also a lot of optional parameters. You should specify these in your parameter file if you want to override the default parameters, whose value appears in the file spikedetekt/spikedetekt/defaultparameters.py. Note that the parameter files are also python scripts, but again you don't need to worry about this, just think of them as text files.  These parameters, and an explanation of what they do, can be found in the file defaultparameters.py.


4) Running
----------------------------

To run the program type the following into the command line:

    python spikedetekt/scripts/detektspikes.py myexperiment.params
 
See below on how to configure your parameter file, myexperiment.params according to the specifics of your experimental setup.


5) Output
---------------

SpikeDetekt will output the following files for each shank, where n is the shank number:

+ .fet.n (feature file - required for all versions of KlustaKwik)

+ .fmask.n (float masks for using with the new masked KlustaKwik)

+ .mask.n (soon to be obsolete: binary masks for using with the new masked KlustaKwik)

+ .spk.n (spike waveforms)

+ .upsk.n (unfiltered spike waveforms)

+ .res.n (list of spike times)

+ .clu.n (a trivial clu file that puts all spikes in a single cluster. This is replaced by running KlustaKwik later on)

In addition, the following file will also be output:

+ .xml (an xml file with the parameters that are needed by the data visualization programs: Neuroscope and Klusters). We now recommend using KlustaViewa for manual clustering.

+ .fil (highpass filtered data)

+ .h5 files (an [.h5](http://en.wikipedia.org/wiki/Hierarchical_Data_Format) file duplicating a lot of the above data, which will later replace the above).
  .high.h5,
  .low.h5,
  .waves.h5,
  .main.h5,
  .raw.h5. (See spikdetekt/docs/fileformat.md for more details).

