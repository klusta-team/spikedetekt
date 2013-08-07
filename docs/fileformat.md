Proposition for a new spike sorting file format
===============================================

Currently, the spike sorting software products used in several labs (klusters,
klustakwik, neuroscope, etc.) use file formats that raise some issues:
heterogeneity of formats (binary, text) or structure (first line different
from the remaining lines in text files, etc.). Also, as the number of channels
in multielectrode probes grows, these files become quite large and do not fit
in memory. The user has to open them in chunks and must do some sort
of memory mapping manually. Finally, there's a need for a flexible and
carefully designed file format that may be shared between labs. The flexibility
is crucial since different labs use different hardware, different protocols,
etc.

Here we give some propositions for a new file format that aims at resolving
these issues. The main objectives of this format are:

  * flexibility and compatibility: each lab should be able to customize the
    file format while maintaining compatibility as far as that's possible,
  * to be based on a low-level file format that is widely used and allows for
    easy and efficient memory mapping,
  * simplicity and efficiency.

Notes:
  * add date and time of the experiment
  * hierarchy of metadata (global, local) with priority given to the more local
    when conflicts exist
  * storing metadata as JSON strings
  

  
  

HDF5
====

The new file format is based on 
[HDF5 (Hierarchical Data Format)](http://en.wikipedia.org/wiki/Hierarchical_Data_Format), 
an open,
BSD-licensed file format originally
developed in the National Center for Supercomputing 
Applications (NCSA) at the University of Illinois at Urbana-Champaign in
the late 80s. Today, it is widely used, particularly in institutions 
dealing with large amounts of data (NASA, NOAA, etc.). Also, the netCDF4 file 
format, particularly used in climatology, meteorology and oceanography, is
based on HDF5.

The HDF5 format defines an API
for opening and saving HDF5 files, which has been implemented in many
languages (C, Fortran, Python, Matlab, R...). 
Actually, the Matlab format for `.mat` arrays seems to be HDF5 itself
starting from version 7.3.
In Python, the main external
libraries which read HDF5 files are PyTables and h5py. The latter is 
lighter and is a thin wrapper around the C library. In addition, it is
fully compatible with Numpy and is simpler to use than PyTables.

Technically, an HDF5 file is a binary file containing a structured set of
data: text, numbers, multidimensional arrays, etc. The data are
contained in a hierarchical structure very similar to the POSIX
file system structure: groups are folders, datasets are files, with the syntax
`/group/subgroup/dataset`. A dataset is a multidimensional array containing
any data type. Both groups and datasets can contain metadata (numbers,
strings, etc.).

Access to an HDF5 file is extremely fast since the format integrates
efficient memory mapping features. That is, it is possible to request part
of the data contained in the file without loading the full file in memory.
Any group or dataset can be requested efficiently, as well as
contiguous or strided portions of a dataset.


File format features
====================

With the new file format, an extracellular recording session will be
contained in several HDF5 files, along with a structured text file with
non-critical visualization-related metadata (XML, YAML or any other simple text
format) and a probe file (a text file containing Python code, with a .py
extension) describing the geometry of the probe. The spike sorting information
(clusters, spike times, waveforms, etc.) will be contained in these HDF5 files.

A light Python library will also be developed so as to read and write
information easily from and to this new file format.

Main file and external links
----------------------------

The HDF5 files contain a master file, and external files. The master file
gives access to the information contained in all files: it virtually
contains the external files. That is, the main file contains
*external links* to the external files so that, from the library point of view,
it looks like the data contained in these external files are actually 
contained in the master file.

The reason for this is the following. For conveniency, everything should be 
accessible from a single HDF5 file. However, with more and more channels
being available on multielectrode probes, the size of this file may be very 
large (tens or hundreds of GB). Yet, it may be useful to have files 
of reasonable size with
just partial information about the experiment (e.g. only the spike information,
or only the raw data and not the filtered data, etc.). 
A dataset appears within the main HDF5 file 
but actually refers to another HDF5 file, in a totally transparent manner.
Medium-sized data sets could be stored in the main HDF5 file, whereas
large data sets could be stored in the external files. This way, most
information could still be available in the main file, whereas additional
information would only be available if the external files are present in the
folder.


File size estimation
--------------------

Here is the list of information that needs to be stored in the files,
with an estimation of the size for a probe with 100 channels
(20 kHz, 16 bits) and a one hour long experiment (1000 different neurons
with a mean firing rate of 1 Hz, 32 samples per waveform).

  * **raw data**: 13GB.
  
  * **high-pass filtered data**: 13GB.
  
  * **low-pass filtered data**: 860MB (16x downsampled).
    
  * **filtered waveforms of spikes across all channels**: 21GB (there is redundancy, 
    i.e. spike overlap across channels, which explains why it is actually
    *bigger* than the raw data file! do we really need it, since the
    data is already there? we can retrieve the waveforms just with view
    on the raw data, no need for copying all samples. There's the issue
    of spike aligning/interpolation though.).
    
  * **unfiltered waveforms**: 21GB (like waveforms but directly from raw data,
    without high-pass filter).
    
  * **spike features across all channels**: 2GB (3 PCA features, coefficients
    stored as 16 bits numbers).
  
  * **masks of channels for all spikes**: 350MB (masks coded on 8 bits).
  
  * **spike times**: 13MB (32 bits = max value ~4 billion samples = 60 hours
    at 20 kHz).
  
  * **clusters for all spikes**: 15MB (int32 = max value ~4 billion)
  
  * **cluster info** (colors, groups, etc.): negligible size (10KB).
  
  * **events** (optional): one or several lists of temporal events with some description
    (small size).
    
  * **trajectory** (optional): the trajectory of the animal in space, as a list
    of coordinates (small size).
  
  * **metadata** about the experiment (probe information like number of channels
    or geometry, sampling frequency, etc.): negligible size.

Total size: about 70GB.


Multiple sessions
-----------------

When several recording sessions are made with the same neurons, several binary
files (in a format specific to the recording hardware, e.g. ns5) are created.
Along with the probe file, they will be given as input to a Python IO module
(using Neo) which concatenates them and creates several HDF5 files. 
The length of each subfile is saved so that we can easily know to which file
belongs any spike.


Multiple shanks
---------------

Several independent shanks can be stored in a single HDF5 file. The
shank information is stored in the probe file.


Probe file
----------

The probe structure is described in 
[JSON](http://en.wikipedia.org/wiki/JSON), 
a text-based open standard designed
for human-readable data interchange and derived from JavaScript.
It contains the following information:

  * total number of channels
  * total number of shanks
  * [optional] a list of channel names; otherwise, the channels are numbered sequentially according to data row
  * [optional] a list of dead channels (which can be added to, or removed, in a single place)
  * for each shank:
      * the shank index
      * the list of channel (absolute) indices in this shank (all the channels, including dead channels)
      * the adjacency graph of the channels in that shank, as a list of pairs
        of indices (this graph contains all channels, even the unkept ones,
        so that the actual adjacency graph is a subgraph defined by the list
        of kept channels)
      * [optional] the shank geometry, as a list of x,y pairs for each channel
  
Example:

    {
        "shanks": 
            [
                {
                    "index": 0,
                    "channels": [0, 1, 2, 3],
                    "graph": [[0, 1], [2, 3], ...],
                    "geometry": {"0": [0.1, 0.2], "1": [0.3, 0.4], ...}
                },
                {
                    "index": 1,
                    "channels": [4, 5, 6, 7],
                    "graph": [[4, 5], [6, 7], ...],
                    "geometry": {"4": [0.1, 0.2], "5": [0.3, 0.4], ...}
                }
            ]
    }

Note: in Python, this string `s` can be loaded as a Python dictionary with:

    import json  # a native Python module
    dic = json.loads(s)  # JSON deserialization
    print(dic["shanks"][0]["geometry"])  # [[0.123, 0.456]]
    
    
HDF5 files
----------

In the following file names, `experiment` refers to the experiment name.
`/xxx` is a dataset in the root group, `/xxx (group)` is a group in root.
Each shank corresponds to a group in the file, i.e. `/shank/0` is the group for
shank 0.

  * `experiment.main.h5`: main file with:
      * /shank/0/spiketimes
      * /shank/0/masks_binary (one value per feature)
      * /shank/0/masks_float (one value per feature)
      * /shank/0/features
      * /shank/0/clusters
      * /shank/0/clusters_original
      * /shank/0/actionlog (log with ordered list of all merge/split actions)
      * /shank/0/cluster_info (group)
          * manual group for each cluster (good/bad/ugly, used for the manual
            spike sorting stage)
          * neuron type (pyramidal, interneuron...) for each cluster
          * mean waveform
          * quality measures, etc.
      * /events:
          * eventname1: array with time and name of each event
          * attribute with dt for the time stamp (1 ms for backward compatibility or the sampling rate)
      * /trajectory: array with the timestamp and position of the animal (and maybe 
        metadata with the dt)
      * /metadata:
          * /metadata/parameters (all information in parameters.py)
              * sampling rate
              * voltage gain (vector with one value per channel, possibly a 
                single value in the paramters file, then converted into a vector)
          * /metadata/probe (probe structure as a JSON string)
          * /metadata/datfiles_start_points (the length of each session subfile)
      * external links (options in parameters.py, used by spikedetekt, to save 
        or not these files, disabled by default for raw, high, low for now):
          * /rawdata ==> experiment.raw.h5/rawdata
          * /highpass ==> experiment.high.h5/highpass
          * /lowpass ==> experiment.low.h5/lowpass
          * /shank/0/waveforms ==> experiment.wave.h5/shank/0/waveforms
          * /shank/0/uwaveforms ==> experiment.uwave.h5/shank/0/uwaveforms
          * the external files contain a copy of the master file metadata

  * `experiment.raw.h5`: raw data and part of the metadata contained in the
    main file:
      * sampling rate
      * voltage gain
  * `experiment.high.h5`: high-pass filtered data and part of the metadata.
  * `experiment.low.h5`: low-pass filtered data and part of the metadata.
  * `experiment.wave.h5`: waveforms and part of the metadata.

Every dataset if per shank, except the raw and filtered data which contain
all channels, and the metadata.

The main file size is about 2-3GB (in the conditions specified above).

SpikeDetekt
-----------

A single input to the command line interface: `parameters.py`. This file 
contains:

  * the `probe.json` filename, which contains:
      * probe structure
      * nchannels
  * the `*.dat` filename(s) as a string or a list of strings (for concatenation
    of the .dat files)
  * some metadata:
      * sampling rate
      * voltage gain
  
Output of spikedetekt: the HDF5 file with possibly the external files.
There's no `.clu` file as output, as it will be automatically replaced by
a trivial cluster array by klustakwik and klustaviewa.
  


New notes (07/08/2013)
======================

Old files:

  * dat: raw data, the general input file
  * probe: Python file with probe information for SpikeDetekt
  * xml: metadata
  * clu/fet/spk/...: spike and cluster information, generated by Klusters and co
  * aclu/acluinfo/groupinfo: generated by KlustaViewa 0.1.0

New files:

  * klx (HDF5): main HDF5 file
  * klxw (HDF5): waveforms
  * [raw/low/high].klxd (HDF5): raw or filtered multichannel data + some metadata (sampling frequency, voltage gain, ...)
  * klxa (JSON text file): most metadata, including probe file, dead channels, channel colors, channel groups, channel names, group names, is visible. Example:

        {
            "channel_height": 0.25,
            "lastviewed_x": 0.24,
            
            "channels":
                [
                    {
                         "0": {"group": "0", "name": "ch0"},
                         "1": {"group": "0", "name": "ch1", "channel_height": 0.1},
                         "2": {"group": "0", "name": "LFP", "color": 12},
                         "3": {"group": "0", "ignore": true},
                         ...
                         "7": {"group": 1, "name": "ch7", "ypos", 0.25},
                         "8": {"group": 2, "name": "Sync Pulse", "voltage_gain_multiplier": 1.2, "visible": false, "ignore": true}
                    },
                ],
                         
            "groups":
                [
                    {
                         "0": {"name": "Hippocampus", "color": 14},
                         "1": {"name": "Prefrontal Cortex", "visible": false, "color": 13},
                         "2": {"name": "Auxiliary Data", "color": 2}
                    },
                ],
        }
        
  * klxp: new probe file
        
        {
            "shanks": 
                [
                    {
                        "index": 0,
                        "channels": [0, 1, 2, 3],
                        "graph": [[0, 1], [2, 3], ...],
                        "geometry": {"0": [0.1, 0.2], "1": [0.3, 0.4], ...}
                    },
                    {
                        "index": 1,
                        "channels": [4, 5, 6, 7],
                        "graph": [[4, 5], [6, 7], ...],
                        "geometry": {"4": [0.1, 0.2], "5": [0.3, 0.4], ...}
                    }
                ]
        }
      

Conversion from old to new files:

  * standalone tool klustakonvert (in repo klustalib): from (DAT [+XML] + probe) to (KLXD + KLXA + KLXP)

        # Convert a dat file into several KLXD and KLXA and KLXP (probe JSON).
        # The probe file (Python) is mandatory, unless --noprobe is specified.
        # It is not allowed to specify neither --probe nor --noprobe: the program will issue an error in this case, telling the user to specify one of the two.
        klustakonvert data mydata.dat [-o /home/me/newdata/mydata.raw.klxd] [--probe myprobe.probe] ([--x myxml.xml])|([--nchannels 32] [--freq 20000] [--nbits 16])  [--ignore-channels 1,2,3] --noprobe
        
        # If klxd does not exist, do the conversion as above.
        # Else, will not overwrite it, but add probe file to already klxa.
        klustakonvert probe myprobe.probe /home/me/newdata/mydata.klxp
        

  * NeuraViewa: 
  
      * can open .dat
          * if XML: freq, nchannels, nbits, ...
          * else: open GUI requesting freq, nchannels, nbits, ...
          * save: 
              * modal window: warning that conversion is mandatory + button to choose probe file
              * if given, new window Save As with KLXD + call klustakonvert --probe when press OK
              * else, new window Save As with KLXD + call klustakonvert --noprobe when press OK
          
      * can open .klxd
          * if .klxa exists, load it
          * else, when saving, create one with default values
        
  * SpikeDetekt (with Parameters file):
  
      * cannot open .dat. If .dat provided, message with:

            Conversion is going to happen (through call to code in klustalib), and old files will be moved to "oldklusters". If you do not want your files to be moved, add --nomove to the spikedetekt command: your files will still be converted in the new format. Do you want to continue? [Y/n]. If the user presses no, abort. Once it is done, call the same spikedetekt command with the new files in arguments.

      * can open .klxd + .klxa + .klxp, and generate .klx + .klxw + .[high/low].klxd with filters
      
  * KlustaViewa:
  
      * if provided with old Klusters files, automatically generate the .KLX + .KLXW + .KLXA
          * TODO: move colors and cluster groups into .KLXA
          


Compatibility with NetCDF4
==========================

If possible, we may want to avoid using the following uncommon features of
HDF5, so that we have 
[100% compatibility with the NetCDF4 format](http://www.unidata.ucar.edu/software/netcdf/#netcdf_faq).

  * Multidimensional data that doesn't use shared dimensions implemented using 
    HDF5 "dimension scales". (This restriction was eliminated in netCDF 4.1.1, 
    permitting access to HDF5 datasets that don't use dimension scales.)

  * Non-hierarchical organizations of Groups, in which a Group may have multiple 
    parents or may be both an ancestor and a descendant of another Group, 
    creating cycles in the subgroup graph. In the netCDF-4 data model, Groups 
    form a tree with no cycles, so each Group (except the top-level unnamed 
    Group) has a unique parent.

  * HDF5 "references" which are like pointers to objects and data regions within 
    a file. The netCDF-4 data model does not support references.

  * Additional primitive types not included in the netCDF-4 data model, 
    including H5T_TIME, H5T_BITFIELD, and user-defined atomic types.

  * Multiple names for data objects such as variables and groups. The netCDF-4 
    data model requires that each variable and group have a single distinguished 
    name.

  * Attributes attached to user-defined types.

  * Stored property lists

  * Object names that begin or end with a space

