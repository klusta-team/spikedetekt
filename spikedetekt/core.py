from __future__ import with_statement, division
import itertools as it, numpy as np, scipy.signal as signal
from scipy.stats import rv_discrete
from scipy.stats.mstats import mquantiles
from xml.etree.ElementTree import ElementTree
import re, tables, json, os

import probes
from files import write_fet
from graphs import contig_segs, complete_if_none, add_penumbra
from utils import indir, basename_noext, get_padded, switch_ext
from floodfill import connected_components
from features import compute_pcs, reget_features, project_features
from files import num_samples, spike_dtype, klusters_files,\
                  get_chunk_for_thresholding, chunks
from filtering import apply_filtering, get_filter_params
from progressbar import ProgressReporter
from alignment import extract_wave
from os.path import join, abspath, dirname
from parameters import Parameters, GlobalVariables
from time import sleep
from subsets import cluster_withsubsets
from masking import get_float_mask
from log import log_message

def set_globals_samples(sample_rate):
    """
    parameters are set in terms of time (seconds).
    this sets corresponding parameters in terms of sample rate. should be run
    before any processing
    """
    Parameters['SAMPLE_RATE'] = sample_rate
    exec 'F_HIGH = .95*SAMPLE_RATE/2' in Parameters
    exec 'S_BEFORE = int(T_BEFORE*SAMPLE_RATE)' in Parameters
    exec 'S_AFTER = int(T_AFTER*SAMPLE_RATE)' in Parameters
    exec 'S_TOTAL = S_BEFORE + S_AFTER' in Parameters
    exec 'S_JOIN_CC = T_JOIN_CC*SAMPLE_RATE' in Parameters
    
####################################
######## High-level scripts ########
####################################

def spike_detection_job(DatFileNames, ProbeFileName, output_dir, output_name):
    """
    Top level function that starts a data processing job.
    """
    for DatFileName in DatFileNames:
        if not os.path.exists(DatFileName):
            raise Exception("Dat file %s does not exist"%DatFileName)
    DatFileNames = [os.path.abspath(DatFileName) for DatFileName in DatFileNames]
    
    probe = probes.Probe(ProbeFileName)
    
    n_ch_dat = Parameters['NCHANNELS']
    sample_rate = Parameters['SAMPLERATE']
    set_globals_samples(sample_rate)
    Parameters['CHUNK_OVERLAP'] = int(sample_rate*Parameters['CHUNK_OVERLAP_SECONDS'])
    
    Parameters['N_CH'] = probe.num_channels
    
    max_spikes = Parameters['MAX_SPIKES']
    
    basename = basenamefolder = output_name
        
    OutDir = join(output_dir, basenamefolder)
    with indir(OutDir):    
        # Create a log file
        GlobalVariables['log_fd'] = open(basename+'.log', 'w')    
        
        Channels_dat = np.arange(probe.num_channels)
        spike_detection_from_raw_data(basename, DatFileNames, n_ch_dat,
                                      Channels_dat, probe.channel_graph,
                                      probe, max_spikes)
        
        numwarn = GlobalVariables['warnings']
        if numwarn:
            log_message('WARNINGS ENCOUNTERED: '+str(numwarn)+', check log file.')
            

def spike_detection_from_raw_data(basename, DatFileNames, n_ch_dat, Channels_dat,
                                  ChannelGraph, probe, max_spikes):
    """
    Filter, detect, extract from raw data.
    """
    ### Detect spikes. For each detected spike, send it to spike writer, which
    ### writes it to a spk file. List of times is small (memorywise) so we just
    ### store the list and write it later.

    hdf5file = tables.openFile(basename+".h5", "w")
    spike_table = hdf5file.createTable("/", "SpikeTable_temp",
                                       spike_dtype())
    np.savetxt("dat_channels.txt", Channels_dat, fmt="%i")
    hdf5file.createArray("/", "DatChannels", Channels_dat)
    
    for (USpk, Spk, PeakSample,
         ChannelMask, FloatChannelMask) in extract_spikes(basename,
                                                          DatFileNames,
                                                          n_ch_dat,
                                                          Channels_dat,
                                                          ChannelGraph,
                                                          max_spikes):
        spike_table.row["unfiltered_wave"] = USpk
        spike_table.row["wave"] = Spk
        spike_table.row["time"] = PeakSample
        spike_table.row["channel_mask"] = ChannelMask
        spike_table.row["float_channel_mask"] = FloatChannelMask
        spike_table.row.append()
        
    spike_table.flush()    

    ### Feature extraction on spikes    
    PC_3s = reget_features(spike_table.cols.wave[:10000])
    for row in spike_table: 
        row["fet"] = project_features(PC_3s, row["wave"])
        fet_mask = np.hstack((np.repeat(row["channel_mask"],
                                        Parameters['FPC']),
                              [0])) # 0 is added for time feature
        row["fet_mask"] = fet_mask
        float_fet_mask = np.hstack((np.repeat(row["float_channel_mask"],
                                              Parameters['FPC']),
                                    [0])) # 0 is added for time feature
        row["float_fet_mask"] = float_fet_mask
        row.update()
            
    #m write out all the output files
    spike_table.flush()
    klusters_files(spike_table, basename, probe)

    ### And use batch clustering procedure (EM) to get clusters.
    #m CluArr is a 1D array containing the cluster to which every spike is assigned
    if Parameters['DO_GLOBAL_CLUSTERING']:
        CluArr = cluster_withsubsets(spike_table,
                                     Parameters['SORT_CLUS_BY_CHANNEL'])

    hdf5file.close()
                
###########################################################
############# Spike extraction helper functions ###########    
###########################################################

#m  
#m this function returns for each detected spike a triplet Spk,PeakSample,ST
#m Spk is an array of shape no. of samples (to record for each spike) x no. of channels, e.g., 28X16
#m PeakSample is the position of the peak (e.g. 435688)
#m ST (to the best of my understanding) is a bool array (no. of channels long) which shows on which 
#m                                      channels the threshold was crossed (?)
def extract_spikes(basename, DatFileNames, n_ch_dat, ChannelsToUse, ChannelGraph,
                   max_spikes=None):
    # some global variables we use
    CHUNK_SIZE = Parameters['CHUNK_SIZE']
    CHUNKS_FOR_THRESH = Parameters['CHUNKS_FOR_THRESH']
    DTYPE = Parameters['DTYPE']
    CHUNK_OVERLAP = Parameters['CHUNK_OVERLAP']
    N_CH = Parameters['N_CH']
    S_JOIN_CC = Parameters['S_JOIN_CC']
    S_BEFORE = Parameters['S_BEFORE']
    S_AFTER = Parameters['S_AFTER']
    THRESH_SD = Parameters['THRESH_SD']
    
    # filter coefficents for the high pass filtering
    filter_params = get_filter_params()

    progress_bar = ProgressReporter()
    
    #m A code that writes out a high-pass filtered version of the raw data (.fil file)
    if Parameters['WRITE_FIL_FILE']:
        fil_fd = open(basename+'.fil', 'wb')

    # Just use first dat file for getting the thresholding data
    with open(DatFileNames[0], 'rb') as fd:
        # Use 5 chunks to figure out threshold
        DatChunk = get_chunk_for_thresholding(fd, n_ch_dat, ChannelsToUse,
                                              num_samples(DatFileNames[0],
                                                          n_ch_dat))
        FilteredChunk = apply_filtering(filter_params, DatChunk)
        # .6745 converts median to standard deviation
        if Parameters['USE_SINGLE_THRESHOLD']:
            ThresholdSDFactor = np.median(np.abs(FilteredChunk))/.6745
        else:
            ThresholdSDFactor = np.median(np.abs(FilteredChunk), axis=0)/.6745
        Threshold = ThresholdSDFactor*THRESH_SD
    
    n_samples = num_samples(DatFileNames, n_ch_dat)
    
    spike_count = 0
    for (DatChunk, s_start, s_end,
         keep_start, keep_end) in chunks(DatFileNames, n_ch_dat, ChannelsToUse):
        ############## FILTERING ########################################
        FilteredChunk = apply_filtering(filter_params, DatChunk)
        
        # write filtered output to file
        if Parameters['WRITE_FIL_FILE']:
            if s_end>keep_end: #m writing out the high-pass filtered data
                FilteredChunkInt = FilteredChunk[keep_start-s_start:keep_end-s_end]
                FilteredChunkInt = np.int16(FilteredChunkInt)
            else: #m we're in the end
                FilteredChunkInt = np.int16(FilteredChunk[keep_start-s_start:])
            fil_fd.write(FilteredChunkInt) #m

        ############## THRESHOLDING #####################################
        if Parameters['DETECT_POSITIVE']:
            BinaryChunk = np.abs(FilteredChunk)>Threshold
        else:
            BinaryChunk = (FilteredChunk<-Threshold)
        BinaryChunk = BinaryChunk.astype(np.int8)
        ############### FLOOD FILL  ######################################
        ChannelGraphToUse = complete_if_none(ChannelGraph, N_CH)
        IndListsChunk = connected_components(BinaryChunk,
                            ChannelGraphToUse, S_JOIN_CC)
        ############## ALIGN AND INTERPOLATE WAVES #######################
        nextbits = []
        for IndList in IndListsChunk:
            wave, s_peak, cm = extract_wave(IndList, FilteredChunk,
                                            S_BEFORE, S_AFTER, N_CH,
                                            s_start)
            s_offset = s_start+s_peak
            if keep_start<=s_offset<keep_end:
                spike_count += 1
                nextbits.append((wave, s_offset, cm))
        # and return them in time sorted order
        nextbits.sort(key=lambda (wave, s, cm): s)
        for wave, s, cm in nextbits:
            uwave = get_padded(DatChunk, int(s)-S_BEFORE-s_start,
                               int(s)+S_AFTER-s_start).astype(np.int32)
            cm = add_penumbra(cm, ChannelGraphToUse,
                              Parameters['PENUMBRA_SIZE'])
            fcm = get_float_mask(wave, cm, ChannelGraphToUse,
                                 ThresholdSDFactor)
            yield uwave, wave, s, cm, fcm
        progress_bar.update(float(s_end)/n_samples,
            '%d/%d samples, %d spikes found'%(s_end, n_samples, spike_count))
        if max_spikes is not None and spike_count>=max_spikes:
            break
    
    progress_bar.finish()
    
    if Parameters['WRITE_FIL_FILE']:
        fil_fd.close()  
