from __future__ import with_statement, division
import itertools as it, numpy as np, scipy.signal as signal
from scipy.stats import rv_discrete
from scipy.stats.mstats import mquantiles
from xml.etree.ElementTree import ElementTree
import re, tables, json, os, h5py
from itertools import izip

import probes
from files import write_fet
from graphs import contig_segs, complete_if_none, add_penumbra
from utils import indir, basename_noext, get_padded, switch_ext
from floodfill import connected_components, connected_components_twothresholds
from features import compute_pcs, reget_features, project_features
from files import (num_samples, klusters_files,
                   get_chunk_for_thresholding, chunks, shank_description,
                   waveform_description, FilWriter)
from filtering import apply_filtering, get_filter_params
from progressbar import ProgressReporter
from alignment import extract_wave, extract_wave_hilbert_old, extract_wave_hilbert_new,extract_wave_twothresholds, InterpolationError
from os.path import join, abspath, dirname
from parameters import Parameters, GlobalVariables
from time import sleep
from subsets import cluster_withsubsets
from masking import get_float_mask
from log import log_message, log_warning
from IPython import embed
import debug
from debug import plot_diagnostics, plot_diagnostics_twothresholds# for debugging with Parameters['DEBUG'] 


def set_globals_samples(sample_rate,high_frequency_factor):
    """
    parameters are set in terms of time (seconds).
    this sets corresponding parameters in terms of sample rate. should be run
    before any processing
    """
    Parameters['SAMPLE_RATE'] = sample_rate
    Parameters['HFF']=high_frequency_factor
    exec 'F_HIGH = HFF*SAMPLE_RATE/2' in Parameters
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
    high_frequency_factor = Parameters['F_HIGH_FACTOR']
    set_globals_samples(sample_rate,high_frequency_factor)
    Parameters['CHUNK_OVERLAP'] = int(sample_rate*Parameters['CHUNK_OVERLAP_SECONDS'])
    
    Parameters['N_CH'] = probe.num_channels
    
    max_spikes = Parameters['MAX_SPIKES']
    
    basename = basenamefolder = output_name
        
   # OutDir = join(output_dir, basenamefolder)
    OutDir = output_dir
    with indir(OutDir):    
        # Create a log file
        GlobalVariables['log_fd'] = open(basename+'.log', 'w') 
         
        if Parameters['DEBUG']:
            GlobalVariables['debug_fd'] = open(basename+'.debug', 'w')    
        
        Channels_dat = np.arange(probe.num_channels)
        # Print Parameters dictionary to .log file
        log_message("\n".join(["{0:s} = {1:s}".format(key, str(value)) for key, value in sorted(Parameters.iteritems()) if not key.startswith('_')]))
        spike_detection_from_raw_data(basename, DatFileNames, n_ch_dat,
                                      Channels_dat, probe.channel_graph,
                                      probe, max_spikes)
        
        numwarn = GlobalVariables['warnings']
        if numwarn:
            log_message('WARNINGS ENCOUNTERED: '+str(numwarn)+', check log file.')
    # Print Parameters dictionary to .log file
    #log_message("\n".join(["{0:s} = {1:s}".format(key, str(value)) for key, value in Parameters.iteritems()]))
    
    	# Close the log file at the end.
    	if 'log_fd' in GlobalVariables:
            GlobalVariables['log_fd'].close()
            

def spike_detection_from_raw_data(basename, DatFileNames, n_ch_dat, Channels_dat,
                                  ChannelGraph, probe, max_spikes):
    """
    Filter, detect, extract from raw data.
    """
    ### Detect spikes. For each detected spike, send it to spike writer, which
    ### writes it to a spk file. List of times is small (memorywise) so we just
    ### store the list and write it later.

    np.savetxt("dat_channels.txt", Channels_dat, fmt="%i")
    
    # Create HDF5 files
    h5s = {}
    h5s_filenames = {}
    for n in ['main', 'waves']:
        filename = basename+'.'+n+'.h5'
        h5s[n] = tables.openFile(filename, 'w')
        h5s_filenames[n] = filename
    for n in ['raw', 'high', 'low']:
        if Parameters['RECORD_'+n.upper()]:
            filename = basename+'.'+n+'.h5'
            h5s[n] = tables.openFile(filename, 'w')
            h5s_filenames[n] = filename
    main_h5 = h5s['main']
    # Shanks groups
    shanks_group = {}
    shank_group = {}
    shank_table = {}
    for k in ['main', 'waves']:
        h5 = h5s[k]
        shanks_group[k] = h5.createGroup('/', 'shanks')
        for i in probe.shanks_set:
            shank_group[k, i] = h5.createGroup(shanks_group[k], 'shank_'+str(i))
    # waveform data for wave file
    for i in probe.shanks_set:
        shank_table['waveforms', i] = h5s['waves'].createTable(
            shank_group['waves', i], 'waveforms',
            waveform_description(len(probe.channel_set[i])))
    # spikedetekt data for main file, and links to waveforms
    for i in probe.shanks_set:
        shank_table['spikedetekt', i] = main_h5.createTable(shank_group['main', i],
            'spikedetekt', shank_description(len(probe.channel_set[i])))
        main_h5.createExternalLink(shank_group['main', i], 'waveforms', 
                                   shank_table['waveforms', i])
    # Metadata
    n_samples = np.array([num_samples(DatFileName, n_ch_dat) for DatFileName in DatFileNames])
    for k, h5 in h5s.items():
        metadata_group = h5.createGroup('/', 'metadata')
        parameters_group = h5.createGroup(metadata_group, 'parameters')
        for k, v in Parameters.items():
            if not k.startswith('_'):
                if isinstance(v, bool):
                    r = int(v)
                elif isinstance(v, (int, float)):
                    r = v
                else:
                    r = repr(v)
                h5.setNodeAttr(parameters_group, k, r)
        h5.setNodeAttr(metadata_group, 'probe', json.dumps(probe.probes))
        h5.createArray(metadata_group, 'datfiles_offsets_samples',
                       np.hstack((0, np.cumsum(n_samples)))[:-1])
    
    ########## MAIN TIME CONSUMING LOOP OF PROGRAM ########################
    for (USpk, Spk, PeakSample,
         ChannelMask, FloatChannelMask) in extract_spikes(h5s, basename,
                                                          DatFileNames,
                                                          n_ch_dat,
                                                          Channels_dat,
                                                          ChannelGraph,
                                                          max_spikes,
                                                          ):
        # what shank are we in?
        nzc, = ChannelMask.nonzero()
        internzc = list(set(nzc).intersection(probe.channel_to_shank.keys()))
        if internzc:
            shank = probe.channel_to_shank[internzc[0]]
        else:
            continue
        # write only the channels of this shank
        channel_list = np.array(sorted(list(probe.channel_set[shank])))
        t = shank_table['spikedetekt', shank]
        t.row['time'] = PeakSample
        t.row['mask_binary'] = ChannelMask[channel_list]
        t.row['mask_float'] = FloatChannelMask[channel_list]
        t.row.append()
        # and the waveforms
        t = shank_table['waveforms', shank]
        t.row['wave'] = Spk[:, channel_list]
        t.row['unfiltered_wave'] = USpk[:, channel_list]
        t.row.append()
        
    for h5 in h5s.values():
        h5.flush()

    # Feature extraction
    for shank in probe.shanks_set:
        X = shank_table['waveforms', shank].cols.wave[:Parameters['PCA_MAXWAVES']]
        if len(X) == 0:
            continue
        PC_3s = reget_features(X)
        for sd_row, w_row in izip(shank_table['spikedetekt', shank],
                                  shank_table['waveforms', shank]):
            f = project_features(PC_3s, w_row['wave'])
            
            ### NEW
            # add PCA components
            sd_row['PC_3s'] = PC_3s.flatten()
            
            sd_row['features'] = np.hstack((f.flatten(), sd_row['time']))
            sd_row.update()
            
    main_h5.flush()
            
    klusters_files(h5s, shank_table, basename, probe)

#    for h5 in h5s.values():
#        h5.close()

    for key, h5 in h5s.iteritems():
        h5.close()
        if not Parameters['KEEP_OLD_HDF5_FILES']:
            # NEW: erase the HDF5 files at the end, because we're using a direct 
            # conversion tool in KlustaViewa for now.
            os.remove(h5s_filenames[key])
                
###########################################################
############# Spike extraction helper functions ###########    
###########################################################

#m  
#m this function returns for each detected spike a triplet Spk,PeakSample,ST
#m Spk is an array of shape no. of samples (to record for each spike) x no. of channels, e.g., 28X16
#m PeakSample is the position of the peak (e.g. 435688)
#m ST (to the best of my understanding) is a bool array (no. of channels long) which shows on which 
#m                                      channels the threshold was crossed (?)
def extract_spikes(h5s, basename, DatFileNames, n_ch_dat,
                   ChannelsToUse, ChannelGraph,
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
    THRESH_SD_LOWER = Parameters['THRESH_SD_LOWER']

    # filter coefficents for the high pass filtering
    filter_params = get_filter_params()

    progress_bar = ProgressReporter()
    
    #m A code that writes out a high-pass filtered version of the raw data (.fil file)
    fil_writer = FilWriter(DatFileNames, n_ch_dat)

    # Just use first dat file for getting the thresholding data
    with open(DatFileNames[0], 'rb') as fd:
        # Use 5 chunks to figure out threshold
        DatChunk = get_chunk_for_thresholding(fd, n_ch_dat, ChannelsToUse,
                                              num_samples(DatFileNames[0],
                                                          n_ch_dat))
        FilteredChunk = apply_filtering(filter_params, DatChunk)
        # get the STD of the beginning of the filtered data
        if Parameters['USE_HILBERT']:
            first_chunks_std = np.std(FilteredChunk)
        else:
            if Parameters['USE_SINGLE_THRESHOLD']:
                ThresholdSDFactor = np.median(np.abs(FilteredChunk))/.6745
            else:
                ThresholdSDFactor = np.median(np.abs(FilteredChunk), axis=0)/.6745
            Threshold = ThresholdSDFactor*THRESH_SD
            print 'Threshold = ', Threshold, '\n' 
            Parameters['THRESHOLD'] = Threshold #Record the absolute Threshold used
        
    # set the high and low thresholds 
    if Parameters['USE_HILBERT']:
        ThresholdStrong = Parameters['THRESH_STRONG']
        ThresholdWeak = Parameters['THRESH_WEAK']
    elif Parameters['USE_COMPONENT_ALIGNFLOATMASK']:#to be used with a single threshold only
        ThresholdStrong = Threshold
        ThresholdWeak = ThresholdSDFactor*THRESH_SD_LOWER
    
        
    n_samples = num_samples(DatFileNames, n_ch_dat)
    spike_count = 0
    for (DatChunk, s_start, s_end,
         keep_start, keep_end) in chunks(DatFileNames, n_ch_dat, ChannelsToUse):
        ############## FILTERING ########################################
        FilteredChunk = apply_filtering(filter_params, DatChunk)
        
        # write filtered output to file
        if Parameters['WRITE_FIL_FILE']:
            fil_writer.write(FilteredChunk, s_start, s_end, keep_start, keep_end)

        ############## THRESHOLDING #####################################
        
        
        # NEW: HILBERT TRANSFORM
        if Parameters['USE_HILBERT']:
            FilteredChunkHilbert = np.abs(signal.hilbert(FilteredChunk, axis=0) / first_chunks_std) ** 2
            BinaryChunkWeak = FilteredChunkHilbert > ThresholdWeak
            BinaryChunkStrong = FilteredChunkHilbert > ThresholdStrong
            BinaryChunkWeak = BinaryChunkWeak.astype(np.int8)
            BinaryChunkStrong = BinaryChunkStrong.astype(np.int8)
        #elif Parameters['USE_COMPONENT_ALIGNFLOATMASK']:
        else: # Usual method
            FilteredChunk = apply_filtering(filter_params, DatChunk)
            if Parameters['USE_COMPONENT_ALIGNFLOATMASK']:
                if Parameters['DETECT_POSITIVE']:
                    BinaryChunkWeak = FilteredChunk > ThresholdWeak
                    BinaryChunkStrong = FilteredChunk > ThresholdStrong
                else:
                    BinaryChunkWeak = FilteredChunk < -ThresholdWeak
                    BinaryChunkStrong = FilteredChunk < -ThresholdStrong
                BinaryChunkWeak = BinaryChunkWeak.astype(np.int8)
                BinaryChunkStrong = BinaryChunkStrong.astype(np.int8)
            else:
                if Parameters['DETECT_POSITIVE']:
                    BinaryChunk = np.abs(FilteredChunk)>Threshold
                else:
                    BinaryChunk = (FilteredChunk<-Threshold)
                BinaryChunk = BinaryChunk.astype(np.int8)
        # write filtered output to file
        #if Parameters['WRITE_FIL_FILE']:
        #    fil_writer.write(FilteredChunk, s_start, s_end, keep_start, keep_end)
        #    print 'I am here at line 313'

        ############### FLOOD FILL  ######################################
        ChannelGraphToUse = complete_if_none(ChannelGraph, N_CH)
        if (Parameters['USE_HILBERT'] or Parameters['USE_COMPONENT_ALIGNFLOATMASK']):
            IndListsChunk = connected_components_twothresholds(BinaryChunkWeak, BinaryChunkStrong,
                            ChannelGraphToUse, S_JOIN_CC)
            BinaryChunk = 1 * BinaryChunkWeak + 1 * BinaryChunkStrong
        else:
            IndListsChunk = connected_components(BinaryChunk,
                            ChannelGraphToUse, S_JOIN_CC)
            
        
        if Parameters['DEBUG']:  #TO DO: Change plot_diagnostics for the HILBERT case
            if Parameters['USE_HILBERT']:
                plot_diagnostics_twothresholds(s_start,IndListsChunk,BinaryChunk,DatChunk,FilteredChunk,FilteredChunkHilbert,ThresholdStrong,ThresholdWeak)
            elif Parameters['USE_COMPONENT_ALIGNFLOATMASK']:
                plot_diagnostics_twothresholds(s_start,IndListsChunk,BinaryChunk,DatChunk,FilteredChunk,-FilteredChunk,ThresholdStrong,ThresholdWeak)#TODO: change HIlbert in plot_diagnostics_twothresholds
            else:
                plot_diagnostics(s_start,IndListsChunk,BinaryChunk,DatChunk,FilteredChunk,Threshold)
        if Parameters['WRITE_BINFIL_FILE']:
            fil_writer.write_bin(BinaryChunk, s_start, s_end, keep_start, keep_end)
        
        print len(IndListsChunk)
        ############## ALIGN AND INTERPOLATE WAVES #######################
        nextbits = []
        if Parameters['USE_HILBERT']:
            
            for IndList in IndListsChunk:
                try:
                    wave, s_peak, cm, fcm = extract_wave_hilbert_new(IndList, FilteredChunk,
                                                    FilteredChunkHilbert,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start, ThresholdStrong, ThresholdWeak)
                    s_offset = s_start+s_peak
                    if keep_start<=s_offset<keep_end:
                        spike_count += 1
                        nextbits.append((wave, s_offset, cm, fcm))
                except np.linalg.LinAlgError:
                    s = '*** WARNING *** Unalignable spike discarded in chunk {chunk}.'.format(
                            chunk=(s_start, s_end))
                    log_warning(s)
                except InterpolationError:
                    s = '*** WARNING *** Interpolation error in chunk {chunk}.'.format(
                            chunk=(s_start, s_end))
                    log_warning(s)
            # and return them in time sorted order
            nextbits.sort(key=lambda (wave, s, cm, fcm): s)
            for wave, s, cm, fcm in nextbits:
                uwave = get_padded(DatChunk, int(s)-S_BEFORE-s_start,
                                   int(s)+S_AFTER-s_start).astype(np.int32)
                # cm = add_penumbra(cm, ChannelGraphToUse,
                                  # Parameters['PENUMBRA_SIZE'])
                # fcm = get_float_mask(wave, cm, ChannelGraphToUse,
                                     # 1.)
                yield uwave, wave, s, cm, fcm
        elif Parameters['USE_COMPONENT_ALIGNFLOATMASK']:
            for IndList in IndListsChunk:
                try:
                    if Parameters['DETECT_POSITIVE']:
                        wave, s_peak, cm, fcm = extract_wave_twothresholds(IndList, FilteredChunk,
                                                    FilteredChunk,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start, ThresholdStrong, ThresholdWeak) 
                    else:
                        wave, s_peak, cm, fcm = extract_wave_twothresholds(IndList, FilteredChunk,
                                                    -FilteredChunk,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start, ThresholdStrong, ThresholdWeak)
                    s_offset = s_start+s_peak
                    if keep_start<=s_offset<keep_end:
                        spike_count += 1
                        nextbits.append((wave, s_offset, cm, fcm))
                except np.linalg.LinAlgError:
                    s = '*** WARNING *** Unalignable spike discarded in chunk {chunk}.'.format(
                            chunk=(s_start, s_end))
                    log_warning(s)
                except InterpolationError:
                    s = '*** WARNING *** Interpolation error in chunk {chunk}.'.format(
                            chunk=(s_start, s_end))
                    log_warning(s)
            # and return them in time sorted order
            nextbits.sort(key=lambda (wave, s, cm, fcm): s)
            for wave, s, cm, fcm in nextbits:
                uwave = get_padded(DatChunk, int(s)-S_BEFORE-s_start,
                                   int(s)+S_AFTER-s_start).astype(np.int32)
                # cm = add_penumbra(cm, ChannelGraphToUse,
                                  # Parameters['PENUMBRA_SIZE'])
                # fcm = get_float_mask(wave, cm, ChannelGraphToUse,
                                     # 1.)
                yield uwave, wave, s, cm, fcm   
        else:    #Original SpikeDetekt. This code duplication is regretable but probably easier to deal with
            
            for IndList in IndListsChunk:
                try:
                    wave, s_peak, cm = extract_wave(IndList, FilteredChunk,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start,Threshold)
                    s_offset = s_start+s_peak
                    if keep_start<=s_offset<keep_end:
                        spike_count += 1
                        nextbits.append((wave, s_offset, cm))
                except np.linalg.LinAlgError:
                    s = '*** WARNING *** Unalignable spike discarded in chunk {chunk}.'.format(
                            chunk=(s_start, s_end))
                    log_warning(s)
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
