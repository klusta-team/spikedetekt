#!/usr/bin/env python
'''
Main script file for SpikeDetekt
'''
from spikedetekt.core import spike_detection_job
from spikedetekt.parameters import Parameters
import sys
import os

usage = '''
SpikeDetekt should be called as:

python detektspikes.py parameter_filename.py

All options must be specified in the parameters file.
'''

if __name__=='__main__':
    if len(sys.argv)<=1 or len(sys.argv)>2:
        print usage.strip()
        exit()
        
    # Read parameters file
    parameters_file = sys.argv[1]
    print 'Reading parameters from file', parameters_file
    execfile(parameters_file, Parameters)
        
    # Make sure we have probe and raw data, and that the files exist
    try:
        probe_file = Parameters['PROBE_FILE']
    except KeyError:
        print 'Parameters file needs a PROBE_FILE option.'
        exit()
    if not os.path.exists(probe_file):
        print 'Probe file %s does not exist.' % probe_file
    
    try:
        raw_data_files = Parameters['RAW_DATA_FILES']
    except KeyError:
        print 'Parameters file needs a RAW_DATA_FILES option.'
        exit()
    for file in raw_data_files:
        if not os.path.exists(file):
            print 'Raw data file %s does not exist.' % file
            exit()
    
    # Check other options are present in parameters file
    if not 'NCHANNELS' in Parameters or not 'SAMPLERATE' in Parameters:
        print 'Parameters file needs NCHANNELS and SAMPLERATE options.'
        exit()
        
    if len(raw_data_files)>1:
        print 'Can only process a single data file at the moment, multiple files will be added later.'
        exit()
            
    spike_detection_job(raw_data_files[0], probe_file,
                        max_spikes=Parameters['MAX_SPIKES'],
                        output_dir=Parameters['OUTPUT_DIR'])
    