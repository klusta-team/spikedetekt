#!/usr/bin/env python
'''
TODO: documentation
'''
import sys, os, shutil
from optparse import OptionParser, Option
from os.path import abspath, dirname, join
from spikedetekt.core import classify_from_raw_data
from spikedetekt.parameters import Parameters

usage = """
This is the main script that you use to spike-sort your data. Just make the
probe file and you're good to go (see documentation). If you don't specify a
probe file or probe directory, I will use every probe file in directory of dat
file. I will prompt you for sample rate and number of channels if no xml file is
found.

%prog your_dat_file.dat [options]
%prog -h displays help
"""
        
probe = Option("-p", "--probe", action="store", dest="probe",
    help="Specify the location of the probe file. By default, program looks "
         "for a .probe file in .dat directory")

max_spikes = Option("-n", type=int, dest="max_spikes",
    help="Extract and cluster only the first n spikes.")

output = Option("-o", "--output", action="store", dest="output",
    help="Directory where the output directory 'basename/' ends up. By "
          "default, output goes next to .dat file.")

params = Option("--params", action="store", dest="params",
                help="Set parameters by executing selected file. By default I "
                     "load parameters from %s. If you're really sure of "
                     "yourself, modify that file. Otherwise, make a copy of it "
                     "and modify values."%join(dirname(abspath(__file__)),
                                               "parameters.py"))

kwdparams = Option("--kwdparams", action="store", dest="kwdparams",
                   help="Set keyword parameters, the argument should be a "
                        "string with a Python dictionary of keyword/argument "
                        '''pairs, e.g. --kwdparams "{'FPC':4}".''')

parser = OptionParser(usage)
parser.add_options([probe, max_spikes, output, params, kwdparams])
parser.add_option("--probe-dir", action="store", dest="probe_dir",
                  help="Run the clustering script once for every probe file in "
                       "the specified directory. Defaults to the directory of "
                       "the data file. If you want to select a single probe "
                       "file, use the -p (or --probe=) option.")

if __name__ == '__main__':
    (opts,args) = parser.parse_args()                
        
    if len(args) == 0:
        parser.error("Must specify a dat file")
                        
    DatFileName = args[0]
    if not os.path.exists(DatFileName):
        parser.error("raw data file not found: %s"%DatFileName)

    if opts.probe is not None:
        probe_files = [opts.probe]
    else:
        probe_dir = opts.probe_dir or os.path.dirname(os.path.abspath(DatFileName))
        print "I'll use all probe files in %s"%probe_dir
        probe_files = [os.path.join(probe_dir, fname) for fname in os.listdir(probe_dir) if fname.endswith(".probe")]
    print "%s will be run for each probe file in %s"%(__file__, probe_files)
    
    if opts.params is not None:
        execfile(opts.params, Parameters)
        
    if opts.kwdparams is not None:
        Parameters.update(eval(opts.kwdparams))
    
    if len(probe_files) == 0: parser.error("no probe files found!")
    for probe_file in probe_files:
        classify_from_raw_data(DatFileName, probe_file,
                               max_spikes=opts.max_spikes,
                               output_dir=opts.output)
