from __future__ import division, with_statement
import os, numpy as np, collections, itertools as it
from files import write_fet, read_clu
import probes
from os.path import join
import multiprocessing

DEBUG = False
MINCLUSTERS = 3
MAXCLUSTERS = 14
ACCEPTABLE_FRAC = .8
MIN_CLU_SIZE = 10
NUMPROCESSES = 2

    
def spike_subsets(ST_nc, ChSubsets):
    ch2spikes = [np.flatnonzero(ST_n) for ST_n in ST_nc.transpose()]
    #  ch2spikes is a list of no.-of-channels arrays. Each array contains the consecutive number of spikes detected on the channel
    return [reduce(np.union1d,
                   [ch2spikes[ch] for ch in subset]) 
            for subset in ChSubsets]
    
#m
#m Inputs: spike_table - table with fields named "wave", "time", "st", "fet"
#m         reorder_clus - bool, has to do with the SORT_CLUS_BY_CHANNEL within Caton's running parameters file (ANYHOW not implemented)
#m Output: a 1D array containing the cluster to which every spike is assigned                       
def cluster_withsubsets(spike_table,reorder_clus=True):
    if reorder_clus: print "Cluster reordering not implemented!"
    ST_nc = np.bool8(spike_table.cols.channel_mask[:])
    Fet_nc3 = spike_table.cols.fet[:]    
    
    # TODO: implement this and remove the raise exception
    raise NotImplementedError("To use cluster_withsubsets you will need to implement some code to find the groups from the probe graph.")
    ChSubsets = probes.SORT_GROUPS #m these are all 4-channel subsets to be computed (based on probe's topology)
    
    SpkSubsets = spike_subsets(ST_nc, ChSubsets) #m for each subset  - the consecutive numbers of spikes that are relevant (?) 
    print "%i subsets total"%len(SpkSubsets)
    n_spikes, n_ch, _FPC = Fet_nc3.shape #m _FPC is no. of features per channel
    
#    for i_subset,ChHere,SpkHere in zip(it.count(), ChSubsets, SpkSubsets):   #m SpkHere - the consecutive numbers of spikes belonging to this subset     
#        print("Sorting channels %s"%ChHere.__repr__())
#        FetHere_nc3 = Fet_nc3[np.ix_(SpkHere, ChHere)] #m features of spikes in this subset
#        #m FetHere_nc3 is a 3D array of size (no. of spikes in this subset) x 4(subsets are of 4 channels) x 3 (no. of features per channel)
#        CluArr = klustakwik_cluster(FetHere_nc3, i_subset, ChHere, SpkHere)
#        print 'KlustaKwik returned', max(CluArr), 'clusters.'

    args = []
    for i_subset,ChHere,SpkHere in zip(it.count(), ChSubsets, SpkSubsets):   #m SpkHere - the consecutive numbers of spikes belonging to this subset
        print("Sorting channels %s"%ChHere.__repr__())
        FetHere_nc3 = Fet_nc3[np.ix_(SpkHere, ChHere)] #m features of spikes in this subset
        #m FetHere_nc3 is a 3D array of size (no. of spikes in this subset) x 4(subsets are of 4 channels) x 3 (no. of features per channel)
        args.append((FetHere_nc3, i_subset, ChHere, SpkHere))
        #CluArr = klustakwik_cluster(FetHere_nc3, i_subset, ChHere, SpkHere)
        #print 'KlustaKwik returned', max(CluArr), 'clusters.'
    pool = multiprocessing.Pool(NUMPROCESSES)
    pool.map(klustakwik_cluster_args, args)

def klustakwik_cluster_args(args):
    return klustakwik_cluster(*args)
            
def klustakwik_cluster(Fet_nc3, i_subset, ChHere, SpkHere, shank):
    kk_path = "KlustaKwik"
    kk_input_filepath = 'k_input.'+str(i_subset)+'.fet.'+shank
    kk_output_filepath = 'k_input.'+str(i_subset)+'.clu.'+shank
    open('channels.'+str(i_subset), 'w').write(' '.join(map(str, ChHere)))
    open('spikes.'+str(i_subset), 'w').write(' '.join(map(str, SpkHere)))
    Fet_nf = Fet_nc3.reshape(len(Fet_nc3),-1)
    write_fet(Fet_nf, kk_input_filepath)
    n_fet = Fet_nf.shape[1]
    
    os.system(' '.join([kk_path,
                        kk_input_filepath[:-6], shank,
                        '-UseFeatures', shank*n_fet,
                        '-MinClusters', str(MINCLUSTERS),
                        '-MaxClusters', str(MAXCLUSTERS),
                        '-Screen','0']))
    return read_clu(kk_output_filepath)
