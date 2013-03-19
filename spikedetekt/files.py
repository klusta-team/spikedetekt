'''
File handling routines, to separate data access from algorithm details.
'''
import os
from utils import basename_noext
from tables import IsDescription, Int32Col, Float32Col, Int8Col
import numpy as np
from xml.etree.ElementTree import ElementTree,Element,SubElement
from utils import switch_ext
import os.path
from parameters import Parameters

#m chops n_samples into chunks according to chunk_size,overlap
#m Overlap probably controls for the artifacts of filtering on the ends of the signal?
def chunk_bounds(n_samples, chunk_size, overlap):
    '''
    Returns chunks of the form:
    [ overlap/2 | chunk_size-overlap | overlap/2 ]
    s_start   keep_start           keep_end     s_end
    Except for the first and last chunks which do not have a left/right overlap
    '''
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end-overlap//2
    yield s_start,s_end,keep_start,keep_end
    
    while s_end-overlap+chunk_size < n_samples:
        s_start = s_end-overlap
        s_end = s_start+chunk_size
        keep_start = keep_end
        keep_end = s_end-overlap//2
        yield s_start,s_end,keep_start,keep_end
        
    s_start = s_end-overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    yield s_start,s_end,keep_start,keep_end   

def chunks(fd, n_ch_dat, ChannelsToUse, n_samples):
    '''
    Yields the chunks from the data file
    '''
    CHUNK_SIZE = Parameters['CHUNK_SIZE']
    CHUNK_OVERLAP = Parameters['CHUNK_OVERLAP']
    DTYPE = Parameters['DTYPE']
    for s_start, s_end, keep_start, keep_end in chunk_bounds(n_samples,
                                                             CHUNK_SIZE,
                                                             CHUNK_OVERLAP):
        # Load next chunk from file
        DatChunk = np.fromfile(fd, dtype=DTYPE,
                               count=(s_end-s_start)*n_ch_dat)
        DatChunk = DatChunk.reshape(s_end-s_start, n_ch_dat)
        DatChunk = DatChunk[:, ChannelsToUse]
        DatChunk = DatChunk.astype(np.float32)
        fd.seek(fd.tell()-CHUNK_OVERLAP*n_ch_dat*np.nbytes[DTYPE])
        yield DatChunk, s_start, s_end, keep_start, keep_end

def get_chunk_for_thresholding(fd, n_ch_dat, ChannelsToUse, n_samples):
    '''
    Returns the initial chunk of the file for doing thresholding
    '''
    CHUNK_SIZE = Parameters['CHUNK_SIZE']
    CHUNKS_FOR_THRESH = Parameters['CHUNKS_FOR_THRESH']
    DTYPE = Parameters['DTYPE']
    n_samps_thresh = min(CHUNK_SIZE*CHUNKS_FOR_THRESH, n_samples)
    x = np.fromfile(fd, dtype=DTYPE, count=n_samps_thresh*n_ch_dat)
    DatChunk = np.fromfile(fd, dtype=DTYPE, count=n_samps_thresh*n_ch_dat)
    DatChunk = DatChunk.reshape(n_samps_thresh, n_ch_dat)[:, ChannelsToUse]
    DatChunk = DatChunk.astype(np.int32)
    fd.seek(0)
    return DatChunk

def spike_dtype():
    N_CH, S_TOTAL, FPC = eval('(N_CH, S_TOTAL, FPC)', Parameters)
    class description(IsDescription):    
        time = Int32Col()
        channel_mask = Int8Col(shape=(N_CH,))
        float_channel_mask = Float32Col(shape=(N_CH,))
        wave = Float32Col(shape=(S_TOTAL, N_CH))
        unfiltered_wave = Int32Col(shape=(S_TOTAL, N_CH))
        fet = Float32Col(shape=(N_CH, FPC))
        clu = Int32Col()
        fet_mask = Int8Col(shape=(1+FPC*N_CH,))
        float_fet_mask = Float32Col(shape=(1+FPC*N_CH,))
    return description    

def klusters_files(table, basename, probe):
    N_CH,  FPC = eval('(N_CH, FPC)', Parameters)
    CM = table.cols.channel_mask[:] # shape (numspikes, numfeatures)
    shanknum = np.zeros(CM.shape[0], dtype=int)
    for i in xrange(CM.shape[0]):
        M = CM[i, :]
        j = M.nonzero()[0][0] # first nonzero element
        shanknum[i] = probe.channel_to_shank[j]
    for shank in probe.shanks_set:
        I = (shanknum==shank).nonzero()[0] # set of spike indices on given shank
        C = np.array(sorted(list(probe.channel_set[shank])),dtype=int)
        write_clu(table.cols.clu[:][I], basename+'.clu.'+str(shank))
        F = table.cols.fet[:][I]
        L = F[:,C,:]
        write_fet(L.reshape(L.shape[0],-1),
                  basename+'.fet.'+str(shank),
                  samples=table.cols.time[:][I])
        write_res(table.cols.time[:][I], basename+'.res.'+str(shank))
        write_spk_buffered(table,'wave', basename+'.spk.'+str(shank),indices=I,channels=C)
        write_spk_buffered(table,'unfiltered_wave', basename+'.uspk.'+str(shank), indices=I, channels=C)

        write_xml(probe,
                  n_ch=Parameters['N_CH'],
                  n_samp=Parameters['S_TOTAL'],
                  n_feat=Parameters['FPC'],
                  sample_rate=Parameters['SAMPLE_RATE'],
                  filepath=basename+'.xml')
        CFPC = C*FPC     
#       Multiplies the channel indices by the numebr of principle components, FPC    
        Cmaskindices = np.tile(np.arange(FPC,dtype=int),len(C)) + CFPC.repeat(FPC,axis=None)
#       Takes list of channel indices, e.g. [0 6  13] and outputs mask indices
#        according to number of principle components: [0 1 2 15 16 17 36 37 38]   
        Cmaskindices = np.append(Cmaskindices,[FPC*N_CH])
#        print 'Cmaskindices = ', Cmaskindices 
        G = table.cols.fet_mask[:][I]        
        Gf = table.cols.float_fet_mask[:][I]
        if Parameters['USE_FLOAT_MASKS']:
            write_mask(Gf[:,Cmaskindices], basename+'.fmask.'+str(shank), fmt="%f")
        write_mask(G[:,Cmaskindices], basename+'.mask.'+str(shank))

def write_mask(mask, filename, fmt="%i"):
    fd = open(filename, 'w')
    fd.write(str(mask.shape[1])+'\n') # number of features
    np.savetxt(fd, mask, fmt=fmt)
    fd.close()

def processed_basename(DatFileName, ProbeFileName):
    return "%s_%s"%(basename_noext(DatFileName),
                    basename_noext(ProbeFileName))
       
def num_samples(FileName, n_ch_dat, n_bytes=2):
    total_bytes = os.path.getsize(FileName)
    if total_bytes % (n_ch_dat*n_bytes) != 0:
        raise Exception("Size of file %s is not consistent with %i channels and %i bytes"%(FileName, n_ch_dat, n_bytes))
    return os.path.getsize(FileName)//n_ch_dat//n_bytes

def write_clu(clus, filepath):
    """writes cluster cluster assignments to text file readable by klusters and neuroscope.
    input: clus is a 1D or 2D numpy array of integers
    output:
        top line: number of clusters (max cluster)
        next lines: one integer per line"""
    clu_file = open( filepath,'w')
    #header line: number of clusters
    n_clu = clus.max()+1
    clu_file.write( '%i\n'%n_clu)
    #one cluster per line
    np.savetxt(clu_file,np.int16(clus),fmt="%i")
    clu_file.close()
    
    
def read_clu(filepath):
    """skip first line, read the rest into an array and return it"""
    return np.loadtxt(filepath, dtype=np.int32, skiprows=1)

def write_fet(feats,filepath,samples=None):
    """writes array of feature vectors to text file readable by klusters and klustakwik
    FOR KLUSTERS, YOU MUST GIVE samples! OTHERWISE IT WILL CRASH WITHOUT SENSIBLE MESSAGE
    input: feats is a 2D ndarray of floats. n_vectors x n_features per vector
        optionally also input samples (times) vector
    output:
        top line: number of features
        next line: one feature vector per line, as integers. last column is time vector if specified.
        last line ends in newline."""
    feat_file = open(filepath,'w')
    #rescaling features so they line between -16383 and 16384
    #feat_scaling_factor = 16000./max(feats.max(),-feats.min())
    #feats *= feat_scaling_factor
    feats = np.int32(feats)
    if samples is not None:
        feats = np.hstack( (feats,samples.reshape(-1,1)) )
    #header line: number of features
    feat_file.write( '%i\n'%feats.shape[1] )
    #next lines: one feature vector per line
    np.savetxt(feat_file,feats,fmt="%i")
    feat_file.close()
    
def read_fet(filepath):
    """reads feature file and returns it as an array. note that the last
    column might contain the times"""
    #skip first line and read the rest
    return np.loadtxt(filepath,dtype=np.int32,skiprows=1).astype(np.float32)
    
    
def write_res(samples,filepath):
    """input: 1D vector of times shape = (n_times,) or (n_times, 1)
    output: writes .res file, which has integer sample numbers"""
    np.savetxt(filepath,samples,fmt="%i")
    
def read_res(filepath):
    """reads .res file, which is just a list of integer sample numbers"""
    return np.loadtxt( filepath,dtype=np.int32)

def write_spk(waves,filepath,nonzero=None):
    """input: waves: 3D array of waveforms. n_spikes x n_channels x n_samples
    nonzero [optional]: 2D boolean array n_spikes x n_channels
    rescaled to signed 16-bit integer and written to file filedir/filebase.spk.1"""
    #wave_scaling_factor = 16000./max(waves.max(),-waves.min())
    if nonzero is not None:
        waves = waves*nonzero.reshape( nonzero.shape + (1,) )
    #waves *= wave_scaling_factor
    waves = np.int16(waves)
    waves.tofile(filepath)

def write_spk_buffered(table, column, filepath, indices, channels, buffersize=512):
    with open(filepath, 'wb') as f:
        numitems = len(indices)
        for i in xrange(0, numitems, buffersize):
            waves = table[indices[i:i+buffersize]][column]
            waves = waves[:, :, channels]
            waves = np.int16(waves)
            waves.tofile(f)
    
def read_spk(filepath,n_ch,n_s):
    return np.fromfile(filepath,dtype=np.int16).reshape(-1,n_s,n_ch)
    
def write_xml(probe,n_ch,n_samp,n_feat,sample_rate,filepath):
    """makes an xml parameters file so we can look at the data in klusters"""
    parameters = Element('parameters')
    acquisitionSystem = SubElement(parameters,'acquisitionSystem')
    SubElement(acquisitionSystem,'nBits').text = '16'
    SubElement(acquisitionSystem,'nChannels').text = str(n_ch)
    SubElement(acquisitionSystem,'samplingRate').text = str(int(sample_rate))
    SubElement(acquisitionSystem,'voltageRange').text = '20'
    SubElement(acquisitionSystem,'amplification').text = "1000"
    SubElement(acquisitionSystem,'offset').text = "2048"
    
    anatomicalDescription = SubElement(SubElement(parameters,'anatomicalDescription'),'channelGroups')
    for shank in probe.shanks_set:
        shankgroup = SubElement(anatomicalDescription,'group')
        for i_ch in probe.channel_set[shank]:
            SubElement(shankgroup,'channel').text=str(i_ch)
#    channels = SubElement(SubElement(SubElement(parameters,'channelGroups'),'group'),'channels')
#    for i_ch in range(n_ch):
#        SubElement(channels,'channel').text=str(i_ch)
    
    spikeDetection = SubElement(SubElement(parameters,'spikeDetection'),'channelGroups')
    for shank in probe.shanks_set:
        shankgroup = SubElement(spikeDetection,'group')
        channels = SubElement(shankgroup,'channels')
        for i_ch in probe.channel_set[shank]:
            SubElement(channels,'channel').text=str(i_ch)
#    channels = SubElement(group,'channels')
#    for i_ch in range(n_ch):
#        SubElement(channels,'channel').text=str(i_ch)
        SubElement(shankgroup,'nSamples').text = str(n_samp)
        SubElement(shankgroup,'peakSampleIndex').text = str(n_samp//2)
        SubElement(shankgroup,'nFeatures').text = str(n_feat)
    
    indent_xml(parameters)
    ElementTree(parameters).write(filepath)    
    

def indent_xml(elem, level=0):
    """input: elem = root element
    changes text of nodes so resulting xml file is nicely formatted.
    copied from http://effbot.org/zone/element-lib.htm#prettyprint"""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            
            
def walk_etree(root):
    yield root.tag,root.text
    for child in root.getchildren():
        for tag,text in walk_etree(child):
            yield tag,text
    
def search_etree(root,the_tag):
    for tag,text in walk_etree(root):
        if tag==the_tag: return text
