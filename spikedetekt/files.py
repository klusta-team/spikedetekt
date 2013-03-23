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


def chunks(DatFileNames, n_ch_dat, ChannelsToUse):
    '''
    Yields the chunks from the data file
    '''
    CHUNK_SIZE = Parameters['CHUNK_SIZE']
    CHUNK_OVERLAP = Parameters['CHUNK_OVERLAP']
    DTYPE = Parameters['DTYPE']
    dtype_size = np.nbytes[DTYPE]
    n_samples = [num_samples(DatFileName,
                             n_ch_dat,
                             n_bytes=dtype_size) for DatFileName in DatFileNames]
    n_samples = np.array(n_samples, dtype=int)
    offsets = np.hstack((0, np.cumsum(n_samples)))
    total_n_samples = np.sum(n_samples)
    fileobjs = [open(DatFileName, 'rb') for DatFileName in DatFileNames]
    objs_and_offsets = zip(fileobjs, offsets[:-1], offsets[1:])
    for s_start, s_end, keep_start, keep_end in chunk_bounds(total_n_samples,
                                                             CHUNK_SIZE,
                                                             CHUNK_OVERLAP):
        # read from sample s_start to sample s_end, i.e. bytes from
        # s_start*n_ch_dat*sizeof(DTYPE) to s_end*n_ch_dat*sizeof(DTYPE)
        # but we are reading from a virtual concatenated file
        pieces = []
        for fd, f_start, f_end in objs_and_offsets:
            # find the intersection of [f_start, f_end] and [s_start, s_end]
            i_start = max(f_start, s_start)
            i_end = min(f_end, s_end)
            # intersection is nonzero if i_end>i_start only
            if i_end>i_start:
                # start and end of intersection as an offset into the file in
                # samples
                o_start = i_start-f_start
                o_end = i_end-f_start
                # start of the data in bytes
                fd.seek(o_start*n_ch_dat*dtype_size, 0)
                DatChunk = np.fromfile(fd, dtype=DTYPE,
                                       count=(o_end-o_start)*n_ch_dat)
                DatChunk = DatChunk.reshape(o_end-o_start, n_ch_dat)
                DatChunk = DatChunk[:, ChannelsToUse]
                DatChunk = DatChunk.astype(np.float32)
                pieces.append(DatChunk)
        if len(pieces)==1:
            DatChunk = pieces[0]
        else:
            DatChunk = np.vstack(pieces)
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

#def spike_dtype():
#    N_CH, S_TOTAL, FPC = eval('(N_CH, S_TOTAL, FPC)', Parameters)
#    class description(IsDescription):    
#        time = Int32Col()
#        channel_mask = Int8Col(shape=(N_CH,))
#        float_channel_mask = Float32Col(shape=(N_CH,))
#        wave = Float32Col(shape=(S_TOTAL, N_CH))
#        unfiltered_wave = Int32Col(shape=(S_TOTAL, N_CH))
#        fet = Float32Col(shape=(N_CH, FPC))
#        clu = Int32Col()
#        fet_mask = Int8Col(shape=(1+FPC*N_CH,))
#        float_fet_mask = Float32Col(shape=(1+FPC*N_CH,))
#    return description

def shank_description(shanksize):
    s_total = Parameters['S_TOTAL']
    fpc = Parameters['FPC']
    class description(IsDescription):
        time = Int32Col()
        mask_binary = Int8Col(shape=(shanksize,))
        mask_float = Float32Col(shape=(shanksize,))
        features = Float32Col(shape=(1+fpc*shanksize,))
    return description

def waveform_description(shanksize):
    s_total = Parameters['S_TOTAL']
    class description(IsDescription):
        wave = Float32Col(shape=(s_total, shanksize))
        unfiltered_wave = Float32Col(shape=(s_total, shanksize))
    return description    


def klusters_files(h5s, shank_table, basename, probe):
    N_CH,  FPC = eval('(N_CH, FPC)', Parameters)
    for shank in probe.shanks_set:
        T = shank_table['spikedetekt', shank]
        write_fet(T.cols.features[:], basename+'.fet.'+str(shank))
        time = T.cols.time[:]
        write_trivial_clu(time, basename+'.clu.'+str(shank))
        write_res(time, basename+'.res.'+str(shank))
        write_spk_buffered(shank_table['waveforms', shank],
                           'wave', basename+'.spk.'+str(shank),
                           np.arange(len(time)))
        write_spk_buffered(shank_table['waveforms', shank],
                           'unfiltered_wave', basename+'.uspk.'+str(shank),
                           np.arange(len(time)))
        write_xml(probe,
                  n_ch=Parameters['N_CH'],
                  n_samp=Parameters['S_TOTAL'],
                  n_feat=Parameters['FPC'],
                  sample_rate=Parameters['SAMPLE_RATE'],
                  filepath=basename+'.xml')
        # compute feature masks from channel masks
        M = np.repeat(T.cols.mask_binary[:], Parameters['FPC'], axis=1)
        M = np.hstack((M, np.zeros(M.shape[0], dtype=M.dtype)[:, np.newaxis]))
        write_mask(M, basename+'.mask.'+str(shank))
        if Parameters['USE_FLOAT_MASKS']:
            M = np.repeat(T.cols.mask_float[:], Parameters['FPC'], axis=1)
            M = np.hstack((M, np.zeros(M.shape[0], dtype=M.dtype)[:, np.newaxis]))
            write_mask(M, basename+'.fmask.'+str(shank), fmt='%f')


def write_mask(mask, filename, fmt="%i"):
    fd = open(filename, 'w')
    fd.write(str(mask.shape[1])+'\n') # number of features
    np.savetxt(fd, mask, fmt=fmt)
    fd.close()

def num_samples(FileNames, n_ch_dat, n_bytes=2):
    if isinstance(FileNames, str):
        FileNames = [FileNames]
    total_bytes = sum(os.path.getsize(FileName) for FileName in FileNames)
    if total_bytes % (n_ch_dat*n_bytes) != 0:
        raise Exception("Size of file(s) %s not consistent with %i channels and %i bytes"%(', '.join(FileNames), n_ch_dat, n_bytes))
    return total_bytes//(n_ch_dat*n_bytes)

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

def write_trivial_clu(restimes,filepath):
    """writes cluster cluster assignments to text file readable by klusters and neuroscope.
    input: clus is a 1D or 2D numpy array of integers
    output:
        top line: number of clusters (max cluster)
        next lines: one integer per line"""
    clus = np.zeros_like(restimes) 
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

def write_fet(feats, filepath):
    feat_file = open(filepath, 'w')
    feats = np.array(feats, dtype=np.int32)
    #header line: number of features
    feat_file.write('%i\n' % feats.shape[1])
    #next lines: one feature vector per line
    np.savetxt(feat_file, feats, fmt="%i")
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

def write_spk_buffered(table, column, filepath, indices,
                       channels=slice(None), buffersize=512):
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
