'''
Various assorted useful functions used throughout
'''
from __future__ import division, with_statement
from time import time
import itertools as it,operator as op, numpy as np 
import re,cPickle, os
from os.path import join

########################
####### Arrays/math ####
########################

def get_padded(Arr, Start, End):
    '''
    Returns Arr[Start:End] filling in with zeros outside array bounds
    
    Assumes that EITHER Start<0 OR End>len(Arr) but not both (raises error).
    '''
    if Start < 0 and End >= Arr.shape[0]:
        raise IndexError("Can have Start<0 OR End>len(Arr) but not both.")
    if Start < 0:
        StartZeros = np.zeros((-Start, Arr.shape[1]), dtype=Arr.dtype)
        return np.vstack((StartZeros, Arr[:End]))
    elif End > Arr.shape[0]:
        EndZeros = np.zeros((End-Arr.shape[0], Arr.shape[1]), dtype=Arr.dtype)
        return np.vstack((Arr[Start:], EndZeros))
    else:
        return Arr[Start:End]
        
########################
###### Files ###########
########################

class indir(object):
    '''
    Safely temporarily work in a subdirectory
    
    Usage::
    
        with indir(directory):
            ... do stuff here
            
    Guarantees that the code in the with block will be executed in directory,
    and that after the block is completed we return to the original directory.
    Also prints a record of which directories were entered.
    '''
    def __init__(self,new_dir):
        self.orig_dir = os.getcwd()
        self.new_dir = new_dir
    def __enter__(self):
        print("entering %s"%self.new_dir)
        mkdir_and_enter(self.new_dir)
    def __exit__(self,*exc_info):
        print("exiting %s"%self.new_dir)        
        os.chdir(self.orig_dir)

def mkdir_and_enter(DirName):
    '''
    Creates a directory and enters it. If the directory already exists, creates
    a new version with '_1' appended to it. Returns the directory name.
    '''
    if not os.path.exists(DirName):
        os.mkdir(DirName)
        os.chdir(DirName)
    else:
        is_repeat = re.match("(.+_)(\d+)$",DirName)
        if is_repeat:
            DirName = is_repeat.group(1)+str(int(is_repeat.group(2))+1)
        else:
            DirName = DirName+"_1"
        DirName = mkdir_and_enter(DirName)
    return DirName

# Dan: Not sure what the point of this command is that makes it different
# from os.path.splitext but it seems harmless and safe to just use it.
def splitext(filename):
    m = re.match(r"(.+)\.(\w+\.\d+)",filename)
    return m.groups() if m is not None else os.path.splitext(filename)    
        
def switch_ext(filepath,new_ext):
    '''
    Returns a filename with the extension (part after the .) changed to another
    one, e.g. switch_ext('test.txt', 'c')=='test.c'.
    '''
    return splitext(filepath)[0]+"."+new_ext

def basename_noext(filepath):
    '''
    Returns the filename, without the directory or the extension, e.g.
    basename_noext("/home/username/test.txt')=='test'.
    '''
    return splitext(os.path.basename(filepath))[0]
