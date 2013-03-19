"""
Functions for creating float masks
"""

from numpy import *
from parameters import Parameters
from graphs import add_penumbra

def get_float_mask(wave, channelmask, channelgraph, sdfactor):
    '''
    Input arguments are:
    
    wave
        An array of shape (nsamples, nchannels) giving the aligned wave on all
        channels
    channelmask
        A boolean array of length nchannels giving the unmasked channels
        returned for the connected component
    channelgraph
        The graph of the channels, a dictionary with keys the channel indices
        and values a set of neighbouring channels
    sdfactor
        The standard deviation, so that wave/sdfactor is dimensionless
        
    Should return an array of floats between 0 and 1 of length nchannels.
    '''
    # wavemax should be the maximum (or minimum in case of negative thresholds)
    # value of the wave for each channel, we use this to construct the mask
    if Parameters['DETECT_POSITIVE']:
        wavemax = amax(abs(wave), axis=0)
    else:
        wavemax = amax(-wave, axis=0)
    # z score is this value normalised by the standard deviation
    z = wavemax/sdfactor
    zmin, zmax = Parameters['FLOAT_MASK_THRESH_SD']
    # x score is between 0 and 1, 0 at the minimum threshold in SD, and 1 at the
    # maximum threshold in SD
    x = clip((z-zmin)/(zmax-zmin), 0, 1)
    #x = (z-zmin)/(zmax-zmin) #For use when actual values are desired (use together with a high value of ADDITIONAL_FLOAT_PENUMBRA)
    if Parameters['USE_INTERPOLATION']:
        # the interpolation function should use the channelmask            
        channelmask = add_penumbra(channelmask, channelgraph,
                                   Parameters['ADDITIONAL_FLOAT_PENUMBRA'])
        # and this function varies from 0 to 1 for x varying from 0 to 1
        return eval(Parameters['FLOAT_MASK_INTERPOLATION'])*channelmask
    else:
        newchannelmask = channelmask.astype(float32)
        channelmaskdifference={}
        for j in range(Parameters['ADDITIONAL_FLOAT_PENUMBRA']):
            channelmaskdifference[j] =  (add_penumbra(channelmask, channelgraph,j+1)*1 -add_penumbra(channelmask, channelgraph,j)*1)
            channelmaskdifference[j] = channelmaskdifference[j].astype(float32)
            channelmaskdifference[j] = channelmaskdifference[j]/(2**(j+1))
            newchannelmask = newchannelmask+channelmaskdifference[j]
        return newchannelmask    
            
  
