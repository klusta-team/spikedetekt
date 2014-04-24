import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle 
from matplotlib.backends.backend_pdf import PdfPages
from parameters import Parameters, GlobalVariables
from IPython import embed # For manual debugging
from alignment import extract_wave, extract_wave_hilbert_old, extract_wave_hilbert_new,extract_wave_twothresholds, InterpolationError

#multdetection_times_ms = [4630, 4640]
#multdetection_times_ms = Parameters['OBSERVATION_TIMES']
#print Parameters['OBSERVATION_TIMES']


def plot_diagnostics(s_start,indlistchunk,binarychunk,datchunk,filteredchunk,threshold):
  #  print "Printing globals", globals()
    debug_fd = GlobalVariables['debug_fd']

    #print 'OBSERVATION TIMES are:', Parameters['OBSERVATION_TIMES'],'\n'
    samplingrate= Parameters['SAMPLERATE']
 

#  multdetection_times_ms = Parameters['OBSERVATION_TIMES']
  #  multdetection_times =  np.array(multdetection_times_ms, dtype=np.int32)
  #  multdetection_times = multdetection_times*samplingrate/1000
  #  multdetection_times = multdetection_times.astype(int)

    multdetection_times = Parameters['OBSERVATION_TIMES_SAMPLES']



    chunk_size_less= Parameters['CHUNK_SIZE']-200
#-Parameters['CHUNK_OVERLAP']
#    print 'Parameters: \n', Parameters
   # probefilename = Parameters['PROBE_FILE']
#    print 'chunk_size_less = ', chunk_size_less
    #window_width = 120
    #samples_backward = 60
    window_width = 140
    samples_backward = 70

 # path='/home/skadir/alignment/'
    path = Parameters['OUTPUT_DIR']
    for interestpoint in multdetection_times:
    #for interestpoint_ms in multdetection_times_ms:
    #    interestpoint = int(interestpoint_ms*samplingrate/1000)
   #      pp = PdfPages('/home/skadir/alignment/multipagegraphs.pdf')
        if (interestpoint - chunk_size_less) <= s_start < (interestpoint):
      #      print interestpoint_ms, ':\n'
      #      debug_fd.write(str(interestpoint_ms)+':\n')
            print interestpoint, ':\n'
            debug_fd.write(str(interestpoint)+':\n')
           # sampmin = interestpoint - s_start - 3
            sampmin = np.amax([0,interestpoint - s_start - samples_backward])
            sampmax = sampmin + window_width 
           # figdat= plt.figure()
          #  plt.figure()
          #   axdat = figdat.add_subplot(111) # stupid axis object (trivial subplot)
          #   axdat.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
           #  plt.savefig('%s datchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
            
         #    plt.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
         #    plt.savefig('%s datchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
         #    plt.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
         #    plt.savefig('%s filteredchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
         #    plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
          #   plt.savefig('%s binarychunk_%s.pdf'%(path,interestpoint))
         #   plt.rcParams('figure.subplot.hspace') = 0.3
      #      embed() # Start IPython

            # recover connected componentslist for our range of interest
            #e.g.indlist = [(19634, 9), (19635, 9), (19635, 11), (19636, 11)], want to know if any 
            #of the sample components are between sampmin and sampmax.
            
            connected_comp_enum = np.zeros_like(binarychunk)
            j = 0
            for k,indlist in enumerate(indlistchunk):
                indtemparray = np.array(indlist)
                if (set(indtemparray[:,0]).intersection(np.arange(sampmin,sampmax+1)) != set()):
                    
                    print k,':',indlist, '\n'
                    print '\n'
                    j = j+1
                    connected_comp_enum[indtemparray[:,0],indtemparray[:,1]] = j
                    
                    debug_fd.write(str(k)+': '+'\n')
                    debug_fd.write(str(indlist)+'\n')
                    debug_fd.write('\n') 
                    debug_fd.flush()       # makes sure everything is written to the debug file as program proceeds 
                    
                    
                    

            plt.figure()
          #  plt.suptitle('%s \n with %s \n Time %s ms'%(Parameters['RAW_DATA_FILES'],Parameters['PROBE_FILE'],interestpoint_ms), fontsize=10, fontweight='bold')
            plt.suptitle('%s \n with %s \n Time %s samples'%(Parameters['RAW_DATA_FILES'],Parameters['PROBE_FILE'],interestpoint), fontsize=10, fontweight='bold')
            plt.subplots_adjust(hspace = 0.5)
            dataxis = plt.subplot(3,2,1)
            dataxis.set_title('DatChunks',fontsize=10)
            imdat = dataxis.imshow(np.transpose(datchunk[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imdat);
            binaxis = plt.subplot(3,2,3)
            binaxis.set_title('FilteredChunks',fontsize=10)
            imbin = binaxis.imshow(np.transpose(filteredchunk[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imbin);
            filaxis = plt.subplot(3,2,2)
            filaxis.set_title('BinChunks',fontsize=10)
            imfil = filaxis.imshow(np.transpose(binarychunk[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imfil);
            conaxis = plt.subplot(3,2,4)
            conaxis.set_title('Connected Components',fontsize=10)
            imcon = conaxis.imshow(np.transpose(connected_comp_enum[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imcon);
            sdaxis = plt.subplot(3,2,5)
            sdaxis.set_title('SDChunks',fontsize=10)
            if Parameters['USE_CHANNEL_INDEPENDENT_THRESHOLD']:
                imsd = sdaxis.imshow(np.transpose(filteredchunk[sampmin:sampmax,:]/(-threshold)),interpolation="nearest");plt.colorbar(imsd);
            else: 
                imsd = sdaxis.imshow(np.transpose(filteredchunk[sampmin:sampmax,:]/(-threshold[:])),interpolation="nearest");plt.colorbar(imsd);

            #plt.savefig('%s_floodfillchunk_%s.pdf'%(path,interestpoint_ms))
            #plt.savefig('floodfillchunk_%s.pdf'%(interestpoint_ms))
            plt.savefig('floodfillchunk_%s_samples.pdf'%(interestpoint))
            
# plt.show()
   #         plt.figure();plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();plt.savefig(pp,format='pdf')
        #     figbin=plt.figure()
       #      plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest")
        #     plt.colorbar()
       #      pp.savefig(figbin)
            
        #    figfil= plt.figure()
        #    axfil = figfil.add_subplot(111) # stupid axis object (trivial subplot)
        #    axfil.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
        #    plt.savefig('%s filteredchunk_%s.pdf'%(path,interestpoint))
            
           
    #     pp.close()

#Grossly inelegant, but will do for now. For use with the Hilbert transform
def plot_diagnostics_twothresholds(s_start,indlistchunk,binarychunkweak, binarychunkstrong,binarychunk,datchunk,filteredchunk,hilbertchunk,ThresholdStrong, ThresholdWeak):
  #  print "Printing globals", globals()
    debug_fd = GlobalVariables['debug_fd']
    
    samplingrate= Parameters['SAMPLERATE']
 #   samplingrate= 20000 
  
#  multdetection_times_ms = Parameters['OBSERVATION_TIMES']
  #  multdetection_times =  np.array(multdetection_times_ms, dtype=np.int32)
  #  multdetection_times = multdetection_times*samplingrate/1000
  #  multdetection_times = multdetection_times.astype(int)

    multdetection_times = Parameters['OBSERVATION_TIMES_SAMPLES']

    chunk_size_less= Parameters['CHUNK_SIZE']-200
#-Parameters['CHUNK_OVERLAP']
#    print 'Parameters: \n', Parameters
   # probefilename = Parameters['PROBE_FILE']
#    print 'chunk_size_less = ', chunk_size_less
#    window_width = 120
#    samples_backward = 60
    window_width = 140
    samples_backward = 70

 # path='/home/skadir/alignment/'
    path = Parameters['OUTPUT_DIR']
    for interestpoint in multdetection_times:
   # for interestpoint_ms in multdetection_times_ms:
    #    interestpoint = int(interestpoint_ms*samplingrate/1000)
  
 #      pp = PdfPages('/home/skadir/alignment/multipagegraphs.pdf')
        if (interestpoint - chunk_size_less) <= s_start < (interestpoint):
            #print interestpoint_ms, ':\n'
            #debug_fd.write(str(interestpoint_ms)+':\n')
            print interestpoint, ':\n'
            debug_fd.write(str(interestpoint)+':\n')
             # sampmin = interestpoint - s_start - 3
            sampmin = np.amax([0,interestpoint - s_start - samples_backward])
            sampmax = sampmin + window_width 
            print 'sampmin, sampmaz ',sampmin, sampmax
           # figdat= plt.figure()
          #  plt.figure()
          #   axdat = figdat.add_subplot(111) # stupid axis object (trivial subplot)
          #   axdat.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
           #  plt.savefig('%s datchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
            
         #    plt.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
         #    plt.savefig('%s datchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
         #    plt.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
         #    plt.savefig('%s filteredchunk_%s.pdf'%(path,interestpoint))

         #    plt.figure()
         #    plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
          #   plt.savefig('%s binarychunk_%s.pdf'%(path,interestpoint))
         #   plt.rcParams('figure.subplot.hspace') = 0.3
      #      embed() # Start IPython

            # recover connected componentslist for our range of interest
            #e.g.indlist = [(19634, 9), (19635, 9), (19635, 11), (19636, 11)], want to know if any 
            #of the sample components are between sampmin and sampmax.
            
            connected_comp_enum = np.zeros_like(binarychunk)
            j = 0
            debugnextbits = []
            for k,indlist in enumerate(indlistchunk):
                indtemparray = np.array(indlist)
                #print k,':',indlist, '\n'
               # print '\n'
               # j = j+1
               # connected_comp_enum[indtemparray[:,0],indtemparray[:,1]] = j
                
               # debug_fd.write(str(k)+': '+'\n')
               # debug_fd.write(str(indlist)+'\n')
               # debug_fd.write('\n') 
               # debug_fd.flush()   
                if (set(indtemparray[:,0]).intersection(np.arange(sampmin,sampmax+1)) != set()):
                    
                    print k,':',indlist, '\n'
                    print '\n'
                    j = j+1
                    connected_comp_enum[indtemparray[:,0],indtemparray[:,1]] = j
                    
                    debug_fd.write(str(k)+': '+'\n')
                    debug_fd.write(str(indlist)+'\n')
                    debug_fd.write('\n') 
                    debug_fd.flush()       # makes sure everything is written to the debug file as program proceeds
                    
                    
                    N_CH = Parameters['N_CH']
                    
                    S_BEFORE = Parameters['S_BEFORE']
                    S_AFTER = Parameters['S_AFTER']
                    if Parameters['DETECT_POSITIVE']:
                        wave, s_peak, sf_peak, cm, fcm, comp_normalised, comp_normalised_power = extract_wave_twothresholds(indlist, filteredchunk,
                                                    filteredchunk,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start, ThresholdStrong, ThresholdWeak) 
                    else:
                        wave, s_peak, sf_peak, cm, fcm, comp_normalised, comp_normalised_power = extract_wave_twothresholds(indlist, filteredchunk,
                                                    -filteredchunk,
                                                    S_BEFORE, S_AFTER, N_CH,
                                                    s_start, ThresholdStrong, ThresholdWeak)
                    #embed()                                
                    debugnextbits.append((s_peak, sf_peak))
                    print 'debugnextbits =', debugnextbits
                    debug_fd.write('debugnextbits ='+ str(debugnextbits)+'\n')
                    debug_fd.flush()  
                    #embed()


            plt.figure()
            filtchunk_normalised = np.maximum((filteredchunk - ThresholdWeak) / (ThresholdStrong - ThresholdWeak),0)
            filtchunk_normalised_power = np.power(filtchunk_normalised,Parameters['WEIGHT_POWER'])
            
            print 'plotting figure now'
            
            #plt.suptitle('%s \n with %s \n Time %s samples'%(Parameters['RAW_DATA_FILES'],Parameters['PROBE_FILE'],interestpoint), fontsize=10, fontweight='bold')
           # plt.suptitle('Time %s ms'%(interestpoint_ms), fontsize=14, fontweight='bold')
            #plt.subplots_adjust(hspace = 0.5)
            plt.subplots_adjust(hspace = 0.25,left= 0.12, bottom = 0.10, right = 0.90, top = 0.90, wspace = 0.2)
            
            #dataxis = plt.subplot(4,2,1)
            #dataxis.set_title('DatChunks',fontsize=10)
            #imdat = dataxis.imshow(np.transpose(datchunk[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");#plt.colorbar(imdat);
            
            #binaxis = plt.subplot(4,2,3)
            #binaxis.set_title('FilteredChunks',fontsize=10)
            #imbin = binaxis.imshow(np.transpose(filteredchunk[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");#plt.colorbar(imbin);
            
            #filaxis = plt.subplot(4,2,2)
            filaxis = plt.subplot(3,1,2)
            #filaxis.set_title('BinChunks',fontsize=10)
            imfil = filaxis.imshow(np.transpose(binarychunk[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");
            plt.ylabel('Channels')#plt.colorbar(imfil);
            
            #conaxis = plt.subplot(4,2,8)
            conaxis = plt.subplot(3,1,3)
            #conaxis.set_title('Connected Components',fontsize=10)
            imcon = conaxis.imshow(np.transpose(connected_comp_enum[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");#plt.colorbar(imcon);
            plt.xlabel('Samples')
            plt.ylabel('Channels')
            for spiketimedebug in debugnextbits:
                conaxis.axvline(spiketimedebug[1]-sampmin,color = 'w') #plot a vertical line for s_fpeak
                print spiketimedebug[1]-sampmin
            
            #constrongaxis = plt.subplot(4,2,4)
            #constrongaxis.set_title('Strong Connected Components',fontsize=10)
            #imconstrong = constrongaxis.imshow(np.transpose(binarychunkstrong[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");#plt.colorbar(imconstrong);
            
            ##compoweraxis = plt.subplot(4,2,7)
            ##compoweraxis.set_title('power weight',fontsize=10)
            ##imcompower = compoweraxis.imshow(np.transpose(filtchunk_normalised_power[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imcompower);
            ##for spiketimedebug in debugnextbits:
            ##    compoweraxis.set_title('comp_normalised_power',fontsize=10)
            ##imconstrong = compoweraxis.imshow(np.transpose(binarychunkstrong[sampmin:sampmax,:]),interpolation="nearest");plt.colorbar(imconstrong);
            
            #conweakaxis = plt.subplot(4,2,6)
           # conweakaxis.set_title('Weak Connected Components',fontsize=10)
           # imconweak = conweakaxis.imshow(np.transpose(binarychunkweak[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");#plt.colorbar(imconweak);
          # # filaxis.set_title('SDChunks',fontsize=10)
          ## imfil = filaxis.imshow(filteredchunk[sampmin:sampmax,:]/(-threshold[:]),interpolation="nearest");plt.colorbar(imfil);
            
            #hilaxis = plt.subplot(4,2,5)
            hilaxis = plt.subplot(3,1,1)
            #hilaxis.set_title('ManipulatedChunks, Hilbert: %s'%(Parameters['USE_HILBERT']),fontsize=10)
            #hilaxis.set_title('Filtered data', fontsize=10)
            imhil = hilaxis.imshow(np.transpose(hilbertchunk[sampmin:sampmax,:]),interpolation="nearest",aspect="auto");
            plt.ylabel('Channels')#plt.colorbar(imhil);
            #plt.savefig('floodfillchunk_%s.pdf'%(interestpoint_ms))
            #plt.tight_layout()
            plt.show()
            plt.savefig('floodfillchunk_%s_samples.pdf'%(interestpoint))
            tosave = [debugnextbits,binarychunkweak,binarychunkstrong,binarychunk,filteredchunk,datchunk,connected_comp_enum,sampmin,sampmax]
            pickle.dump(tosave,open('savegraphdata_%s.p'%(interestpoint),'wb'))
            
# plt.show()
   #         plt.figure();plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();plt.savefig(pp,format='pdf')
        #     figbin=plt.figure()
       #      plt.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest")
        #     plt.colorbar()
       #      pp.savefig(figbin)
            
        #    figfil= plt.figure()
        #    axfil = figfil.add_subplot(111) # stupid axis object (trivial subplot)
        #    axfil.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar();
        #    plt.savefig('%s filteredchunk_%s.pdf'%(path,interestpoint))
            
           
    #     pp.close()


#def collect_fulldata(s_start,indlistchunk,binarychunk,datchunk,filteredchunk,threshold):
#    indlist = indlist
    


