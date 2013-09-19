import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from parameters import Parameters, GlobalVariables
from IPython import embed # For manual debugging

#  multdetection_times_ms = [5, 52, 119, 204, 208, 232, 241, 251,28695,28742,28831,28855,66906,167264,214241,334526,402506,402508]
multdetection_times_ms = [2, 1000]


def plot_diagnostics(s_start,indlistchunk,binarychunk,datchunk,filteredchunk,threshold):
  #  print "Printing globals", globals()
    debug_fd = GlobalVariables['debug_fd']
    multdetection_times =  np.array(multdetection_times_ms, dtype=np.int32)
    samplingrate= Parameters['SAMPLERATE']
 #   samplingrate= 20000 
    multdetection_times = multdetection_times*samplingrate/1000
    multdetection_times = multdetection_times.astype(int)
    chunk_size_less= Parameters['CHUNK_SIZE']-200
#-Parameters['CHUNK_OVERLAP']
#    print 'Parameters: \n', Parameters
   # probefilename = Parameters['PROBE_FILE']
#    print 'chunk_size_less = ', chunk_size_less
    samples_forward = 25

 # path='/home/skadir/alignment/'
    path = Parameters['OUTPUT_DIR']
   #  for interestpoint in multdetection_times:
    for interestpoint_ms in multdetection_times_ms:
        interestpoint = int(interestpoint_ms*samplingrate/1000)
   #      pp = PdfPages('/home/skadir/alignment/multipagegraphs.pdf')
        if (interestpoint - chunk_size_less) <= s_start < (interestpoint):
            print interestpoint_ms, ':\n'
            debug_fd.write(str(interestpoint_ms)+':\n')
            sampmin = interestpoint - s_start - 3
            sampmax = sampmin + samples_forward 
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
            
            for indlist in indlistchunk:
                indtemparray = np.array(indlist)
                if (set(indtemparray[:,0]).intersection(np.arange(sampmin,sampmax+1)) != set()):
                    
                    print indlist, '\n'
                    print '\n'
                    
                    debug_fd.write(str(indlist)+'\n')
                    debug_fd.write('\n') 
                    debug_fd.flush()       # makes sure everything is written to the debug file as program proceeds 

            plt.figure()
            plt.suptitle('Time %s ms'%(interestpoint_ms), fontsize=14, fontweight='bold')
            plt.subplots_adjust(hspace = 0.5)
            dataxis = plt.subplot(4,1,1)
            dataxis.set_title('DatChunks',fontsize=10)
            imdat = dataxis.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imdat);
            binaxis = plt.subplot(4,1,2)
            binaxis.set_title('FilteredChunks',fontsize=10)
            imbin = binaxis.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imbin);
            filaxis = plt.subplot(4,1,3)
            filaxis.set_title('BinChunks',fontsize=10)
            imfil = filaxis.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imfil);
            filaxis = plt.subplot(4,1,4)
            filaxis.set_title('SDChunks',fontsize=10)
            imfil = filaxis.imshow(filteredchunk[sampmin:sampmax,:]/(-threshold[:]),interpolation="nearest");plt.colorbar(imfil);
            plt.savefig('%s_floodfillchunk_%s.pdf'%(path,interestpoint_ms))
            
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
def plot_diagnostics_twothresholds(s_start,indlistchunk,binarychunk,datchunk,filteredchunk,hilbertchunk,ThresholdStrong, ThresholdWeak):
  #  print "Printing globals", globals()
    debug_fd = GlobalVariables['debug_fd']
    multdetection_times =  np.array(multdetection_times_ms, dtype=np.int32)
    samplingrate= Parameters['SAMPLERATE']
 #   samplingrate= 20000 
    multdetection_times = multdetection_times*samplingrate/1000
    multdetection_times = multdetection_times.astype(int)
    chunk_size_less= Parameters['CHUNK_SIZE']-200
#-Parameters['CHUNK_OVERLAP']
#    print 'Parameters: \n', Parameters
   # probefilename = Parameters['PROBE_FILE']
#    print 'chunk_size_less = ', chunk_size_less
    samples_forward = 25

 # path='/home/skadir/alignment/'
    path = Parameters['OUTPUT_DIR']
   #  for interestpoint in multdetection_times:
    for interestpoint_ms in multdetection_times_ms:
        interestpoint = int(interestpoint_ms*samplingrate/1000)
   #      pp = PdfPages('/home/skadir/alignment/multipagegraphs.pdf')
        if (interestpoint - chunk_size_less) <= s_start < (interestpoint):
            print interestpoint_ms, ':\n'
            debug_fd.write(str(interestpoint_ms)+':\n')
            sampmin = interestpoint - s_start - 3
            sampmax = sampmin + samples_forward 
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
            
            for indlist in indlistchunk:
                indtemparray = np.array(indlist)
                if (set(indtemparray[:,0]).intersection(np.arange(sampmin,sampmax+1)) != set()):
                    
                    print indlist, '\n'
                    print '\n'
                    
                    debug_fd.write(str(indlist)+'\n')
                    debug_fd.write('\n') 
                    debug_fd.flush()       # makes sure everything is written to the debug file as program proceeds 

            plt.figure()
            plt.suptitle('Time %s ms'%(interestpoint_ms), fontsize=14, fontweight='bold')
            plt.subplots_adjust(hspace = 0.5)
            dataxis = plt.subplot(4,1,1)
            dataxis.set_title('DatChunks',fontsize=10)
            imdat = dataxis.imshow(datchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imdat);
            binaxis = plt.subplot(4,1,2)
            binaxis.set_title('FilteredChunks',fontsize=10)
            imbin = binaxis.imshow(filteredchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imbin);
            filaxis = plt.subplot(4,1,3)
            filaxis.set_title('BinChunks',fontsize=10)
            imfil = filaxis.imshow(binarychunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imfil);
            filaxis = plt.subplot(4,1,4)
           # filaxis.set_title('SDChunks',fontsize=10)
           # imfil = filaxis.imshow(filteredchunk[sampmin:sampmax,:]/(-threshold[:]),interpolation="nearest");plt.colorbar(imfil);
           # filaxis = plt.subplot(5,1,5)
            filaxis.set_title('HilbertChunks',fontsize=10)
            imfil = filaxis.imshow(hilbertchunk[sampmin:sampmax,:],interpolation="nearest");plt.colorbar(imfil);
            plt.savefig('%s_floodfillchunk_%s.pdf'%(path,interestpoint_ms))
            
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
    


