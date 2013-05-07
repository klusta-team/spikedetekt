import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from parameters import Parameters, GlobalVariables


print "Printing globals", globals()
multdetection_times_ms = [5, 52, 119, 204, 208, 232, 241, 251,28695,28742,28831,28855,66906,167264,214241,334526,402506,402508]
multdetection_times =  np.array(multdetection_times_ms, dtype=np.int32)
 # samplingrate= Parameters['SAMPLERATE']
samplingrate= 20000 
multdetection_times = multdetection_times*samplingrate/1000
multdetection_times = multdetection_times.astype(int)
chunk_size_less= Parameters['CHUNK_SIZE']-200
#-Parameters['CHUNK_OVERLAP']
print 'Parameters: \n', Parameters
probefilename = Parameters['PROBE_FILE']
print 'chunk_size_less = ', chunk_size_less
samples_forward = 30

 # path='/home/skadir/alignment/'
path = Parameters['OUTPUT_DIR']

def plot_diagnostics(s_start,binarychunk,datchunk,filteredchunk,threshold):
   #  for interestpoint in multdetection_times:
    for interestpoint_ms in multdetection_times_ms:
        interestpoint = int(interestpoint_ms*samplingrate/1000)
   #      pp = PdfPages('/home/skadir/alignment/multipagegraphs.pdf')
        if (interestpoint - chunk_size_less) <= s_start < (interestpoint):
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
            plt.savefig('%s %s_floodfillchunk_%s.pdf'%(path,probefilename,interestpoint_ms))
            
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
