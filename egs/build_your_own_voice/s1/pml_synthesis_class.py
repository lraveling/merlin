
'''
Copyright(C) 2016 Engineering Department, University of Cambridge, UK.

License
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Author

    Laura Raveling l.raveling@campus.tu-berlin.de 

'''
import numpy as np
import os
import uuid, getpass, errno, warnings
import scipy
from scipy import signal as sig  
import distutils 


class MerlinPmlSyn(object):
    def __init__(self, f=0, fs=0, dftlen=0):
        self.f = f
        self.fs = fs 
        self.dftlen = dftlen
        #set option to configure file!! 

    def makedirs(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def settmppath(self, p):
        global TMPPATH
        self.TMPPATH = p
        print('sigproc: Temporary directory: '+TMPPATH)

    def gentmpfile(self, name, tmppath):
        custom_path = tmppath
        tmpdir = os.path.join(custom_path,getpass.getuser())
        self.makedirs(tmpdir)
        tmpfile = os.path.join(tmpdir,'sigproc.pid%s.%s.%s' % (os.getpid(), str(uuid.uuid4()), name))
        return tmpfile

    def check_executable(self, execname, errstring, exception_if_error=True):
        execpath = distutils.spawn.find_executable(os.path.join(sigproc.BINPATH, execname))
        if exception_if_error:
            if not execpath:
                raise ValueError('sigproc: Cannot find "'+execname+'" executable ('+errstring+').')
            else:
                if not execpath is None: return True
                else:                    return False

    def read_binfile(self, filename, dim=60, dtype=np.float64):
        fid = open(filename, 'rb')
        v_data = np.fromfile(fid, dtype=dtype)
        fid.close()
        if np.mod(v_data.size, dim) != 0:
            raise ValueError('Dimension provided not compatible with file size.')
        m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
        m_data = np.squeeze(m_data)
        return  m_data

    def get_label_type(n_frames_total, label_feature_matrix, discrete_dict, continuous_dict):
    
        lab_idx_rowwise = []

        for row in range(n_frames_total):
            curr_row_bin = label_feature_matrix[row,0:len(discrete_dict)]
            curr_row_cont = label_feature_matrix[0,len(discrete_dict):(len(discrete_dict) + len(continuous_dict))]
            count_bin = 0
            count_cont = 0
            row_idx = {}
            
            for binlab in curr_row_bin:
                if binlab > 0.01:
                    row_idx['bin %d' %count_bin] = count_bin
                count_bin += 1

            for contlab in curr_row_cont:
                if contlab > 0.01:
                    row_idx['cont %d' %count_cont] = count_cont
                count_cont += 1

            lab_idx_rowwise.append(row_idx)

        return lab_idx_rowwise

    def label_lookup_all(lab_idx, discrete_dict, continuous_dict, ph_sel_dict):

        label_type_file = []
        #phoneme_superset = [1, 5, 6, 47, 52, 57, 48, 53, 59]
        phoneme_superset = [0, 4, 5, 46, 51, 56, 47, 52, 58]
        label_string = ph_sel_dict
        for label_row in lab_idx:
            for k, v in label_row.items():
                if k.startswith('bin') and v in phoneme_superset:
                    label_type_file.append(ph_sel_dict[str(v)])
        return(label_type_file)

    def pdd_segments_nested(pdd_list, label_len_list): #hier so versucht zu definieren, dass es alle labels in eine lange liste schreibt und nicht erst in list of list per file, klappt evtl eh nicht 

        pdd_list = pdd_list
        label_len_list = label_len_list 
        pdd_seq_list = []
        
        for file in range(0,len(pdd_list)):
            
            pdd_file = pdd_list[int(file)]
            curr_lab_len = label_len_list[int(file)]
            
            pdd_seqs = []
            
            for item in range(0,len(curr_lab_len)):
                
                curr_item = curr_lab_len[int(item)]
                
                if item ==0:
                    pdd_slice = pdd_file[0:curr_item]
                    pdd_seqs.append(pdd_slice)
                else:
                    prev_item = curr_lab_len[int(item-1)]           
                    pdd_slice = pdd_file[int(prev_item):int(prev_item+curr_item)]
                    pdd_seqs.append(pdd_slice)

            pdd_seq_list.append(pdd_seqs)

        return pdd_seq_list

    def spec2mcep(self, SPEC, alpha, order):
        self.check_executable('mcep', 'You can find it in SPTK (https://sourceforge.net/projects/sp-tk/)')

        self.dftlen = 2*(SPEC.shape[1]-1)

        self.tmpspecfile = self.gentmpfile('spec2mcep.spec')
        self.outspecfile = self.gentmpfile('spec2mcep.mcep')

        try:
            SPEC.astype(np.float32).tofile(tmpspecfile)
            self.cmd = os.path.join(BINPATH,'mcep')+' -a '+str(alpha)+' -m '+str(int(order))+' -l '+str(dftlen)+' -e 1.0E-8 -j 0 -f 0.0 -q 3 '+tmpspecfile+' > '+outspecfile
            self.ret = os.system(cmd)
            if ret>0: raise ValueError('ERROR during execution of mcep')
            self.MCEP = np.fromfile(outspecfile, dtype=np.float32)
            self.MCEP = MCEP.reshape((-1, 1+int(order)))
        except:
            if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
            if os.path.exists(outspecfile): os.remove(outspecfile)
            raise

        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(outspecfile): os.remove(outspecfile)

        return MCEP

    def mcep2spec(self, MCEP, alpha, dftlen, temp_dir, sptk_dir):

        sptk = sptk_dir

        order = 59
        #order = MCEP.shape[1]-1
        TMPPATH = temp_dir
        tmpspecfile = self.gentmpfile('mcep2spec.mcep', temp_dir)
        outspecfile = self.gentmpfile('mcep2spec.spec', temp_dir)

        try:
            MCEP.astype(np.float32).tofile(tmpspecfile)
            cmd = os.path.join(sptk,'mgc2sp')+' -a '+str(alpha)+' -g 0 -m '+str(int(order))+' -l '+str(dftlen)+' -o 2 '+tmpspecfile+' > '+outspecfile
            ret = os.system(cmd)
            if ret>0: raise ValueError('ERROR during execution of mgc2sp')
            SPEC = np.fromfile(outspecfile, dtype=np.float32)
            SPEC = SPEC.reshape((-1, int(dftlen/2+1)))
        except:
            if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
            if os.path.exists(outspecfile): os.remove(outspecfile)
            raise

        if os.path.exists(tmpspecfile): os.remove(tmpspecfile)
        if os.path.exists(outspecfile): os.remove(outspecfile)

        return SPEC


    def bark_alpha(self, fs):
        return 0.8517*np.sqrt(np.arctan(0.06583*fs/1000.0))-0.1916

    def getwinlen(self, f0, fs, nbper):
        return int(np.max((0.05*fs, nbper*fs/f0))/2)*2+1 

    def mag2db(self, a):
        return 20.0*np.log10(np.abs(a)) 

    def f0s_rmsteps(self, f0s):
        f0sori = f0s.copy()
        f0s = f0s.copy()
        voicedi = np.where(f0s[:,1]>0)[0]
        shift = np.mean(np.diff(f0s[:,0]))
        fc = (1.0/shift)/4.0  # The cut-off frequency
        hshift = (1.0/fc)/8.0 # The high sampling rate for resampling the original curve
        data = np.interp(np.arange(0.0, f0s[-1,0], hshift), f0s[voicedi,0], f0s[voicedi,1])
        b, a = sig.butter(8, fc/(0.5/hshift), btype='low')
        f0ss = sig.filtfilt(b, a, data)
        f0s[voicedi,1] = np.interp(f0s[voicedi,0], np.arange(0.0, f0s[-1,0], hshift), f0ss)
        if 0:
            plt.plot(f0sori[:,0], f0sori[:,1], 'k')
            plt.plot(f0s[:,0], f0s[:,1], 'b')
            from IPython.core.debugger import  Pdb; Pdb().set_trace()
        return f0s

    def f0s_resample_cst(f0s, timeshift):
        f0s = f0s.copy()

        vcs = f0s.copy()
        vcs[vcs[:,1]>0,1] = 1

        nts = np.arange(f0s[0,0], f0s[-1,0], timeshift)

        # The voicing resampling has to be done using nearest ...
        vcsfn = scipy.interpolate.interp1d(vcs[:,0], vcs[:,1], kind='nearest', bounds_error=False, fill_value=0)

        # ... whereas the frequency resampling need linear interpolation, while ignoring the voicing
        f0s = np.interp(nts, f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

        # Put back the voicing
        f0s[vcsfn(nts)==0] = 0.0

        f0s = np.vstack((nts, f0s)).T

        if 0:
            plt.plot(f0s[:,0], f0s[:,1])

        return f0s

    def analysis_f0postproc(self, wav, fs, f0s=None, f0_min=60, f0_max=600,
             shift=0.005,        # Usually 5ms
             f0estimator='REAPER',
             verbose=1):
        if f0s is None:
            f0s = sigproc.interfaces.reaper(wav, fs, shift, f0_min, f0_max)

        # If only values are given, make two column matrix [time[s], value[Hz]] (ljuvela)
        if len(f0s.shape)==1:
            ts = (shift)*np.arange(len(f0s))
            f0s = np.vstack((ts, f0s)).T

        if not (f0s[:,1]>0).any():
            warnings.warn('''\n\nWARNING: No F0 value can be estimated in this signal.
             It will be replaced by the constant f0_min value ({}Hz).
            '''.format(f0_min), RuntimeWarning)
            f0s[:,1] = f0_min


        # Build the continuous f0
        f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])
         # Avoid erratic values outside of the given interval
        f0s[:,1] = np.clip(f0s[:,1], f0_min, f0_max)
        # Removes steps in the f0 curve (see sigproc.resampling.f0s_rmsteps(.) )
        f0s = self.f0s_rmsteps(f0s)
        # Resample the given f0 to regular intervals
        if np.std(np.diff(f0s[:,0]))>2*np.finfo(f0s[0,0]).resolution:
            warnings.warn('''\n\nWARNING: F0 curve seems to be sampled non-uniformly (mean(F0)={}, std(F0s')={}).
             It will be resampled at {}s intervals.
            '''.format(np.std(f0s[:,0]), np.std(np.diff(f0s[:,0])), shift), RuntimeWarning)
            f0s = self.f0s_resample_cst(f0s)
        return f0s

    nbperperiod = 4 

    
    def f0s_resample_pitchsync(self, f0s, nbperperiod, f0min=20.0, f0max=5000.0):
        f0s = f0s.copy()

        # Interpolate where there is zero values
        f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

        f0s[:,1] = np.clip(f0s[:,1], f0min, f0max)

        ts = [0.0]
        while ts[-1]<f0s[-1,0]:
            cf0 = np.interp(ts[-1], f0s[:,0], f0s[:,1])
            ts.append(ts[-1]+(1.0/nbperperiod)/cf0)
        
        f0s = np.vstack((ts, np.interp(ts, f0s[:,0], f0s[:,1]))).T
        return f0s

    def butter2hspec(self, fc, o, fs, dftlen, high=False):
        F = fs*np.arange(dftlen/2+1)/dftlen
        H = 1.0/np.sqrt(1.0 + (F/fc)**(2*o))

        if high:
            H = 1.0-H

        return H

    def hspec2minphasehspec(self, X, replacezero=True):
        if replacezero:
            X[X==0.0] = np.finfo(X[0]).resolution
        dftlen = (len(X)-1)*2
        cc = np.fft.irfft(np.log(X))
        cc = cc[:int(dftlen/2+1)] #"bug"?
        cc[1:-1] *= 2
        return np.exp(np.fft.rfft(cc, dftlen))

    def analysis_nm_bin(self, fs,    #removed wav argument- lr
             f0s,                # Has to be continuous (should use analysis_f0postproc)
             PDD,                # Phase Distortion Deviation [2]
                                 # Its length should match f0s'
             pdd_threshold=0.75, # 0.75 as in [2]
             nm_clean=True,      # Use morphological opening and closure to
                                 # clean the mask and avoid learning rubish.
             verbose=1):
        
        if f0s.shape[0]!=PDD.shape[0]:
            raise ValueError('f0s size and PDD size do not match!') # pragma: no cover

        shift = np.mean(np.diff(f0s[:,0])) # Get the time shift from the F0 times
        dftlen = (PDD.shape[1]-1)*2 # and the DFT len from the PDD feature

        # The Noise Mask is just a thresholded version of PDD
        HARM = PDD.copy()
        HARM[PDD<=pdd_threshold] = 0
        HARM[PDD>pdd_threshold] = 1

        if nm_clean:
            # Clean the PDD mask to avoid learning rubish details
            import scipy.ndimage
            frq = 70.0 # [Hz]
            morphstruct = np.ones((int(np.round((1.0/frq)/shift)),int(np.round(frq*dftlen/float(fs)))))
            HARM = 1.0-HARM
            HARM = scipy.ndimage.binary_opening(HARM, structure=morphstruct)
            HARM = scipy.ndimage.binary_closing(HARM, structure=morphstruct)
            HARM = 1.0-HARM

        # Avoid noise in low-freqs
        for n in range(len(f0s[:,0])):
            HARM[n,:int(np.round(1.5*f0s[n,1]*dftlen/float(fs)))] = 0.0

        NM = HARM
        return NM

    def analysis_nm_steps(self, fs, f0s, file_id, harm_lab, dftlen):

        HARM_LAB = harm_lab
        HARM_NEW = []
        for labels in HARM_LAB:

            if labels[0] == 'Vowel':
                noise_val = 0.230
                HARM_SEG_Z = labels[1]
                mean = 0.708
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Liquid':
                noise_val = 0.229
                HARM_SEG_Z = labels[1]
                mean = 0.710
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Nasal':
                noise_val = 0.263
                HARM_SEG_Z = labels[1]
                mean = 0.716
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Voiced_Stop':
                noise_val = 0.252
                HARM_SEG_Z = labels[1]
                mean = 0.727
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Voiced_Fricative':
                noise_val = 0.234
                HARM_SEG_Z = labels[1]
                mean = 0.726
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Affricate_Consonant':
                noise_val = 0.356
                HARM_SEG_Z = labels[1]
                mean = 0.692
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Unvoiced_Stop':
                noise_val = 0.277
                HARM_SEG_Z = labels[1]
                mean = 0.701
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Unvoiced_Fricative':
                noise_val = 0.302
                HARM_SEG_Z = labels[1]
                mean = 0.692
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

            elif labels[0] == 'Silence':
                noise_val = 0.252
                HARM_SEG_Z = labels[1]
                mean = 0.689
                HARM_SEG_Z[HARM_SEG_Z < 0.35] = 0
                HARM_SEG_Z[HARM_SEG_Z >= 0.35] = 0.71 + noise_val
                HARM_NEW.append(HARM_SEG_Z)

        #import scipy.ndimage
        #frq = 70.0
        #shift = np.mean(np.diff(f0s[:,0]))
        #mean_all = (0.708 + 0.710 + 0.716 + 0.727 + 0.726 + 0.692 + 0.701 + 0.692 + 0.689)/9
        dftlen = dftlen
        #morphstruct = np.ones((int(np.round((frq)/shift)),int(np.round(frq*dftlen/float(fs)))))
        HARM_NEW = np.asarray(HARM_NEW)
        #HARM_NEW = 1.0-HARM_NEW
        #HARM_NEW = scipy.ndimage.grey_opening(HARM_NEW)
        #HARM_NEW = scipy.ndimage.grey_closing(HARM_NEW)
        #HARM_NEW = 1.0-HARM_NEW
        
        for n in range(len(f0s[:,0])):
            HARM_NEW[n,:int(np.round(1.5*f0s[n,1]*dftlen/float(fs)))] = 0.0

        return HARM_NEW

    def synthesize(self, fs, f0s, SPEC, NM=None, wavlen=None
                , ener_multT0=False
                , nm_cont=False     # If False, force binary state of the noise mask (by thresholding at 0.5)
                , nm_lowpasswinlen=9
                , hp_f0coef=0.5     # factor of f0 for the cut-off of the high-pass filter (def. 0.5*f0)
                , antipreechohwindur=0.001 # [s] Use to damp the signal at the beginning of the signal AND at the end of it
                # Following options are for post-processing the features, after the generation/transformation and thus before waveform synthesis
                , pp_f0_rmsteps=False # Removes steps in the f0 curve
                                      # (see sigproc.resampling.f0s_rmsteps(.) )
                , pp_f0_smooth=None   # Smooth the f0 curve using median and FIR filters of given window duration [s]
                , pp_atten1stharminsilences=None # Typical value is -25
                , verbose=1):

        winnbper = 4    # Number of periods in a synthesis windows. It still contains only one single pulse, but leaves space for the VTF to decay without being cut abruptly.

        # Copy the inputs to avoid modifying them
        f0s = f0s.copy()
        SPEC = SPEC.copy()
        if not NM is None: NM = NM.copy()
        else:              NM = np.zeros(SPEC.shape)

        NM = np.clip(NM, 0.0, 1.0)  # The noise mask is supposed to be in [0,1]

        # Check the size of the inputs
        if f0s.shape[0]!=SPEC.shape[0]:
            raise ValueError('F0 size {} and spectrogram size {} do not match'.format(f0s.shape[0], SPEC.shape[0])) # pragma: no cover
        if not NM is None:
            if SPEC.shape!=NM.shape:
                raise ValueError('spectrogram size {} and NM size {} do not match.'.format(SPEC.shape, NM.shape)) # pragma: no cover

        if wavlen==None: wavlen = int(np.round(f0s[-1,0]*fs))
        dftlen = (SPEC.shape[1]-1)*2
        shift = np.median(np.diff(f0s[:,0]))
        if verbose>0:
            print('PML Synthesis (dur={}s, fs={}Hz, f0 in [{:.0f},{:.0f}]Hz, shift={}s, dftlen={})'.format(wavlen/float(fs), fs, np.min(f0s[:,1]), np.max(f0s[:,1]), shift, dftlen))

        # Prepare the features

        # Enforce continuous f0
        f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])
        # If asked, removes steps in the f0 curve
        if pp_f0_rmsteps:
            f0s = self.f0s_rmsteps(f0s)
        # If asked, smooth the f0 curve using median and FIR filters
        if not pp_f0_smooth is None:
            print('    Smoothing f0 curve using {}[s] window'.format(pp_f0_smooth))
            import scipy.signal as sig
            lf0 = np.log(f0s[:,1])
            bcoefslen = int(0.5*pp_f0_smooth/shift)*2+1
            lf0 = sig.medfilt(lf0, bcoefslen)
            bcoefs = np.hamming(bcoefslen)
            bcoefs = bcoefs/sum(bcoefs)
            lf0 = sig.filtfilt(bcoefs, [1], lf0)
            f0s[:,1] = np.exp(lf0)

        winlenmax = self.getwinlen(np.min(f0s[:,1]), fs, winnbper)
        if winlenmax>dftlen:
            warnings.warn('\n\nWARNING: The maximum window length ({}) is bigger than the DFT length ({}). Please, increase the DFT length of your spectral features (the second dimension) or check if the f0 curve has extremly low values and try to clip them to higher values (at least higher than 50Hz). The f0 curve has been clipped to {}Hz.\n\n'.format(winlenmax, dftlen, winnbper*fs/float(dftlen))) # pragma: no cover
            f0s[:,1] = np.clip(f0s[:,1], winnbper*fs/float(dftlen-2), 1e6)

        if not NM is None:
            # Remove noise below f0, as it is supposed to be already the case
            for n in range(NM.shape[0]):
                NM[n,:int((float(dftlen)/fs)*2*f0s[n,1])] = 0.0

        if not nm_cont:
            print('    Forcing binary noise mask')
            NM[NM<=0.5] = 0.0 # To be sure that voiced segments are not hoarse
            NM[NM>0.5] = 1.0  # To be sure the noise segments are fully noisy

        # Generate the pulse positions [1](2) (i.e. the synthesis instants, the GCIs in voiced segments)
        ts = [0.0]
        while ts[-1]<float(wavlen)/fs:
            cf0 = np.interp(ts[-1], f0s[:,0], f0s[:,1])
            if cf0<50.0: cf0 = 50
            ts.append(ts[-1]+(1.0/cf0))
        ts = np.array(ts)
        f0s = np.vstack((ts, np.interp(ts, f0s[:,0], f0s[:,1]))).T


        # Resample the features to the pulse positions

        # Spectral envelope uses the nearest, to avoid over-smoothing
        SPECR = np.zeros((f0s.shape[0], int(dftlen/2+1)))
        for n, t in enumerate(f0s[:,0]): # Nearest: Way better for plosives
            idx = int(np.round(t/shift))
            idx = np.clip(idx, 0, SPEC.shape[0]-1)
            SPECR[n,:] = SPEC[idx,:]

        # Keep trace of the median energy [dB] over the whole signal
        ener = np.mean(SPECR, axis=1)
        idxacs = np.where(self.mag2db(ener) > self.mag2db(np.max(ener))-30)[0] # Get approx active frames # TODO Param
        enermed = self.mag2db(np.median(ener[idxacs])) # Median energy [dB]
        ener = self.mag2db(ener)

        # Resample the noise feature to the pulse positions
        # Smooth the frequency response of the mask in order to avoid Gibbs
        # (poor Gibbs nobody want to see him)
        nm_lowpasswin = np.hanning(nm_lowpasswinlen)
        nm_lowpasswin /= np.sum(nm_lowpasswin)
        NMR = np.zeros((f0s.shape[0], int(dftlen/2+1)))
        for n, t in enumerate(f0s[:,0]):
            idx = int(np.round(t/shift)) # Nearest is better for plosives
            idx = np.clip(idx, 0, NM.shape[0]-1)
            NMR[n,:] = NM[idx,:]
            if nm_lowpasswinlen>1:
                NMR[n,:] = scipy.signal.filtfilt(nm_lowpasswin, [1.0], NMR[n,:])

        NMR = np.clip(NMR, 0.0, 1.0)

        # The complete waveform that we will fill with the pulses
        wav = np.zeros(wavlen)
        # Half window on the left of the synthesized segment to avoid pre-echo
        dampinhwin = np.hanning(1+2*int(np.round(antipreechohwindur*fs))) # 1ms forced dampingwindow
        dampinwinidx = int((len(dampinhwin)-1)/2+1)
        dampinhwin = dampinhwin[:dampinwinidx]

        for n, t in enumerate(f0s[:,0]):
            f0 = f0s[n,1]
            # Window's length
            # TODO It should be ensured that the beggining and end of the
            #      noise is within the window. Nothing is doing this currently!
            winlen = self.getwinlen(f0, fs, winnbper)
            # TODO We also assume that the VTF's decay is shorter
            #      than winnbper-1 periods (dangerous with high pitched and tense voice).
            if winlen>dftlen: raise ValueError('The window length ({}) is bigger than the DFT length ({}). Please, increase the dftlen of your spectral features or check if the f0 curve has extremly low values and try to clip them to higher values (at least higher than 50[Hz])'.format(winlen, dftlen)) # pragma: no cover

            # Set the rough position of the pulse in the window (the closest sample)
            # We keep a third of the window (1 period) on the left because the
            # pulse signal is minimum phase. And 2/3rd (remaining 2 periods)
            # on the right to let the VTF decay.
            pulseposinwin = int((1.0/winnbper)*winlen)

            # The sample indices of the current pulse wrt. the final waveform
            winidx = int(round(fs*t)) + np.arange(winlen)-pulseposinwin


            # Build the pulse spectrum

            # Let start with a Dirac
            S = np.ones(int(dftlen/2+1), dtype=np.complex64)

            # Add the delay to place the Dirac at the "GCI": exp(-j*2*pi*t_i)
            delay = -pulseposinwin - fs*(t-int(round(fs*t))/float(fs))
            S *= np.exp((delay*2j*np.pi/dftlen)*np.arange(dftlen/2+1))

            # Add the spectral envelope
            # Both amplitude and phase
            E = SPECR[n,:] # Take the amplitude from the given one
            if hp_f0coef!=None:
                # High-pass it to avoid any residual DC component.
                fcut = hp_f0coef*f0
                if not pp_atten1stharminsilences is None and ener[n]-enermed<pp_atten1stharminsilences:
                    fcut = 1.5*f0 # Try to cut between first and second harm
                HP = self.butter2hspec(fcut, 4, fs, dftlen, high=True)
                E *= HP
                # Not necessarily good as it is non-causal, so make it causal...
                # ... together with the VTF response below.
            # Build the phase of the envelope from the amplitude
            E = self.hspec2minphasehspec(E, replacezero=True) # We spend 2 FFT here!
            S *= E # Add it to the current pulse

            # Add energy correction wrt f0.
            # STRAIGHT and AHOCODER vocoders do it.
            # (why ? to equalize the energy when changing the pulse's duration ?)
            if ener_multT0:
                S *= np.sqrt(fs/f0)

            # Generate the segment of Gaussian noise
            # Use mid-points before/after pulse position
            if n>0: leftbnd=int(np.round(fs*0.5*(f0s[n-1,0]+t)))
            else:   leftbnd=int(np.round(fs*(t-0.5/f0s[n,1]))) # int(0)
            if n<f0s.shape[0]-1: rightbnd=int(np.round(fs*0.5*(t+f0s[n+1,0])))-1
            else:                rightbnd=int(np.round(fs*(t+0.5/f0s[n,1])))   #rightbnd=int(wavlen-1)
            gausswinlen = rightbnd-leftbnd # The length of the noise segment
            gaussnoise4win = np.random.normal(size=(gausswinlen)) # The noise

            GN = np.fft.rfft(gaussnoise4win, dftlen) # Move the noise to freq domain
            # Normalize it by its energy (@Yannis, That's your answer at SSW9!)
            GN /= np.sqrt(np.mean(np.abs(GN)**2))
            # Place the noise within the pulse's window
            delay = (pulseposinwin-(leftbnd-winidx[0]))
            GN *= np.exp((delay*2j*np.pi/dftlen)*np.arange(dftlen/2+1))

            # Add it to the pulse spectrum, under the condition of the mask
            S *= GN**NMR[n,:]

            # That's it! the pulse spectrum is ready!

            # Move it to time domain
            deter = np.fft.irfft(S)[0:winlen]

            # Add half window on the left of the synthesized segment
            # to avoid any possible pre-echo
            deter[:leftbnd-winidx[0]-len(dampinhwin)] = 0.0
            deter[leftbnd-winidx[0]-len(dampinhwin):leftbnd-winidx[0]] *= dampinhwin

            # Add half window on the right
            # to avoid cutting the VTF response abruptly
            deter[-len(dampinhwin):] *= dampinhwin[::-1]

            # Write the synthesized segment in the final waveform
            if winidx[0]<0 or winidx[-1]>=wavlen:
                # The window is partly outside of the waveform ...
                # ... thus copy only the existing part
                itouse = np.logical_and(winidx>=0,winidx<wavlen)
                wav[winidx[itouse]] += deter[itouse]
            else:
                wav[winidx] += deter

        if verbose>1:
            print('\r                                                               \r'),

        if verbose>2:                                             # pragma: no cover
            import matplotlib.pyplot as plt
            plt.ion()
            _, axs = plt.subplots(3, 1, sharex=True, sharey=False)
            times = np.arange(len(wav))/float(fs)
            axs[0].plot(times, wav, 'k')
            axs[0].set_ylabel('Waveform\nAmplitude')
            axs[0].grid()
            axs[1].plot(f0s[:,0], f0s[:,1], 'k')
            axs[1].set_ylabel('F0\nFrequency [Hz]')
            axs[1].grid()
            axs[2].imshow(self.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', extent=(f0s[0,0], f0s[-1,0], 0, 0.5*fs))
            axs[2].set_ylabel('Amp. Envelope\nFrequency [Hz]')

            from IPython.core.debugger import  Pdb; Pdb().set_trace()

        return wav






