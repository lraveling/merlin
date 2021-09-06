################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

#/usr/bin/python -u

'''
This script assumes c-version STRAIGHT which is not available to public. Please use your
own vocoder to replace this script.
'''
import sys, os, subprocess, glob, subprocess
#from utils import GlobalCfg

from io_funcs.binary_io import  BinaryIOCollection
import numpy as np
import scipy
from scipy.io import wavfile

import logging

#import configuration

# cannot have these outside a function - if you do that, they get executed as soon
# as this file is imported, but that can happen before the configuration is set up properly
# SPTK     = cfg.SPTK
# NND      = cfg.NND
# STRAIGHT = cfg.STRAIGHT


def run_process(args,log=True):

    logger = logging.getLogger("subprocess")

    # a convenience function instead of calling subprocess directly
    # this is so that we can do some logging and catch exceptions

    # we don't always want debug logging, even when logging level is DEBUG
    # especially if calling a lot of external functions
    # so we can disable it by force, where necessary
    if log:
        logger.debug('%s' % args)

    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)

        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True, env=os.environ)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()

        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            logger.critical('exit status %d' % p.returncode )
            logger.critical(' for command: %s' % args )
            logger.critical('      stderr: %s' % stderrdata )
            logger.critical('      stdout: %s' % stdoutdata )
            raise OSError

        return (stdoutdata, stderrdata)

    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        logger.critical('exit status %d' % e.returncode )
        logger.critical(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        logger.critical('  output: %s' % e.output )
        raise

    except ValueError:
        logger.critical('ValueError for %s' % args )
        raise

    except OSError:
        logger.critical('OSError for %s' % args )
        raise

    except KeyboardInterrupt:
        logger.critical('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass

        raise KeyboardInterrupt


def bark_alpha(sr):
    return 0.8517*np.sqrt(np.arctan(0.06583*sr/1000.0))-0.1916

def erb_alpha(sr):
    return 0.5941*np.sqrt(np.arctan(0.1418*sr/1000.0))+0.03237

def post_filter(mgc_file_in, mgc_file_out, mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir, cfg):

    SPTK = cfg.SPTK

    line = "echo 1 1 "
    for i in range(2, mgc_dim):
        line = line + str(pf_coef) + " "

    run_process('{line} | {x2x} +af > {weight}'
                .format(line=line, x2x=SPTK['X2X'], weight=os.path.join(gen_dir, 'weight')))

    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                .format(freqt=SPTK['FREQT'], order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=mgc_file_in, c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=mgc_file_in+'_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef,
                        c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=mgc_file_in+'_p_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'], fw=fw_coef,
                        bcp=SPTK['BCP'], base_b0=mgc_file_in+'_b0'))

    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr=SPTK['VOPR'], base_r0=mgc_file_in+'_r0', base_p_r0=mgc_file_in+'_p_r0',
                        sopr=SPTK['SOPR'],
                        base_b0=mgc_file_in+'_b0', base_p_b0=mgc_file_in+'_p_b0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'],  fw=fw_coef,
                        bcp=SPTK['BCP'],
                        merge=SPTK['MERGE'], order2=mgc_dim-2, base_p_b0=mgc_file_in+'_p_b0',
                        b2mc=SPTK['B2MC'], base_p_mgc=mgc_file_out))

    return

def wavgen_straight_type_vocoder(gen_dir, file_id_list, cfg, logger):
    '''
    Waveform generation with STRAIGHT or WORLD vocoders.
    (whose acoustic parameters are: mgc, bap, and lf0)
    '''

    SPTK     = cfg.SPTK
#    NND      = cfg.NND
    STRAIGHT = cfg.STRAIGHT
    WORLD    = cfg.WORLD

    ## to be moved
    pf_coef = cfg.pf_coef
    if isinstance(cfg.fw_alpha, str):
        if cfg.fw_alpha=='Bark':
            fw_coef = bark_alpha(cfg.sr)
        elif cfg.fw_alpha=='ERB':
            fw_coef = bark_alpha(cfg.sr)
        else:
            raise ValueError('cfg.fw_alpha='+cfg.fw_alpha+' not implemented, the frequency warping coefficient "fw_coef" cannot be deduced.')
    else:
        fw_coef = cfg.fw_alpha
    co_coef = cfg.co_coef
    fl_coef = cfg.fl

    if cfg.apply_GV:
        io_funcs = BinaryIOCollection()

        logger.info('loading global variance stats from %s' % (cfg.GV_dir))

        ref_gv_mean_file = os.path.join(cfg.GV_dir, 'ref_gv.mean')
        gen_gv_mean_file = os.path.join(cfg.GV_dir, 'gen_gv.mean')
        ref_gv_std_file  = os.path.join(cfg.GV_dir, 'ref_gv.std')
        gen_gv_std_file  = os.path.join(cfg.GV_dir, 'gen_gv.std')

        ref_gv_mean, frame_number = io_funcs.load_binary_file_frame(ref_gv_mean_file, 1)
        gen_gv_mean, frame_number = io_funcs.load_binary_file_frame(gen_gv_mean_file, 1)
        ref_gv_std, frame_number = io_funcs.load_binary_file_frame(ref_gv_std_file, 1)
        gen_gv_std, frame_number = io_funcs.load_binary_file_frame(gen_gv_std_file, 1)

    counter=1
    max_counter = len(file_id_list)

    for filename in file_id_list:

        logger.info('creating waveform for %4d of %4d: %s' % (counter,max_counter,filename) )
        counter=counter+1
        base   = filename
        files = {'sp'  : base + cfg.sp_ext,
                 'mgc' : base + cfg.mgc_ext,
                 'f0'  : base + '.f0',
                 'lf0' : base + cfg.lf0_ext,
                 'ap'  : base + '.ap',
                 'bap' : base + cfg.bap_ext,
                 'wav' : base + '.wav'}

        mgc_file_name = files['mgc']
        bap_file_name = files['bap']

        cur_dir = os.getcwd()
        os.chdir(gen_dir)

        ### post-filtering
        if cfg.do_post_filtering:

            mgc_file_name = files['mgc']+'_p_mgc'
            post_filter(files['mgc'], mgc_file_name, cfg.mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir, cfg)

        if cfg.vocoder_type == "STRAIGHT" and cfg.apply_GV:
            gen_mgc, frame_number = io_funcs.load_binary_file_frame(mgc_file_name, cfg.mgc_dim)

            gen_mu  = np.reshape(np.mean(gen_mgc, axis=0), (-1, 1))
            gen_std = np.reshape(np.std(gen_mgc, axis=0), (-1, 1))

            local_gv = (ref_gv_std/gen_gv_std) * (gen_std - gen_gv_mean) + ref_gv_mean;

            enhanced_mgc = np.repeat(local_gv, frame_number, 1).T / np.repeat(gen_std, frame_number, 1).T * (gen_mgc - np.repeat(gen_mu, frame_number, 1).T) + np.repeat(gen_mu, frame_number, 1).T;

            new_mgc_file_name = files['mgc']+'_p_mgc'
            io_funcs.array_to_binary_file(enhanced_mgc, new_mgc_file_name)

            mgc_file_name = files['mgc']+'_p_mgc'

        if cfg.do_post_filtering and cfg.apply_GV:
            logger.critical('Both smoothing techniques together can\'t be applied!!\n' )
            raise

        ###mgc to sp to wav
        if cfg.vocoder_type == 'STRAIGHT':
            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sp=files['sp']))
            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], f0=files['f0']))
            run_process('{x2x} +fa {f0} > {f0a}'.format(x2x=SPTK['X2X'], f0=files['f0'], f0a=files['f0'] + '.a'))

            if cfg.use_cep_ap:
                run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} > {ap}'
                            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim-1, fl=cfg.fl, bap=files['bap'], ap=files['ap']))
            else:
                run_process('{bndap2ap} {bap} > {ap}'
                             .format(bndap2ap=STRAIGHT['BNDAP2AP'], bap=files['bap'], ap=files['ap']))

            run_process('{synfft} -f {sr} -spec -fftl {fl} -shift {shift} -sigp 1.2 -cornf 4000 -float -apfile {ap} {f0a} {sp} {wav}'
                        .format(synfft=STRAIGHT['SYNTHESIS_FFT'], sr=cfg.sr, fl=cfg.fl, shift=cfg.shift, ap=files['ap'], f0a=files['f0']+'.a', sp=files['sp'], wav=files['wav']))

            run_process('rm -f {sp} {f0} {f0a} {ap}'
                        .format(sp=files['sp'],f0=files['f0'],f0a=files['f0']+'.a',ap=files['ap']))
        elif cfg.vocoder_type == 'WORLD':

            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']))

            run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(sopr=SPTK['SOPR'],bap=files['bap'],x2x=SPTK['X2X'],ap=files['ap']))

            ### If using world v2, please comment above line and uncomment this
            #run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} | {sopr} -d 32768.0 -P | {x2x} +fd > {ap}'
            #            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim, fl=cfg.fl, bap=bap_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], ap=files['ap']))

            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))

            run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                         .format(synworld=WORLD['SYNTHESIS'], fl=cfg.fl, sr=cfg.sr, f0=files['f0'], sp=files['sp'], ap=files['ap'], wav=files['wav']))

            run_process('rm -f {ap} {sp} {f0}'.format(ap=files['ap'],sp=files['sp'],f0=files['f0']))

        os.chdir(cur_dir)


def wavgen_magphase(gen_dir, file_id_list, cfg, logger):

    # Import MagPhase and libraries:
    sys.path.append(cfg.magphase_bindir)
    import libutils as lu
    import libaudio as la
    import magphase as mp

    nfiles = len(file_id_list)
    for nxf in xrange(nfiles):
        filename_token = file_id_list[nxf]
        logger.info('Creating waveform for %4d of %4d: %s' % (nxf+1, nfiles, filename_token))

        for pf_type in cfg.magphase_pf_type:
            gen_wav_dir = os.path.join(gen_dir + '_wav_pf_' + pf_type)
            lu.mkdir(gen_wav_dir)
            mp.synthesis_from_acoustic_modelling(gen_dir, filename_token, gen_wav_dir, cfg.mag_dim, cfg.real_dim,
                                                            cfg.sr, pf_type=pf_type, b_const_rate=cfg.magphase_const_rate)

    return

from label_normalisation import HTSLabelNormalisation

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
    phoneme_superset = [0, 4, 5, 46, 51, 56, 47, 52, 58]
    label_string = ph_sel_dict
    for label_row in lab_idx:
        for k, v in label_row.items():
            if k.startswith('bin') and v in phoneme_superset:
                label_type_file.append(ph_sel_dict[str(v)])
    return(label_type_file)

def wavwrite(filename, wav, fs, enc=None, norm_max_ifneeded=False, norm_max=False, verbose=0):

    if norm_max:
        wav_max = np.max(np.abs(wav))
        wav /= 1.05*wav_max

    elif norm_max_ifneeded:
        wav_max = np.max(np.abs(wav))
        if wav_max>=1.0:
            print('    WARNING: sigproc.wavwrite: waveform in file {} is clipping. Rescaling between [-1,1]'.format(filename))
            wav /= 1.05*wav_max

    if np.max(np.abs(wav))>=1.0:
        print('    WARNING: sigproc.wavwrite: waveform in file {} is clipping.'.format(filename))

    if enc==None:        enc = np.int16
    elif enc=='pcm16':   enc = np.int16
    elif enc=='float32': raise ValueError('float not supported by scipy.io.wavfile')
    wav = wav.copy()
    wav = enc(np.iinfo(enc).max*wav)
    scipy.io.wavfile.write(filename, fs, wav)
    if verbose>0:
        print('    Output: '+filename)




def wavgen_pulsemodel(gen_dir, file_id_list, nm_type, fs, dftlen, shift, path_to_sptk = 0, path_to_hed = 0, path_to_lab = 0):

    import pml_synthesis_class as pml
    from itertools import groupby
    import scipy
    from scipy.io import wavfile   
    
    syn_func = pml.MerlinPmlSyn()

    gen_dir = gen_dir 
    sptk = path_to_sptk
    path_to_hed = path_to_hed
    lab_dir = path_to_lab
    op_path = lab_dir
    #normlab_dir = os.path.join(path_to_lab, 'nn_no_silence_lab_norm_426')
    
    pml_wav_dir  = os.path.join(gen_dir, 'wav' )

    if not os.path.exists(pml_wav_dir):
        os.mkdir(pml_wav_dir)

    ph_sel_test = {'0': 'Vowel', '4': 'Liquid', '5':'Nasal', '46': 'Voiced_Stop', '51': 'Voiced_Fricative', 
                '56': 'Affricate_Consonant', '47': 'Unvoiced_Stop', '52': 'Unvoiced_Fricative', '58': 'Silence'}

    label_normaliser = HTSLabelNormalisation(question_file_name = path_to_hed)
    discrete_dict, continuous_dict = label_normaliser.load_question_set_continous(path_to_hed)

    nfiles = len(file_id_list)

    fs = fs 
    op_format = 2
    mcsize = 59
    shift = shift
    dftlen = dftlen

    for file in file_id_list:

        path_to_lf0s = os.path.join(gen_dir, file + '.lf0')
        path_to_mgc = os.path.join(gen_dir, file + '.mgc')
        path_to_pddm = os.path.join(gen_dir, file + '.pddm')        

        f0 = np.fromfile(path_to_lf0s, dtype=np.float32)
        f0[f0>0] = np.exp(f0[f0>0])
        ts = (shift)*np.arange(len(f0))
        f0s = np.vstack((ts, f0)).T

        pdd_thresh = 0.75 
        MPDD = np.fromfile(path_to_pddm, dtype=np.float32)
        MPDD = MPDD.reshape((len(f0s), -1))
        PDD = syn_func.mcep2spec(MPDD, syn_func.bark_alpha(fs), dftlen, gen_dir, sptk)
        
        MCEP = np.fromfile(path_to_mgc, dtype=np.float32)
        MCEP = MCEP.reshape((len(f0s), -1))
        SPEC = syn_func.mcep2spec(MCEP, syn_func.bark_alpha(fs), dftlen, gen_dir, sptk)

        curr_file = os.path.join(lab_dir, file + '.lab')
        curr_lfm = syn_func.read_binfile(curr_file, dim=426, dtype=np.float32)
        n_frames = len(curr_lfm)
        curr_lab_idx = get_label_type(n_frames_total = n_frames, label_feature_matrix = curr_lfm, discrete_dict = discrete_dict, continuous_dict = continuous_dict)
        label_strings = label_lookup_all(curr_lab_idx, discrete_dict, continuous_dict, ph_sel_test) 
        HARM = PDD.copy()
        HARM_LAB = list(zip(label_strings, HARM))

        #maybe necessary to clip to 70Hz to avoid warning message 
        f0s = syn_func.analysis_f0postproc(wav=None, fs = 16000, f0s = f0s, f0_min=70, f0_max=600, shift = 0.005)

        if nm_type == 'binary':
            NM = syn_func.analysis_nm_bin(fs = fs, f0s = f0s, PDD= PDD, nm_clean = True, verbose=1)
            wav_bin = syn_func.synthesize(fs = fs, f0s = f0s, SPEC = SPEC, NM = NM)
            outfile = os.path.join(op_path, file)
            wavwrite("%s_bin.wav" %outfile, wav_bin, fs, norm_max_ifneeded=True, verbose=1)

        elif nm_type == 'stepwise':
            NM = syn_func.analysis_nm_steps(fs = fs, f0s = f0s, file_id = file , harm_lab = HARM_LAB, dftlen = dftlen)
            wav_cont = syn_func.synthesize(fs = fs, f0s = f0s, SPEC = SPEC, NM = NM, nm_cont = True)
            outfile = os.path.join(op_path, file)
            wavwrite("%s_cont.wav" %outfile, wav_cont, fs, norm_max_ifneeded=True, verbose=1)
        
        elif nm_type == 'None':
            wav_non = syn_func.synthesize(fs = fs, f0s = f0s, file_id = file, NM = None)
            outfile = os.path.join(op_path, file)
            wavwrite("%s_non.wav" %outfile, wav_non, fs, norm_max_ifneeded=True, verbose=1)


def generate_wav(gen_dir, file_id_list, cfg):

    logger = logging.getLogger("wav_generation")

    ## STRAIGHT or WORLD vocoders:
    if (cfg.vocoder_type=='STRAIGHT') or (cfg.vocoder_type=='WORLD'):
        wavgen_straight_type_vocoder(gen_dir, file_id_list, cfg, logger)

    ## MagPhase Vocoder:
    elif cfg.vocoder_type=='MAGPHASE':
        wavgen_magphase(gen_dir, file_id_list, cfg, logger)

    # Add your favorite vocoder here.
    elif cfg.vocoder_type=='pml':
        wavgen_pulsemodel(gen_dir, file_id_list, nm_type = 'binary', fs = cfg.sr, dftlen = 1024, shift = 0.005, path_to_sptk = cfg.sptk_bindir, path_to_hed = cfg.question_file_name, path_to_lab = cfg.test_synth_dir)
        wavgen_pulsemodel(gen_dir, file_id_list, nm_type = 'stepwise', fs = cfg.sr, dftlen = 1024, shift = 0.005, path_to_sptk = cfg.sptk_bindir, path_to_hed = cfg.question_file_name, path_to_lab = cfg.test_synth_dir)
        #wavgen_pulsemodel(gen_dir, file_id_list, nm_type = 'None', fs = cfg.sr, dftlen = 1024, shift = 0.005)

    # If vocoder is not supported:
    else:
        logger.critical('The vocoder %s is not supported yet!\n' % cfg.vocoder_type )
        raise

    return
