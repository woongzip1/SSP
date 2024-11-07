import numpy as np
from matplotlib import pyplot as plt
import scipy
from HW3.pitchestimate import ThresholdClipper

""" Utility Functions """

## Auto Correlation Sequence with signal length
def auto_corr(signal):
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    return corr 

# My Derbin's Algorithm
def derbin(r, p):
    E = np.zeros(p+1)
    a = np.zeros((p+1,p+1))
    
    a[0][0] = 1
    E[0] = r[0]

    for i in range(1,p+1):
        ## sigma
        j=1
        sumj = 0
        while(j <= i-1):
            sumj+=a[i-1][j]*r[i-j]
            j += 1
        
        k_i = (r[i] - sumj) / E[i-1] 
        a[i][i] = k_i

        ## i-order new coefficient
        for j in range(1,i):
            a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
        E[i] = (1 - k_i**2)*E[i-1]
        coeff = a[p][1:]
    return coeff,E

# # Derbin's Algorithm (As a reference)
# def ref_derbin(r, order):
#     # r : 1-D auto corr array
#     a = np.zeros((order+1,order+1))
#     # store prediction error for each step
#     E = np.zeros(order+1)
#     # First coeff
#     a[0][0] = 1
#     # Initial prediction error : power
#     E[0] = r[0]
    
#     # iterate from 1 to order p 
#     for i in range(1,order+1):
#         sum_j = sum(a[i-1][j] * r[i-j] for j in range(1,i))
#         k_i = (r[i] - sum_j ) / E[i-1]
        
#         # Update coefficeints for current step
#         a[i][i] = k_i
#         for j in range(1,i):
#             a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
#         #Update Error
#         E[i] = (1-k_i**2) * E[i-1]
#         # print("i={}, ki={}".format(i,k_i))
#     # Extract final coeff, exclude a0    
#     coeff = a[order][1:]
#     return coeff,E


# Calculate LPC Coefficients in the Frame: a1 a2 a3 ...
def LPC(frame, order=10):
    coeff_arr = np.zeros(order)
    # error
    if len(frame) < order:
        print('frame is longer than order')
        return -1

    coeff, err = derbin(auto_corr(frame),p=order)
    return coeff, err

## LPC with Direct Matrix Inverse
def LPC_inv(frame, order=10):
    coeff_arr = np.zeros(order)
    # error
    if len(frame) < order:
        print('frame is longer than order')
        return -1
    # Tx = b
    ac = auto_corr(frame)[:order]
    mat_T = make_toeplitz(ac)
    vec_b = auto_corr(frame)[1:order+1]
    coeff_arr  = np.dot(np.linalg.inv(mat_T),vec_b)
    return coeff_arr

# Make Toeplitz Matrix using Auto Correlation
def make_toeplitz(ac):
    p = len(ac)
    toeplitz_mat = np.zeros((p,p))
    ac_flip = ac[::-1][:-1]
    
    for i in range(p):
        toeplitz_mat[i,:] = np.concatenate((ac_flip[p-i-1:],ac[:p-i]))
    return toeplitz_mat

def PlotLPCSpectrum(signal, sr, p=10, dftlen=2048, figsize=(10,6), title=None):
    """
    Plot Envelope Using LP Coefficients
    input: frame(waveform), sr, p(order), dftlen, figsize
    output: None
    """
    freqs = np.linspace(0, sr/2, dftlen//2)
    signal_f = np.fft.rfft(signal, dftlen)[:-1]

    lp_coeff,_ = LPC(signal, order=p) #a1 a2 a3 ...
    
    r = auto_corr(signal)
    gain = np.sqrt(r[0] - np.sum(lp_coeff[:p] * r[1:p+1]))        
    # gain = np.sqrt(energy / SignalEnergy(excitation)) 

    print("gain:",gain)
    # Frequency Response
    w2, h2 = scipy.signal.freqz([gain], np.concatenate(([1],-lp_coeff)), worN = dftlen//2)

    plt.figure(figsize=figsize)
    plt.plot(freqs, 20 * np.log10(np.abs(signal_f)), label='Original')
    plt.xlim(0, sr//2)
    plt.grid(True)

    plt.plot(freqs, 20 * np.log10(np.abs(h2)), linewidth=3, label='Envelope via LPC')
    if title:
        plt.title(title, fontsize=25)
    else:
        plt.title('Order : {}'.format(p), fontsize=30)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid(True)
    # plt.ylim(-50, 35)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()


## This is quite bitter sweet..
def PitchDetector(frames, sr=16000):
    """
    Input: Frame List, sr
    Output: VoicedFlags, PitchList
    """
    ## LPF to signal
    # signal = LowPassFilter(signal, sr, cutoff=900)
    
    num_frames = len(frames)
    voiced_flags = []
    pitch_list = []
    for frame_idx in range(num_frames):
        signal = frames[frame_idx] 
        
        ## Clipping 
        Clipper = ThresholdClipper(signal)
        signal_clipped = Clipper.center_clip(Clipper.CL)
        ac_arr = auto_corr(signal_clipped)

        # Enery
        energy = ac_arr[0]
        voice_thres = energy * 0.35

        # Find Peaks of AC 
        peakval = np.max(ac_arr)    
        maxima_indices, _ = scipy.signal.find_peaks(ac_arr)
        maxima_indices = maxima_indices[maxima_indices>50]
        # print(maxima_indices)
        
        if maxima_indices.size > 0:
            maxval = np.max([ac_arr[i] for i in maxima_indices])
            idx = np.argmax([ac_arr[i] for i in maxima_indices])
            max_idx = maxima_indices[idx]
            # print(maxval, voice_thres)
            voiced_flag = 1 if maxval > voice_thres else 0
            pitch = sr / max_idx if voiced_flag else 0
            
        else:
            voiced_flag = 0
            pitch = 0
        voiced_flags.append(voiced_flag)
        pitch_list.append(pitch)

    pitch_list = scipy.signal.medfilt(pitch_list, kernel_size=5)
    return voiced_flags, pitch_list

def lp_analysis(frames, order=10):
    """
    LP analysis for every frames
    Input: framelist, order
    Output: lpclist [#Frames x Order] contains a1 a2 ... ap 
            gainlist (#Frames)
    """
    frame_num = len(frames)
    gain_list = []
    lpc_list = []
    for frame_idx in range(frame_num):
        frame = frames[frame_idx]
        # lpc = librosa.lpc(frame, order=order) # 1 -a1 -a2 -a3
        # lpc_list.append(-1*lpc[1:])
        lp_coeff, err = LPC(frame, order=order) # a1 a2 a3 
        lpc_list.append(lp_coeff[:])
        
        gain = np.sqrt(auto_corr(frame)[0] - np.sum(lp_coeff[:order] * auto_corr(frame)[1:order+1]))        
        gain_list.append(gain)
    lpc_list = np.array(lpc_list) # [frames x order]
    
    return lpc_list, gain_list

def SynthesizeWaveform(lpc_list, gain_list, pitchlist, sr=8000, win_length=400, hop_length=160, 
                       win_type='hamming', printflag=False, align_impulse=True):
    """
    Synthesize waveform from [lp coefficients, gains, pitchlist]
    After synthesizing the windowed waveform, 
    overlap-add the synthesized frames to get the final waveform.
    Without impulse alignment, the wave sounds much more robotic.
    
    Input: lpc_list, gain_list, pitchlist, sr, win_length, hop_length, align_impulse (bool)
    Output: list of synthesized frames (#frames)
    """
    num_frames, order = lpc_list.shape
    prev_is_voiced = 0
    prev_indarr = [] # previous frame information
    
    synthesized_frames = []
    for frame_idx in range(num_frames):
        gain = gain_list[frame_idx]
        coeff = lpc_list[frame_idx]  # a1 a2 a3 ...
        pitch = pitchlist[frame_idx]
        lpc_coeff = np.concatenate(([1], -coeff))  # 1 -a1 -a2 ...
        impulse_train = np.zeros(win_length)
                
        if pitch > 0:  # voiced
            # Determine the starting point of the impulse based on previous frame's voicing
            current_is_voiced = 1  # update voiced
            samples_per_impulse = int(sr / pitch)  # pitch period in samples
            indarr = []  # index array for impulses
            
            if align_impulse:
                # Align impulses based on previous frame
                if prev_is_voiced:  # voiced -> voiced
                    for ind in prev_indarr:
                        startind = ind - hop_length
                        
                        if startind <= 0:
                            continue  # Skip if start index is out of range
                        
                        # Generate indices for impulse train (Impulse train alignment)
                        indarr = list(range(startind, startind + win_length, samples_per_impulse))
                        indarr = [i for i in indarr if i < win_length + hop_length]
                        break  # Exit loop after finding the first valid startind
                        
                else:  # unvoiced -> voiced transition
                    startind = 0
                    indarr = list(range(startind, win_length, samples_per_impulse))

                if printflag:
                    print(indarr, f"pitch:{samples_per_impulse} prev:{prev_is_voiced}")

            else:
                # No alignment: start impulses from index 0
                indarr = list(range(0, win_length, samples_per_impulse))

            # Create the impulse train based on indarr
            for ind in indarr:
                if ind < win_length:
                    impulse_train[ind] = 1 
            excitation = impulse_train

        else: # unvoiced
            current_is_voiced = 0
            # Gaussian white noise
            excitation = np.random.normal(loc=0.0, scale=gain, size=win_length)    
                    
        # Filtering with LPC coefficients
        synthesized_signal = scipy.signal.lfilter([gain], lpc_coeff, excitation)
        
        # Update previous frame states
        prev_is_voiced = current_is_voiced
        prev_indarr = indarr if current_is_voiced else []
        synthesized_frames.append(synthesized_signal)
    
    from HW3.mystft import overlapadd
    y = overlapadd(synthesized_frames, win_length=win_length, hop_length=hop_length, win_type=win_type, griffin=True)
    return y