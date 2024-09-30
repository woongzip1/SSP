import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.signal import medfilt

def ste(frames, plot=False, figsize=(6,2)):
    """
    Short-Time Energy (STE) calculation.

    Input: Frame array that consists of every frames
    Returns:
    ste_values: 2D NumPy array representing the short-time energy for each frame.
    """
    ste_values = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        # Calculate short-time energy
        energy = np.sum(frame ** 2)
        ste_values.append(energy)
    
    return np.array(ste_values)

def zcr(frames, plot=False, figsize=(6,2)):
    """
    Zero Crossing Rate (ZCR) calculation.

    Returns:
    zcr_values: 2D NumPy array representing the zero crossing rate for each frame.
    """
    zcr_values = []

    for frame_idx in range(len(frames)):
        # Calculate zero crossing rate
        frame = frames[frame_idx]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr_values.append(zero_crossings)
    
    return np.array(zcr_values)

def vad(ste_arr, threshold=None, frame_duration=10):
    """
    Simple VAD algorithm that uses STE as a parameter
    
    Args:
    ste_arr: Short-Time Energy array
    threshold: Threshold for STE to detect voiced segments. default 10% AVG STE
    frame_duration: # of future frames to detect voice activation
    
    Returns:
    vad_flags: Binary array
    """
    if threshold == None:
        threshold = 0.1 * np.mean(ste_arr)
        
    vad_flags = np.zeros(len(ste_arr))
    voiced_flag = False
    silence_counter = 0
    
    for i in range(len(ste_arr)):
        if ste_arr[i] > threshold: # VAD activated
            voiced_flag = True
            silence_counter = 0  # Reset the silence counter
        else: # Silence estimated
            silence_counter += 1
            
            # If the signal is below the threshold for frame_duration frames, set the flag to unvoiced
            if silence_counter > frame_duration:
                voiced_flag = False

        vad_flags[i] = 1 if voiced_flag else 0
    vad_flags = medfilt(vad_flags, 11) # median filtering
    return vad_flags

def plot_ste_zcr_vad(wave, ste, zcr, vad, sr=16000, hop_length=160, figsize=(10, 8)):

    time_axis_ste_zcr = np.arange(len(ste)) * hop_length / sr  # X-axis for STE and ZCR
    time_axis_wave = np.arange(len(wave)) / sr  # X-axis for waveform
    
    # Figures
    fig, axs = plt.subplots(4, 1, figsize=figsize)
        
    axs[0].plot(time_axis_wave, wave, color='b')
    axs[0].set_title("Waveform", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=10)
    axs[0].set_xlim([time_axis_wave[0], time_axis_wave[-1]])  
    
    # Plot STE
    axs[1].plot(time_axis_ste_zcr, ste, color='orange')
    axs[1].set_title("Short-Time Energy (STE)", fontsize=12)
    axs[1].set_ylabel("Energy", fontsize=10)
    axs[1].set_xlim([time_axis_ste_zcr[0], time_axis_ste_zcr[-1]])  
    axs[1].grid(True)  
    
    # Plot ZCR
    axs[2].plot(time_axis_ste_zcr, zcr, color='green')
    axs[2].set_title("Zero Crossing Rate (ZCR)", fontsize=12)
    axs[2].set_ylabel("Rate", fontsize=10)
    axs[2].set_xlabel("Time (s)", fontsize=10)
    axs[2].set_xlim([time_axis_ste_zcr[0], time_axis_ste_zcr[-1]])  
    axs[2].grid(True)
    
    # Plot VAD
    axs[3].plot(time_axis_ste_zcr, vad, color='red')
    axs[3].set_title("Voice Activity Detection (VAD)", fontsize=12)
    axs[3].set_ylabel("VAD Flag", fontsize=10)
    axs[3].set_xlabel("Time (s)", fontsize=10)
    axs[3].set_xlim([time_axis_ste_zcr[0], time_axis_ste_zcr[-1]])  
    axs[3].grid(True) 
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def main():
    ### Usage 
    # y, sr = librosa.load("./p232_011.wav", sr=None)
    y, sr = librosa.load("./sample_crop_16kHz.wav", sr=None)
    # y, sr = librosa.load("./s5_180_mic1.flac")

    from utils_global import lpf
    y = lpf(y, cutoff=1000)

    frames = extract_frames(y, sr=sr, win_type='rectangular', win_length=320, hop_length=160)

    ste_arr = ste(frames)
    zcr_arr = zcr(frames)
    vad_arr = vad(ste_arr, frame_duration=15)

    plot_ste_zcr_vad(y, ste_arr, zcr_arr, vad_arr)
    t = draw_spec(y, sr=sr, win_len=320, hop_len=160, use_colorbar=False, figsize=(11.7,2))
    display(Audio(y, rate=sr))

    print(len(ste_arr))
    
    ### Pitch using AutoCorrelation
    pitches = pitch_estimate_ac(y, sr=16000, vad_arr=vad_arr,  win_type='rectangular', win_length=320, hop_length=160, clip=True)
    plot_pitch_contour(pitches, sr=16000, hop_length=160, figsize=(11.7,2))
    pitches = medfilt(pitches, kernel_size=5) 
    plot_pitch_contour(pitches, sr=16000,hop_length=160, figsize=(11.7,2), title='After Median')

    pitches = pitch_estimate_amdf(y, sr=16000, vad_arr=vad_arr,  win_type='rectangular', win_length=320, hop_length=160, max_lag=200, plot_amdf=False)
    plot_pitch_contour(pitches, sr=16000, hop_length=160, figsize=(11.7,2))
    pitches = medfilt(pitches, kernel_size=5) 
    plot_pitch_contour(pitches, sr=16000,hop_length=160, figsize=(11.7,2), title='After Median')

    print(len(pitches))

if __name__ == '__main__':
    import sys
    import os

    # Add upper directory
    current_dir = os.path.dirname(os.path.abspath('ste_zcr.py'))
    upper_dir = (os.path.join(current_dir, '..'))
    sys.path.append(upper_dir)

    from utils_global import draw_spec, audioshow, extract_frames
    import librosa
    from IPython.display import Audio, display
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.signal import medfilt
    from pitchestimate import pitch_estimate_ac, pitch_estimate_amdf, plot_pitch_contour

    main()