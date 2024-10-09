import librosa
from matplotlib import pyplot as plt
import numpy as np

def draw_spec(x,
              figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=16000, cmap='inferno',
              vmin=-50, vmax=40, use_colorbar=True,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    
    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        # plt.close()
        plt.show()
        return stft
    
    
from scipy.signal import firwin, lfilter,freqz
def lpf(y, sr=16000, cutoff=500, plot_resp=False, window='hamming', figsize=(10,2)):
    """ 
    Applies FIR filter
    cutoff freq: cutoff freq in Hz
    """
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    taps = firwin(numtaps=101, cutoff=normalized_cutoff, window=window)
    y_lpf = lfilter(taps, 1.0, y)
    # y_lpf = np.convolve(y, taps, mode='same')
    
    # plt.plot(taps)
    # plt.show()
    if plot_resp:
        w, h = freqz(taps, worN=8000)
        plt.figure(figsize=figsize)
        plt.plot(0.5*sr*w/np.pi, np.abs(h), 'b')
        plt.title("FIR Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.xlim(0, sr/2)
        plt.grid()
        plt.show()

    return y_lpf

def audioshow(y, sr=16000, figsize=(8,2)):
    """ Draw audio in time domain """
    time = np.linspace(0, len(y)/sr, len(y))
    plt.figure(figsize=figsize)
    plt.plot(time, y)
    plt.xlim(0, time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('amplitude')
    plt.grid()
    plt.show()
    
def extract_frames(y, sr=16000, win_type='hamming', win_length=320, hop_length=160,):
    """ 
    Extract frames identical to librosa STFT 
    ** Returns:
        frame list that contations every time-domain frames
    """
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    if win_length < win_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to win_length ({win_length})")
    
    y = np.pad(y, (win_length//2, win_length//2), mode='constant', constant_values=0)  # padding
    siglen_pad = len(y)  # Length of the padded signal

    # window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    frame_list = []
    # Frame processing
    for center in range(win_length//2, siglen_pad, hop_length):
        if center > siglen_pad - win_length//2:
            break #end condition
        start = center - win_length//2
        end = center + win_length//2
        frame = y[start:end]
        frame = frame * window
        frame_list.append(frame)
    return frame_list

def VisualizeFrames(signal, sr, win_length, hop_length, frame_indices, win_type="rectangular", plot_frames=True, figsize=(10,5)):
    """ 
    Get input signal with frame indices, represent frames and visualize
    Usage: VisualizeFrames(yr, sr, win_len, hop_len, 5, 20, win_type="rectangular", plot_frames=True)
    """
    from HW3.mystft import extract_frames
    fig, ax = plt.subplots(figsize=figsize)
    time = np.arange(0, len(signal) / sr, 1 / sr)
    ax.plot(time, signal, label='waveform')
    
    frame_arr = extract_frames(signal, win_type=win_type, win_length=win_length, hop_length=hop_length)    
    for index in frame_indices:
        startind = hop_length * index
        endind = startind + win_length
        frame = frame_arr[index]
        ax.plot(time[startind:endind], frame, label=f'frame {index}', linestyle='--')
    
    ax.set_xlabel('Time(s)', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=14)
    # ax.set_title('Frame indices: ' + ', '.join(map(str, frame_indices)), fontsize=16, fontweight='bold')
    # ax.set_title('/sa/ Signal', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(0, time[-1])
    ax.legend(fontsize=15, loc='lower left')
    ax.grid()

    # Additionaly plot frames
    if plot_frames: 
        num_frames = len(frame_indices)
        num_cols = 1  
        num_rows = num_frames  
        
        plt.figure(figsize=[6, num_rows * 3])  # Each row has height 3
        frame_color = ['orange','green','red','purple'] # max number of frames to visualize
        for i, index in enumerate(frame_indices):
            start_time_ms = index * hop_length / sr * 1000
            end_time_ms = start_time_ms + win_length / sr * 1000
            time_frame_ms = np.linspace(start_time_ms, end_time_ms, win_length, endpoint=False)

            plt.subplot(num_rows, num_cols, i+1)
            frame = frame_arr[index]
            plt.plot(time_frame_ms, frame, color=frame_color[i])
            plt.grid()
            plt.xlim(start_time_ms, end_time_ms)
            plt.title(f"Frame {index}", fontsize=16)
            # if i==0:
            #     plt.title(f"Unvoiced Region (Frame {index})", fontsize=16, fontweight='bold')
            # else:
            #     plt.title(f"Voiced Region (Frame {index})", fontsize=16, fontweight='bold')
                
            plt.xlabel("Time (ms)", fontsize=12) 
            plt.ylabel("Amplitude",fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
        plt.tight_layout()  
    plt.show()



