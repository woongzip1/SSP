import numpy as np
import torch
import librosa
from IPython.display import Audio, display
from matplotlib import pyplot as plt
import soundfile as sf

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

def stft(y, sr=16000, win_type='hamming', win_length=320, hop_length=160, n_fft=None,
         pad_mode='constant', figsize=(14, 4), cmap='viridis', 
        #  vmin=-50, vmax=40,
         use_colorbar=True, plot=False, return_fig=False):
    
    """ 
    STFT Implementation identical to librosa.stft 
    This implementation is based on center=='True' option
    ** Returns:
        spec: Magnitude spectrogram (NFFT//2+1 x Frames).
        Returns the figure if `return_fig` is True.
    """
    if not n_fft:
        n_fft = win_length
    
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})")
    
    siglen_sec = len(y)/sr
    y = np.pad(y, (n_fft//2, n_fft//2), mode=pad_mode, constant_values=0)  # padding
    siglen_pad = len(y)  # Length of the padded signal

    # window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    spec = []
    # Frame processing
    for center in range(n_fft//2, siglen_pad, hop_length):
        if center > siglen_pad - n_fft//2:
            break #end condition

        start = center - win_length//2
        end = center + win_length//2
        frame = y[start:end]
        frame = frame * window

        # pad until n_fft       
        padlen = n_fft - len(frame)
        frame = np.pad(frame, pad_width=[padlen//2, padlen//2], mode='constant')
        frame_fft = np.fft.fft(frame)[:n_fft//2 + 1]
        spec.append(frame_fft)

    spec = np.array(spec).T  # [freq x timeframe]
    # spec = np.abs(spec)

    # Plot option
    if plot:
        fig = plt.figure(figsize=figsize)
        plt.imshow(np.abs(spec), aspect='auto', 
                   cmap=cmap, 
                #    vmin=vmin, vmax=vmax,
                   origin='lower', extent=[0, siglen_sec, 0, sr//2])

        if use_colorbar: plt.colorbar()
        plt.title("STFT Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        if return_fig:
            plt.close()
            return spec, fig
        else:
            plt.show()
            return spec
    else:
        return spec

def istft(Y_w, win_length, hop_length, n_fft, win_type='hann'):
    """
    ISTFT Implementation identical to librosa.istft     
    ** Returns:
        y_buffer: Reconstructed time-domain signal
    """
    if not n_fft:
        n_fft = win_length  # Default to win_length if n_fft is not provided
        
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})")

    # windows
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")
    padlen = n_fft - win_length
    window = np.pad(window, (padlen//2,padlen//2), mode='constant')

    # Reconstruct Y to get full spectrum in frequency axis
    Y_flip = np.flipud(Y_w)[1:-1]
    Y_w = np.concatenate((Y_w,np.conj(Y_flip)), axis=0) # Note that phase is odd
    
    num_freq_bins, num_frames = Y_w.shape
    x_len = n_fft + (num_frames - 1) * hop_length
    y_buffer = np.zeros(x_len)
    window_sum = np.zeros(x_len)

    # Overlap-add process
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        frame = np.real(np.fft.ifft(Y_w[:,frame_idx]))  # Inverse FFT
        
        # y_buffer[start:start + n_fft] += window * frame
        # window_sum[start:start + n_fft] += window ** 2
        y_buffer[start:start + n_fft] += frame
        window_sum[start:start + n_fft] += window

    # Normalize by window overlap factor
    y_buffer /= np.where(window_sum > 1e-10, window_sum, 1e-10)

    # crop out to remove paddings (center-based STFT)
    y_buffer = y_buffer[n_fft//2:-n_fft//2]
    window_sum = window_sum[n_fft//2:-n_fft//2]
    return y_buffer, window_sum


def main():
    # Usage
    hop_length = 375
    win_length = 500
    n_fft = 512
    siglen = 40000
    sr = 16000
    #########################
    filename = librosa.ex('trumpet')
    y, sr = librosa.load(filename)
    y, sr = librosa.load("./sample_crop_16kHz.wav", sr=None)

    spec = stft(y, sr=sr, win_type='hamming', win_length=win_length, hop_length=hop_length, n_fft=n_fft, plot=False)
    spec_librosa = librosa.stft(y, window='hamming', win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    print(spec.shape, spec_librosa.shape)

    # Phase
    phase1 = np.angle(spec) / np.pi
    phase2 = np.angle(spec_librosa) / np.pi
    spec, spec_librosa = np.abs(spec), np.abs(spec_librosa)

    for freqaxis in range(1):
        print(spec[freqaxis,:5])
    print('')
    for freqaxis in range(1):
        print(spec_librosa[freqaxis,:5])

    # Plot Spectrogram 
    plt.figure(figsize=(10,2))
    plt.imshow(np.abs(spec), aspect='auto', 
                    cmap='viridis', 
                    origin='lower', extent=[0, 2.5, 0, sr//2])
    plt.colorbar()
    plt.ylabel(r"Freq (Hz)")
    plt.title("Spectrogram")
    plt.show()

    # Plot Phase Spectra
    plt.figure(figsize=(10,2))
    plt.imshow(phase1, aspect='auto', 
                    cmap='viridis', 
                    origin='lower', extent=[0, 2.5, 0, 1])
    plt.colorbar()
    plt.ylabel(r"rad ($\pi$)")
    plt.title("Phase(librosa)")
    plt.show()

    plt.figure(figsize=(10,2))
    plt.imshow(phase2, aspect='auto', 
                    cmap='viridis', 
                    origin='lower', extent=[0, 2.5, 0, 1])
    plt.colorbar()
    plt.ylabel(r"rad ($\pi$)")
    plt.title("Phase(Implemented)")
    plt.show()

    
if __name__ == '__main__':
    main()
    