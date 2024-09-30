import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import librosa

class ThresholdClipper:
    def __init__(self, function):
        self.function = function
        self.CL = self.calculate_thres()

    def calculate_thres(self):
        function = np.abs(self.function)
        first_max = np.max(function[0:len(function)//3]) 
        last_max = np.max(function[len(function)//3 * 2:])
        CL = 0.68 * min(first_max, last_max)
        return CL

    def center_clip(self, CL):
        function = self.function
        y = np.zeros_like(function)
        for n in range(len(y)):
            val = function[n]
            if val >= CL:
                y[n] = val - CL
            elif val <= (-1 * CL):
                y[n] = val + CL
            else:
                y[n] = 0
        return y

def autocorrelation(x):
    """ Compute the autocorrelation of a 1D signal. """
    ac = np.correlate(x, x, mode='full')
    return ac[len(ac) // 2:]

def amdf(x, max_lag=None):
    """ 
    Compute the Average Magnitude Difference Function (AMDF) 
    max_lag: maximum lag to calculate AMDF. Default signal length
    """
    if max_lag == None:
        max_lag = len(x)
        
    N = len(x)
    amdf_values = np.zeros(max_lag)
    
    for k in range(1, max_lag):
        diff_sum = 0
        for n in range(N - k):
            diff_sum += np.abs(x[n] - x[n + k])
        # amdf_values[k] = diff_sum 
        amdf_values[k] = diff_sum / (N - k) # Normalize
    
    return amdf_values

def plot_pitch_contour(pitch_list, sr, hop_length, freqlim=350, figsize=(11,3), title=None):
    """ Plot pitch contour over time, excluding pitch values that are 0 """

    pitch_list = np.array(pitch_list)
    
    # Time axis in seconds
    time_axis = np.arange(len(pitch_list)) * hop_length / sr
    
    # Filter out the 0 pitch values
    non_zero_indices = np.where(pitch_list > 0)[0]
    filtered_pitch_list = np.array(pitch_list)[non_zero_indices]
    filtered_time_axis = time_axis[non_zero_indices]
    
    # Plot the pitch contour
    plt.figure(figsize=figsize)
    plt.scatter(filtered_time_axis, filtered_pitch_list, label='Pitch Contour', s=4)
    plt.xlim(time_axis[0], time_axis[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.ylim(0, freqlim)
    
    if title:
        plt.title(title)
    else:
        plt.title('Pitch Contour')
    
    plt.grid(True)
    plt.show()


def pitch_estimate_ac(y, sr=16000, vad_arr=None, win_type='rectangular', win_length=320, hop_length=160, clip=True):
    """ 
    Extract frames, apply center clipping, autocorrelation, and peak estimation for pitch only when vad_arr is True.
    
    ** Returns:
        pitch_list: List of pitch estimates for each frame where vad_arr == True.
    """
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    
    y = np.pad(y, (win_length // 2, win_length // 2), mode='constant', constant_values=0)  # padding
    siglen_pad = len(y)

    # Windowing function
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    pitch_list = []
    
    # Ensure vad_arr is provided and matches the number of frames
    if vad_arr is None:
        raise ValueError("vad_arr must be provided.")
    
    if len(vad_arr) != (len(y) - win_length) // hop_length + 1:
        raise ValueError("vad_arr length does not match the number of frames.")

    frame_idx = 0
    for center in range(win_length // 2, siglen_pad, hop_length):
        if center > siglen_pad - win_length // 2:
            break  # End condition
        
        if vad_arr[frame_idx]:  # Only process pitch if VAD flag is True
            start = center - win_length // 2
            end = center + win_length // 2
            frame = y[start:end] * window

            # Step 1: Apply center clipping
            if clip:
                clipper = ThresholdClipper(frame)
                CL = clipper.calculate_thres()
                clipped_frame = clipper.center_clip(CL)
            else: 
                clipped_frame = frame

            # Step 2: Autocorrelation
            ac = autocorrelation(clipped_frame)

            # Step 3: Peak estimation (pitch estimation)
            if np.max(ac) > 0:
                peaks, _ = find_peaks(ac)
                if len(peaks) > 0:
                    pitch_period = peaks[np.argmax(ac[peaks])]  # Find peak with the highest autocorrelation value
                    pitch_freq = sr / pitch_period if pitch_period != 0 else 0
                    pitch_list.append(pitch_freq)
                else:
                    pitch_list.append(0)  # No pitch found
            else:
                pitch_list.append(0)  # No pitch found
        else:
            pitch_list.append(0)  # Silence / no pitch detected when vad_arr is False
        
        frame_idx += 1
    
    return pitch_list

def pitch_estimate_amdf(y, sr=16000, vad_arr=None, win_type='rectangular', win_length=320, hop_length=160, max_lag=200, plot_amdf=False):
    """ 
    Extract frames, apply center clipping, and use AMDF for pitch estimation only when vad_arr is True.
    
    ** Returns:
        pitch_list: List of pitch estimates for each frame where vad_arr == True.
    """
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    
    y = np.pad(y, (win_length // 2, win_length // 2), mode='constant', constant_values=0)  # padding
    siglen_pad = len(y)

    # Windowing function
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    pitch_list = []
    amdf_results = []
    num_frames_to_plot = 20 * 5
    frame_indices = []
    cnt = 0
    min_lag_offset = 25

    # Ensure vad_arr is provided and matches the number of frames
    if vad_arr is None:
        raise ValueError("vad_arr must be provided.")
    
    if len(vad_arr) != (len(y) - win_length) // hop_length + 1:
        raise ValueError("vad_arr length does not match the number of frames.")
    
    frame_idx = 0
    for center in range(win_length // 2, siglen_pad, hop_length):
        if center > siglen_pad - win_length // 2:
            break  # End condition

        if vad_arr[frame_idx]:  # Only process pitch if VAD flag is True
            start = center - win_length // 2
            end = center + win_length // 2
            frame = y[start:end] * window

            # Step 1: Apply center clipping
            clipper = ThresholdClipper(frame)
            CL = clipper.calculate_thres()
            clipped_frame = clipper.center_clip(CL)

            # Step 2: Compute AMDF
            amdf_values = amdf(clipped_frame, max_lag)
            if cnt < num_frames_to_plot and cnt % 5 == 0:
                amdf_results.append(amdf_values)  # Store for plotting
                frame_indices.append(cnt)

            # Step 3: Peak estimation (pitch estimation)
            if np.min(amdf_values[min_lag_offset:]) > 0:  # Only consider values after the offset
                peaks, _ = find_peaks(-amdf_values[min_lag_offset:])  # Find minima in AMDF values
                if len(peaks) > 0:
                    pitch_period = peaks[np.argmin(amdf_values[peaks + min_lag_offset])] + min_lag_offset
                    pitch_freq = sr / pitch_period if pitch_period != 0 else 0
                    pitch_list.append(pitch_freq)
                else:
                    pitch_list.append(0)  # No pitch found
            else:
                pitch_list.append(0)  # No pitch found
        else:
            pitch_list.append(0)  # No pitch calculated if VAD is False
        
        frame_idx += 1
        cnt += 1

    # Plot the AMDF results for the first 20 frames, if plot_amdf is True
    if plot_amdf:
        fig, axes = plt.subplots(5, 4, figsize=(20, 15))  # Create a 5x4 grid of subplots
        axes = axes.flatten()

        for i in range(min(num_frames_to_plot, len(amdf_results))):
            axes[i].plot(amdf_results[i])
            min_val = np.min(amdf_results[i][min_lag_offset:])  # Minimum value after offset
            min_idx = np.argmin(amdf_results[i][min_lag_offset:]) + min_lag_offset  # Index of minimum value
            axes[i].set_title(f'AMDF - Frame {frame_indices[i]}')
            axes[i].set_xlabel('Lag')
            axes[i].set_ylabel('AMDF Value')
            # Highlight the minimum value on the graph
            axes[i].scatter(min_idx, min_val, color='red')
            axes[i].text(min_idx, min_val, f'{min_val:.2f}', fontsize=10, color='red')

        plt.tight_layout()
        plt.show()

    return pitch_list

