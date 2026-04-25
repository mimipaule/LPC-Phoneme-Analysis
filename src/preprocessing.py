import librosa
import numpy as np
import scipy.signal

def load_and_preprocess(filepath, target_sr=16000):
    """
    Loads audio, downsamples, and applies pre-emphasis filter.
    """
    # load audio and resample to target_sr
    signal, sr = librosa.load(filepath, sr=target_sr)
    
    # pre-emphasis filter: y(t) = x(t) - alpha * x(t-1)
    alpha = 0.97
    pre_emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])

    if np.max(np.abs(pre_emphasized_signal)) > 0:
        pre_emphasized_signal = pre_emphasized_signal / np.max(np.abs(pre_emphasized_signal))
    
    return pre_emphasized_signal, sr

def frame_signal(signal, sr, frame_duration=0.025, hop_duration=0.010, rms_threshold=0.01):
    """
    Finalized framing function:
    1. Removes DC Offset
    2. Pads signal for perfect alignment
    3. Chops into overlapping frames
    4. Applies Hamming window
    5. Filters out silent frames (VAD)
    """
    # DC Offset Removal
    signal = signal - np.mean(signal)
    
    # convert durations (seconds) to samples
    frame_length = int(frame_duration * sr)
    hop_length = int(hop_duration * sr)
    signal_length = len(signal)
    
    # calculate number of frames and pad if necessary
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / hop_length)) + 1
    pad_signal_length = (num_frames - 1) * hop_length + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    
    # extract frames using manual indexing
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # apply Hamming window
    window = np.hamming(frame_length)
    frames *= window
    
    # voice Activity Detection (RMS Energy Filtering)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    voiced_mask = rms > rms_threshold
    
    final_frames = frames[voiced_mask]
    
    print(f"Total frames: {num_frames} | Voiced frames kept: {len(final_frames)}")
    return final_frames