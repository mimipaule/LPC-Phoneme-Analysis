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
    
    return pre_emphasized_signal, sr

def frame_signal(signal, frame_length, hop_length):
    """
    Chops the signal into overlapping frames using a Hamming window.
    """
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    
    signal_length = len(signal)
    num_frames = 1 + int(np.floor((signal_length - frame_length) / hop_length))
    
    # pad signal if it doesn't align perfectly
    pad_signal_length = (num_frames - 1) * hop_length + frame_length
    if pad_signal_length > signal_length:
        zeros = np.zeros(pad_signal_length - signal_length)
        signal = np.append(signal, zeros)
        
    # extract frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1)).T
    
    frames = signal[indices.astype(np.int32, copy=False)]
    
    # apply Hamming window to each frame
    window = np.hamming(frame_length)
    frames *= window
    
    return frames
