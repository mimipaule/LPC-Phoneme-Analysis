from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def _interpolate_nans(track):
    """
    Helper to linearly interpolate NaNs in a 2D array of temp formant tracks.
    """
    # Ensure track is 2D (frames, features)
    if track.ndim == 1:
        track = track.reshape(-1, 1)
    
    if track.size == 0 or track.shape[1] == 0:
        return track

    track = track.copy()
    for col in range(track.shape[1]):
        nans = np.isnan(track[:, col])
        if np.all(nans): 
            continue
        
        # Get indices for non-NaN values
        def get_x(z): return z.nonzero()[0]
        
        non_nan_idx = get_x(~nans)
        nan_idx = get_x(nans)
        
        track[nans, col] = np.interp(nan_idx, non_nan_idx, track[~nans, col])
    return track

def compute_dtw_distance(track1, track2):
    """
    Computes DTW distance between two temporal sequences.
    Interpolates missing values (NaNs) and normalizes by path length.
    """
    # Safety check for empty inputs before processing
    if track1 is None or track2 is None or len(track1) == 0 or len(track2) == 0:
        return np.inf

    t1 = _interpolate_nans(track1)
    t2 = _interpolate_nans(track2)
    
    # Drop any rows that still contain NaNs (where interpolation was impossible)
    t1 = t1[~np.isnan(t1).any(axis=1)]
    t2 = t2[~np.isnan(t2).any(axis=1)]
    
    # Final check after cleaning
    if len(t1) < 2 or len(t2) < 2:
        return np.inf

    # fastdtw expects (N, D) arrays
    distance, path = fastdtw(t1, t2, dist=euclidean)
    
    # Normalize by alignment path length to account for different word durations
    return distance / len(path)

def compute_distance_matrix(tracks_dict):
    """
    Computes pairwise DTW distances for multiple language sequence tracks.
    """
    labels = list(tracks_dict.keys())
    n = len(labels)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                dist = compute_dtw_distance(tracks_dict[labels[i]], tracks_dict[labels[j]])
                matrix[i, j] = dist
                matrix[j, i] = dist
                
    return matrix, labels