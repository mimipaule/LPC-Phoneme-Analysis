from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def _interpolate_nans(track):
    """
    Helper to linearly interpolate NaNs in a 2D array of temp formant tracks.
    """
    track = track.copy()
    for col in range(track.shape[1]):
        nans = np.isnan(track[:, col])
        if np.all(nans): continue
        def get_x(z): return z.nonzero()[0]
        track[nans, col] = np.interp(get_x(nans), get_x(~nans), track[~nans, col])
    return track

def compute_dtw_distance(track1, track2):
    """
    Computes DTW distance between two temporal sequences.
    Interpolates missing values (NaNs) and normalizes by path length.
    """
    t1 = _interpolate_nans(track1)
    t2 = _interpolate_nans(track2)
    
    # drop any edge case rows with lingering NaNs
    t1 = t1[~np.isnan(t1).any(axis=1)]
    t2 = t2[~np.isnan(t2).any(axis=1)]
    
    if len(t1) == 0 or len(t2) == 0:
        return np.inf

    distance, path = fastdtw(t1, t2, dist=euclidean)
    # normalize distance by alignment path length
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
