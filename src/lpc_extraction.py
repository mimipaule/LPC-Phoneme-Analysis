import numpy as np
import librosa

def extract_lpc_coefficients(frame, order):
    """
    Extracts raw LPC coefficients for a given audio frame.
    """
    # librosa.lpc returns the AR coefficients [1, -a_1, -a_2, ..., -a_p]
    return librosa.lpc(y=frame, order=order)

def find_formants_from_lpc(lpc_coeffs, sr):
    """
    Calculates the roots of the LPC predictor to find exact formant frequencies.
    Filters out roots with wide bandwidths.
    """
    roots = np.roots(lpc_coeffs)
    
    # only care about roots with a positive imaginary part
    roots = [r for r in roots if np.imag(r) > 0]
    
    formants = []
    for r in roots:
        angle = np.angle(r)
        radius = np.abs(r)
        
        freq = angle * (sr / (2 * np.pi))
        
        # bandwidth B = - (sr / pi) * ln(radius)
        bandwidth = - (sr / np.pi) * np.log(radius)
        
        # Filter: retain realistically sharp peaks 
        # (Hz > 90 to avoid DC artifacts, bandwidth < 400 for strong resonances)
        if 90 < freq < (sr / 2) and bandwidth < 400:
            formants.append(freq)

    return np.sort(formants)

def process_dynamic_formants(frames, sr, lpc_order, rms_threshold=1e-3):
    """
    Processes all frames to generate temporal formant tracks.
    """
    temporal_tracks = []
    
    for frame in frames:

        if np.sqrt(np.mean(frame**2)) < rms_threshold:
            continue # Skip frames rather than appending NaNs
            
        coeffs = extract_lpc_coefficients(frame, lpc_order)
        formants = find_formants_from_lpc(coeffs, sr)
        
        # capture the first three formants
        if len(formants) >= 3:
            temporal_tracks.append([formants[0], formants[1], formants[2]])
        elif len(formants) == 2:
            # Simple padding if F3 is missing in a single frame
            temporal_tracks.append([formants[0], formants[1], 0])
            
    return np.array(temporal_tracks)
