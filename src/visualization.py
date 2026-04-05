import matplotlib.pyplot as plt
import librosa
import numpy as np

def plot_spectrogram_with_formants(signal, sr, formant_tracks, title="Spectrogram and Formants"):
    """
    Overlays temporal formant tracks directly on the audio spectrogram.
    """
    # create spectrogram
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    
    # overlay formants
    if formant_tracks is not None and len(formant_tracks) > 0:
        num_frames = formant_tracks.shape[0]
        time_bins = np.linspace(0, len(signal) / sr, num_frames)
        
        plt.scatter(time_bins, formant_tracks[:, 0], color='cyan', label='F1', s=5, alpha=0.8)
        if formant_tracks.shape[1] > 1:
            plt.scatter(time_bins, formant_tracks[:, 1], color='magenta', label='F2', s=5, alpha=0.8)
            
        plt.legend()
        
    plt.title(title)
    plt.ylim(0, 4000) # formants primarily interesting < 4000 Hz
    plt.tight_layout()
    plt.savefig(f"reports/{title.replace(' ', '_')}.png")
    plt.close()

def plot_distance_matrix(matrix, labels):
    """
    Plots the computed DTW distance matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='DTW Distance')
    
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            # use white for dark backgrounds and black for light ones
            color = 'white' if val < np.max(matrix)/2 else 'black'
            if i == j: color = 'black'
            
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)
            
    plt.title("Cross-Lingual Acoustic Distances")
    plt.tight_layout()
    plt.savefig("reports/Distance_Matrix.png")
    plt.close()
