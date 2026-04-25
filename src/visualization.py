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
        
        if formant_tracks.shape[1] > 2:
            plt.scatter(time_bins, formant_tracks[:, 2], color='yellow', label='F3', s=5, alpha=0.8)
            
        plt.legend()
        
    plt.title(title)
    plt.ylim(0, 4000)
    plt.tight_layout()
    plt.savefig(f"reports/{title.replace(' ', '_')}.png")
    plt.close()

def plot_distance_matrix(matrix, labels, title="Cross-Lingual Acoustic Distances", vmin=0, vmax=3000):
    """
    Plots the computed DTW distance matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='DTW Distance')
    
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            # Use the fixed vmax for the contrast check
            color = 'white' if val < (vmax / 2) else 'black'
            if i == j: color = 'black'
            
            # Handle Infinity display if you haven't filtered it out
            text_val = "inf" if np.isinf(val) else f"{val:.2f}"
            plt.text(j, i, text_val, ha='center', va='center', color=color)
            
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"reports/{title.replace(' ', '_').replace(':', '')}.png")
    plt.close()

def plot_vowel_space(word_tracks, word_title):
    """
    Plots the F1 vs F2 acoustic space for a given word across languages.
    """
    plt.figure(figsize=(8, 6))
    
    # Assign consistent colors for the languages
    colors = {'English': 'cyan', 'Japanese': 'magenta', 'Arabic': 'yellow'}
    
    for lang, tracks in word_tracks.items():
        if tracks is None or len(tracks) == 0:
            continue
            
        # Filter out NaNs, zeros, or malformed frames before plotting
        valid_idx = ~np.isnan(tracks[:, 0]) & ~np.isnan(tracks[:, 1]) & (tracks[:, 0] > 0) & (tracks[:, 1] > 0)
        f1 = tracks[valid_idx, 0]
        f2 = tracks[valid_idx, 1]
        
        if len(f1) == 0:
            continue
            
        color = colors.get(lang, 'gray')
        
        # Plot the individual frames as small, semi-transparent dots
        plt.scatter(f2, f1, alpha=0.4, label=f"{lang} (frames)", c=color, s=20)
        
        # Calculate and plot the centroid (mean F1, F2) as a large marker
        mean_f1, mean_f2 = np.mean(f1), np.mean(f2)
        plt.scatter(mean_f2, mean_f1, marker='X', s=200, edgecolors='black', c=color, label=f"{lang} Centroid")

    plt.title(f"Acoustic Articulatory Space: {word_title.capitalize()}")
    plt.xlabel("F2 Frequency (Hz) - Tongue Advance (Front -> Back)")
    plt.ylabel("F1 Frequency (Hz) - Jaw Openness (Closed -> Open)")
    
    # Invert axes to match standard acoustic-phonetic mapping
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    # Limit the axes to standard human speech ranges
    plt.xlim(3000, 500)
    plt.ylim(1200, 200)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Prevent duplicate legends for the centroids vs frames
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(f"reports/vowel_space_{word_title}.png")
    plt.close()