import argparse
import os
import glob
from src.preprocessing import load_and_preprocess, frame_signal
from src.lpc_extraction import process_dynamic_formants
from src.dtw_alignment import compute_distance_matrix
from src.visualization import plot_distance_matrix, plot_spectrogram_with_formants

def main():
    print("EC519 Final Project: Dynamic Formant Tracking and DTW")
    
    # audio data located in data/raw/<Language>/*.wav
    languages = ['English', 'Japanese', 'Arabic']
    tracks_dict = {}
    
    # Pipeline hyperparameters
    target_sr = 16000
    lpc_order = 12
    
    for lang in languages:
        wav_files = glob.glob(f"data/raw/{lang}/*.wav")
        if not wav_files:
            print(f"Warning: No valid .wav files found for {lang} in data/raw/{lang}/")
            continue
            
        filepath = wav_files[0]
        print(f"Processing {lang}: {filepath}")
        
        # 1. Preprocessing
        signal, sr = load_and_preprocess(filepath, target_sr=target_sr)
        frames = frame_signal(signal, sr=sr)
        
        # Dynamic Feature Extraction
        formant_tracks = process_dynamic_formants(frames, sr, lpc_order)
        tracks_dict[lang] = formant_tracks
        
        # Visualization
        plot_spectrogram_with_formants(
            signal=signal, 
            sr=sr, 
            formant_tracks=formant_tracks, 
            title=f"{lang} Spectrogram and Formant Overlays"
        )
        
    # Cross-lingual Sequence Alignment (DTW)
    if len(tracks_dict) >= 2:
        print("\nComputing DTW distance matrix...")
        matrix, labels = compute_distance_matrix(tracks_dict)
        
        print("Generating distance matrix heatmap...")
        plot_distance_matrix(matrix, labels)
        print("\nAnalysis complete. Visualizations saved into the reports/ directory.")
    else:
        print("\nNot enough audio files to compute DTW matrix. Please place at least 2 .wav files in the data/raw/<Language>/ folders.")

if __name__ == "__main__":
    main()
