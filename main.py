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
    anchor_words = ['camera', 'broccoli', 'radio', 'pajama', 'tomato', 'telephone', 'chocolate', 'tennis']
    languages = ['English', 'Japanese', 'Arabic']

    
    # Pipeline hyperparameters
    target_sr = 16000
    lpc_order = 18 # 12 or 18? research

    for word in anchor_words:
        print(f"\n" + "="*30)
        print(f"ANALYZING WORD: {word.upper()}")
        print("="*30)
        
        word_tracks = {}
    
        for lang in languages:
            filepath = f"data/raw/{lang}/{word}.wav"
            
            if not os.path.exists(filepath):
                print(f"Skipping: {filepath} (File not found)")
                continue
            print(f"Processing {lang}...")
        
            # Preprocessing
            signal, sr = load_and_preprocess(filepath, target_sr=target_sr)
            frames = frame_signal(signal, sr=sr)
        
            # Dynamic Feature Extraction
            formant_tracks = process_dynamic_formants(frames, sr, lpc_order)
            word_tracks[lang] = formant_tracks
        
            # Visualization
            plot_spectrogram_with_formants(
                signal=signal, 
                sr=sr, 
                formant_tracks=formant_tracks, 
                title=f"{lang} {word.capitalize()} Spectrogram and Formants"
            )
        
        # Cross-lingual Sequence Alignment (DTW) for each word
        if len(word_tracks) >= 2:
            print(f"Computing DTW distances for '{word}'...")
            matrix, labels = compute_distance_matrix(word_tracks)
            plot_distance_matrix(matrix, labels, title=f"Distance Matrix: {word}")
        else:
            print(f"Not enough data to compare languages for '{word}'.")

    print("\nAll words processed. Check the reports/ directory for results.")

if __name__ == "__main__":
    main()
