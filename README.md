# Cross-Linguistic Phoneme Analysis via Short-Time LPC Formant Tracking and Dynamic Time Warping

## Overview
This project investigates the acoustic differences in vocal tract dynamics across different languages (English, Japanese, and Arabic) by analyzing words that sound phonetically similar. The goal is to quantify how the same word physically differs in pronunciation across native linguistic backgrounds using digital signal processing techniques.

## Core Objectives
- Apply **Short-Time Linear Predictive Coding (LPC)** to track dynamic formant frequencies ($F_1, F_2, F_3$) over time.
- Use **Dynamic Time Warping (DTW)** to calculate the "acoustic distance" between pronunciations, accounting for variations in speech duration.
- Visualize linguistic differences through spectrogram overlays, distance matrices, and articulatory vowel space plots.

## Project Structure
```
.
├── data/
│   └── raw/                # Audio files categorized by language (English, Japanese, Arabic)
├── reports/                # Generated visualizations (spectrograms, heatmaps, vowel spaces)
├── src/
│   ├── preprocessing.py    # Audio loading, pre-emphasis, framing, and VAD
│   ├── lpc_extraction.py   # LPC coefficient extraction and formant frequency calculation
│   ├── dtw_alignment.py    # Sequence alignment and distance computation
│   └── visualization.py    # Plotting utilities for analysis
├── main.py                 # Execution script for the full pipeline
└── requirements.txt        # Project dependencies
```

## Technical Implementation

### 1. Preprocessing (`src/preprocessing.py`)
- **Downsampling:** All audio is standardized to 16 kHz.
- **Pre-emphasis:** A high-pass filter ($y(t) = x(t) - 0.97x(t-1)$) is applied to boost higher frequencies and balance the spectrum.
- **Framing & Windowing:** Signal is divided into 25ms overlapping frames (10ms hop) with a **Hamming window**.
- **Voice Activity Detection (VAD):** Silent frames are filtered out based on an RMS energy threshold to ensure analysis is only performed on active speech.

### 2. Dynamic Formant Tracking (`src/lpc_extraction.py`)
- **LPC Analysis:** The vocal tract is modeled as an all-pole IIR filter. LPC coefficients are extracted for each frame.
- **Root-Finding:** Formant frequencies are derived by calculating the roots of the LPC predictor polynomial.
- **Filtering:** Peaks are filtered by bandwidth (< 400 Hz) and frequency (> 90 Hz) to ensure they represent meaningful resonances.

### 3. Sequence Alignment (`src/dtw_alignment.py`)
- **Dynamic Time Warping:** Since the same word varies in duration across languages, DTW is used to align the temporal formant sequences.
- **Distance Metric:** Computes the Euclidean distance between aligned tracks, normalized by the alignment path length to provide a scalar "acoustic distance."

### 4. Visualization (`src/visualization.py`)
- **Spectrogram Overlays:** F1, F2, and F3 tracks plotted directly over audio spectrograms.
- **Distance Matrices:** Heatmaps showing the pairwise acoustic distances between languages for a specific word.
- **Vowel Space Plots:** F1 vs. F2 plots showing the articulatory trajectory and centroids, providing insight into jaw openness and tongue advancement.

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
To process the audio files in `data/raw/` and generate reports:
```bash
python main.py
```

## Data Source
The project utilizes self-recorded speech data of "anchor words" (e.g., *camera*, *broccoli*, *radio*, *pajama*, *tomato*, *telephone*, *chocolate*, *tennis*) spoken in English, Japanese, and Arabic.
