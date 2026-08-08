# Neuriphonium

**Your brain is the instrument.**

Neuriphonium is a real-time EEG-to-music system that translates your mental states into live, evolving musical compositions. Using a MUSE headband, it captures brainwave activity across five frequency bands, derives cognitive metrics (arousal, valence, focus, relaxation), and maps them to musical parameters that drive AI-powered music generation.

The system supports two generation engines:

- **Local engine** — [Magenta RealTime 2](https://github.com/magenta/magenta-rt) (MRT2) running on-device via Apple MLX, with 4-bit quantized models for real-time inference
- **Online engine** — Google Lyria RealTime API for cloud-based synthesis

As your mental state shifts, so does the music. A moment of deep focus brings driving rhythms and clarity. A wave of relaxation unfolds into ambient textures and gentle melodies. The music becomes a mirror of your inner world.

---

## Features

- **Real-time EEG processing** — FFT-based extraction of delta, theta, alpha, beta, and gamma band powers from 4-channel MUSE headband data
- **Personal calibration** — 10-second baseline capture with z-score deviation tracking for personalized state detection
- **Dual music engines** — Switch between local Magenta RT2 (on-device, no API key needed) and online Google Lyria
- **Neuroscience-grounded mapping** — Each band controls multiple musical dimensions: BPM, density, brightness, groove, harmonic complexity, legato, register, dynamics, and more
- **Stable tonality & genre** — Hysteresis-based key and genre selection prevents jittery changes; only sustained mental state shifts trigger tonal transitions
- **Interactive web dashboard** — Live EEG visualization, band power charts, cognitive metrics, and real-time music controls
- **Motion sensor dashboard** — Separate visualization for MUSE accelerometer and gyroscope data
- **EEG fingerprinting** — LaBraM embeddings + UMAP visualization to test whether different music genres produce separable neural signatures
- **Cross-modal alignment training** — CLIP-style contrastive learning aligns EEG representations with MusicCoCa style space using the NMED-T dataset
- **Model benchmarking** — Built-in tools to compare inference speed across MRT2 model sizes (small, base, base_fast)

---

## Architecture

```
MUSE Headband (256 Hz, 4 ch)
        │
        ▼
   muselsl / pylsl
        │
        ▼
   EEGProcessor
   ├── FFT band power extraction (δ, θ, α, β, γ)
   ├── Personal baseline calibration (z-score deviations)
   ├── Cognitive metric derivation (arousal, valence, focus, relaxation)
   └── Neuroscience-grounded mapping → musical parameters
        │
        ▼
   Music Generator
   ├── Local: Magenta RT2 (MLX, 4-bit quantized)
   └── Online: Google Lyria RealTime API
        │
        ▼
   WebSocket Server (port 8767)
        │
        ▼
   Browser Dashboard
   ├── Live EEG charts (Chart.js)
   ├── Cognitive metric gauges
   ├── Music controls (start/stop, engine, model, genre)
   └── Web Audio API playback
```

---

## How It Works

### 1. Brainwave Capture

The MUSE headband (MUSE 2 or MUSE S) streams 4-channel EEG data (TP9, AF7, AF8, TP10) at 256 Hz via the LSL (Lab Streaming Layer) protocol using `muselsl`.

### 2. Signal Processing

An FFT extracts power in five frequency bands:

| Band | Frequency | Cognitive Association | Musical Effect |
|------|-----------|----------------------|----------------|
| Delta | 0.5–4 Hz | Deep/unconscious | Register, sustain, drone depth |
| Theta | 4–8 Hz | Creativity / DMN | Harmonic complexity, rubato, space |
| Alpha | 8–13 Hz | Flow / relaxation | Groove, legato, consonance |
| Beta | 13–30 Hz | Focus / motor | Rhythmic precision, density, tempo |
| Gamma | 30–50 Hz | Binding / insight | Brightness, ornamentation, tension |

### 3. Cognitive Metrics

From band powers, the system derives four cognitive metrics:

- **Arousal** — Weighted sum across all bands (gamma-weighted)
- **Valence** — Alpha dominance over theta
- **Focus** — Beta/theta ratio
- **Relaxation** — Alpha/beta ratio

When calibrated, these metrics use **z-score deviations** from your personal baseline instead of raw values, making the system adaptive to your individual brain patterns.

### 4. Musical Parameter Mapping

The cognitive metrics and band powers map to continuous musical parameters:

- **BPM** (60–170) — driven by arousal + beta
- **Density** (0.05–1.0) — note count per bar, driven by beta/arousal
- **Brightness** (0–1) — spectral content, driven by gamma/valence
- **Groove** (0–1) — swing factor, driven by alpha
- **Harmonic complexity** (0–1) — chord extensions, driven by theta
- **Legato** (0–1) — note connectivity, alpha vs beta
- **Register** (0–1) — low/high tessitura, delta vs gamma
- **Dynamics** (0–1) — loudness variation, driven by arousal
- **Temperature** (0.3–2.0) — generation creativity, driven by theta
- **Guidance** (2.5–6.5) — CFG scale, driven by focus/arousal

**Tonality** (key/scale) and **genre** use hysteresis: a candidate change must persist for 8–12 seconds before committing, preventing flickering from transient brainwave fluctuations.

### 5. AI Music Generation

- **Local (Magenta RT2)**: Runs on Apple Silicon via MLX. Supports `mrt2_small`, `mrt2_base`, and `mrt2_base_fast` (4-bit quantized, single CFG) models. Generates 40ms frames of 48kHz stereo audio in real-time.
- **Online (Lyria)**: Uses Google's Lyria RealTime API via `google-genai`. Requires a `GEMINI_API_KEY` with Lyria access.

### 6. Live Playback

Audio streams to the browser via WebSocket as binary chunks. The Web Audio API handles gapless playback with a ring buffer. The dashboard provides real-time visualization of EEG bands, cognitive metrics, and music state.

---

## Quick Start

### Prerequisites

- Python 3.10+
- MUSE headband (MUSE 2 or MUSE S)
- **For local engine**: Apple Silicon Mac (M1+) with MLX support
- **For online engine**: Google Cloud API key with Lyria access

### Installation

```bash
git clone https://github.com/ebaenamar/neuriphonium.git
cd neuriphonium
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Only needed for the online (Lyria) engine. The local (Magenta RT2) engine works without an API key.

### Running the Main Dashboard

1. Start MUSE EEG streaming:

```bash
muselsl stream
```

2. Launch the dashboard server:

```bash
python eeg_music_dashboard.py
```

3. Open `music_dashboard.html` in your browser (or navigate to `http://localhost:8767`)

4. **Calibrate** — Click the calibrate button and stay relaxed for 10 seconds to capture your baseline

5. **Select engine** — Choose Local (Magenta) or Online (Lyria), and pick a model size if using local

6. Click **Start Music** and let your mind play

### Running the Motion Sensor Dashboard

```bash
muselsl stream -c -g   # stream with motion sensors enabled
python motion_sensor_dashboard.py
```

Opens a separate dashboard at `http://localhost:8768` showing accelerometer and gyroscope data.

---

## Dashboard Controls

### EEG Controls

- **Band Sensitivity** (0x–3x) — Amplify or attenuate each frequency band's influence on the music
- **Smoothing** (1–20) — Higher values produce more stable output; lower values are more reactive
- **Global Sensitivity** — Master multiplier for all band sensitivities
- **Calibrate** — Capture a 10-second personal baseline for z-score-based state detection
- **Reset Baseline** — Clear calibration data
- **Genre Override** — Manually set a genre, or leave on Auto for EEG-driven selection

### Engine Controls

- **Engine selector** — Switch between Local (Magenta RT2) and Online (Lyria)
- **Model selector** (local only) — Choose between `mrt2_small`, `mrt2_base`, `mrt2_base_fast`
- **Start/Stop Music** — Control real-time generation

---

## Project Structure

```
neuriphonium/
├── eeg_music_dashboard.py        # Main dashboard server (WebSocket + EEG + music)
├── music_dashboard.html          # Web UI for the main dashboard
├── muse_adapter.py               # MUSE EEG adapter (streaming + CSV playback)
├── motion_sensor_dashboard.py    # Motion sensor visualization dashboard
├── motion_dashboard.html         # Motion sensor web UI
├── eeg_lyria_live.py             # Standalone live generator (Lyria, with audio playback)
├── eeg_lyria_realtime.py         # Standalone realtime generator (Lyria)
├── eeg_fingerprint.py            # EEG fingerprinting with LaBraM embeddings + UMAP
├── eeg_music_align/              # Cross-modal EEG-to-MusicCoCa alignment module
│   ├── model.py                  # LaBraM backbone + projection head + CLIP loss
│   ├── dataset.py                # NMED-T paired EEG/audio dataset loader
│   └── musiccoca_embed.py        # MusicCoCa audio embedding extractor with cache
├── train_align.py                # Training script for EEG → MusicCoCa alignment
├── benchmark_models.py           # Benchmark MRT2 model inference speed
├── export_fast_model.py          # Re-export MRT2 with 4-bit quantization
├── test_magenta_realtime.py      # Test local Magenta RT2 generation
├── test_ws_pipeline.py           # Test WebSocket pipeline latency
├── test_cold_pipeline.py         # Cold-start generation timing test
├── test_lyria3.py                # Test Lyria generation
├── test_lyria_conn.py            # Test Lyria API connection
├── paper/                        # Research paper (LaTeX + figures)
│   ├── neuriphonium.tex
│   ├── architecture.png
│   └── make_architecture.py
├── output/                       # Saved audio files (gitignored)
├── requirements.txt
└── .env                          # API keys (gitignored)
```

---

## Tech Stack

- **EEG Hardware**: MUSE 2 / MUSE S headband (4 channels, 256 Hz)
- **EEG Streaming**: muselsl + pylsl (LSL protocol)
- **Signal Processing**: NumPy FFT + scipy
- **Local Music Generation**: Magenta RealTime 2 (MLX, Apple Silicon)
- **Online Music Generation**: Google Lyria RealTime API (google-genai)
- **Backend**: Python asyncio + WebSockets (port 8767)
- **Frontend**: Vanilla JS + Chart.js + Web Audio API
- **EEG Embeddings**: LaBraM (braindecode) for fingerprinting and alignment
- **Cross-modal Training**: PyTorch + CLIP-style contrastive loss
- **Audio Output**: 48kHz stereo, WAV export

---

## Research Components

### EEG Fingerprinting (`eeg_fingerprint.py`)

Extracts LaBraM embeddings from MUSE EEG recordings collected while listening to different music genres (trash metal, gregorian chant, silence). Uses UMAP dimensionality reduction to visualize whether neural signatures are separable by genre. Includes k-NN classification with cross-validation.

### Cross-Modal Alignment (`eeg_music_align/` + `train_align.py`)

Trains a LaBraM-based alignment head to project EEG representations into the 768-dim MusicCoCa style space using a CLIP-style symmetric contrastive loss. Trained on the NMED-T dataset (EEG recordings of subjects listening to 10 songs). Evaluation uses leave-songs-out retrieval with Recall@K metrics.

### Paper

The `paper/` directory contains a LaTeX write-up of the system architecture and research findings, including an architecture diagram generated programmatically via `make_architecture.py`.

---

## License

MIT

---

*Built with brainwaves. Your mind has never sounded so good.*
