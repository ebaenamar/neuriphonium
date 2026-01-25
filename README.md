# 🧠🎵 Neuriphonium

**Your brain is the instrument.**

Neuriphonium transforms your thoughts and mental states into live music. Using real-time brainwave data from a MUSE headband, the system captures the subtle rhythms of your mind—focus, relaxation, energy, calm—and translates them into an ever-evolving musical composition powered by Google's Lyria AI.

No buttons. No controllers. Just you.

As your mental state shifts, so does the music. A moment of deep focus might bring driving rhythms and clarity. A wave of relaxation could unfold into ambient textures and gentle melodies. The music becomes a mirror of your inner world—an audible reflection of consciousness itself.

**Think. Feel. Listen.**

---

## ✨ Features

- **Real-time EEG Processing** - Captures delta, theta, alpha, beta, and gamma brainwaves
- **AI Music Generation** - Uses Google Lyria RealTime API for continuous music synthesis
- **Interactive Dashboard** - Web-based visualization with live controls
- **Adjustable Sensitivity** - Fine-tune how each brainwave band affects the music
- **Smoothing Controls** - Balance between reactive and stable musical output

## 🎛️ How It Works

1. **Brainwave Capture** - MUSE headband streams EEG data via LSL protocol
2. **Signal Processing** - FFT extracts power in each frequency band
3. **State Mapping** - Cognitive metrics (arousal, valence, focus, relaxation) are derived
4. **Music Parameters** - BPM, scale, density, brightness, and prompts are generated
5. **AI Synthesis** - Lyria creates real-time audio matching your mental state
6. **Live Playback** - Audio streams directly to your browser

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- MUSE headband (MUSE 2 or MUSE S)
- Google Cloud API key with Lyria access

### Installation

```bash
git clone https://github.com/ebaenamar/neuriphonium.git
cd neuriphonium
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### Running

1. Start MUSE streaming:
```bash
muselsl stream
```

2. Launch the dashboard:
```bash
python eeg_music_dashboard.py
```

3. Open `music_dashboard.html` in your browser

4. Click **Start Music** and let your mind play!

## 🎚️ Dashboard Controls

### Band Sensitivity (0x - 3x)

| Band | Frequency | Musical Effect |
|------|-----------|----------------|
| δ Delta | 0.5-4 Hz | Deep bass, slow tempo |
| θ Theta | 4-8 Hz | Dreamy atmosphere, pads |
| α Alpha | 8-13 Hz | Smooth melodies, relaxation |
| β Beta | 13-30 Hz | Energy, rhythm, focus |
| γ Gamma | 30-50 Hz | Brightness, high details |

### Stability
- **Smoothing** - Higher = more stable/constant output
- **Global Sensitivity** - Amplifies all changes

## 📁 Project Structure

```
neuriphonium/
├── eeg_music_dashboard.py   # Main dashboard server
├── music_dashboard.html     # Web UI (auto-generated)
├── eeg_lyria_live.py        # Standalone live generator
├── muse_adapter.py          # MUSE EEG adapter
├── requirements.txt         # Dependencies
└── output/                  # Saved audio files
```

## 🛠️ Tech Stack

- **EEG**: MUSE headband + muselsl + pylsl
- **AI Music**: Google Lyria RealTime API
- **Backend**: Python asyncio + WebSockets
- **Frontend**: Vanilla JS + Chart.js + Web Audio API

## 📄 License

MIT

---

*Built with 🧠 at a hackathon. Your mind has never sounded so good.*
