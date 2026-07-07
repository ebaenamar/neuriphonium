#!/usr/bin/env python3
"""
MUSE EEG Adapter - Adapter for MUSE headband (4 channels)
Handles real-time streaming and CSV file playback

MUSE Channels:
- TP9: Left ear (temporal)
- AF7: Left forehead (frontal)
- AF8: Right forehead (frontal)
- TP10: Right ear (temporal)
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, List
from collections import deque
import logging
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MUSE channel names
MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

# EEG frequency bands (Hz)
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


class MuseEEGAdapter:
    """
    Adapter for MUSE 4-channel EEG headband.
    
    Supports:
    - Real-time streaming via muselsl/pylsl
    - CSV file playback
    - Band power extraction
    - Neurological state mapping
    """
    
    def __init__(self, sample_rate: int = 256, smoothing_window: int = 5):
        self.sample_rate = sample_rate
        self.smoothing_window = smoothing_window
        
        # Band power history for smoothing
        self.band_history = {band: deque(maxlen=smoothing_window) for band in EEG_BANDS}
        
        # Channel mapping (MUSE specific)
        self.channels = MUSE_CHANNELS
        
        # Hemisphere grouping
        self.left_channels = ['TP9', 'AF7']
        self.right_channels = ['AF8', 'TP10']
        
    def extract_band_powers(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency band powers from EEG data using FFT.
        
        Args:
            eeg_data: numpy array of shape (channels, samples) or (samples,)
        
        Returns:
            Dictionary with normalized band powers
        """
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        band_powers = {}
        n_samples = eeg_data.shape[1]
        
        # Compute FFT
        fft_vals = np.fft.fft(eeg_data, axis=1)
        freqs = np.fft.fftfreq(n_samples, 1.0 / self.sample_rate)
        
        # Extract power in each band
        for band_name, (low_freq, high_freq) in EEG_BANDS.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(np.abs(fft_vals[:, band_mask]) ** 2)
            band_powers[band_name] = band_power
        
        # Normalize
        total_power = sum(band_powers.values())
        if total_power > 0:
            band_powers = {k: v / total_power for k, v in band_powers.items()}
        
        # Apply smoothing
        smoothed = self._smooth_bands(band_powers)
        
        return smoothed
    
    def _smooth_bands(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to band powers"""
        smoothed = {}
        for band, power in band_powers.items():
            self.band_history[band].append(power)
            smoothed[band] = np.mean(list(self.band_history[band]))
        return smoothed
    
    def calculate_cognitive_metrics(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate cognitive metrics from band powers.
        
        Returns:
            Dictionary with arousal, valence, focus, relaxation, etc.
        """
        delta = band_powers.get('delta', 0.2)
        theta = band_powers.get('theta', 0.2)
        alpha = band_powers.get('alpha', 0.2)
        beta = band_powers.get('beta', 0.2)
        gamma = band_powers.get('gamma', 0.2)
        
        # Arousal: high frequency activity
        arousal = (0.1 * delta + 0.2 * theta + 0.3 * alpha + 0.5 * beta + 0.7 * gamma)
        
        # Valence: alpha promotes positive, theta/beta imbalance = negative
        valence = alpha - 0.3 * theta - 0.2 * abs(beta - theta)
        valence = (valence + 0.5) / 1.0
        
        # Focus: beta/theta ratio
        focus = beta / (theta + 0.01)
        focus = np.clip(focus, 0, 2) / 2
        
        # Relaxation: alpha dominance
        relaxation = alpha / (beta + 0.01)
        relaxation = np.clip(relaxation, 0, 2) / 2
        
        # Cognitive load: beta + gamma vs alpha
        cognitive_load = (beta + 0.5 * gamma) / (alpha + 0.1)
        cognitive_load = np.clip(cognitive_load, 0, 1)
        
        return {
            'arousal': np.clip(arousal, 0, 1),
            'valence': np.clip(valence, 0, 1),
            'focus': np.clip(focus, 0, 1),
            'relaxation': np.clip(relaxation, 0, 1),
            'cognitive_load': np.clip(cognitive_load, 0, 1)
        }
    
    def map_to_music_params(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Map EEG band powers to Lyria musical parameters.
        
        Uses abstract musical controls instead of genre/style:
        - bpm, density, brightness, guidance, temperature
        - scale (tonality), mute_bass, mute_drums
        
        Returns:
            Dictionary with all Lyria-compatible parameters
        """
        metrics = self.calculate_cognitive_metrics(band_powers)
        
        delta = band_powers.get('delta', 0.2)
        theta = band_powers.get('theta', 0.2)
        alpha = band_powers.get('alpha', 0.2)
        beta = band_powers.get('beta', 0.2)
        gamma = band_powers.get('gamma', 0.2)
        
        arousal = metrics['arousal']
        valence = metrics['valence']
        focus = metrics['focus']
        relaxation = metrics['relaxation']
        
        # === LYRIA PARAMETERS ===
        
        # BPM: 60-200, driven by arousal
        bpm = int(60 + arousal * 140)
        
        # Density: 0-1, inverse of relaxation (relaxed = sparse)
        density = 1.0 - relaxation * 0.7
        
        # Brightness: 0-1, gamma + valence influence
        brightness = np.clip(gamma * 1.5 + valence * 0.3, 0, 1)
        
        # Guidance: 0-6, focus increases structure
        guidance = 2.0 + focus * 3.0
        
        # Temperature: 0-3, theta increases experimentation
        temperature = 0.8 + theta * 1.5
        
        # Scale selection based on valence + arousal
        scale = self._select_scale(valence, arousal)
        
        # Texture controls
        mute_drums = relaxation > 0.7  # Very relaxed = no drums
        mute_bass = delta > 0.4  # High delta = dreamy, less bass
        only_bass_and_drums = focus > 0.8 and arousal > 0.6  # Hyper-focused = minimal
        
        # Music generation mode
        if theta > 0.35:
            music_mode = 'DIVERSITY'  # Theta = more experimental
        elif focus > 0.6:
            music_mode = 'QUALITY'  # Focused = structured
        else:
            music_mode = 'QUALITY'
        
        music_params = {
            # Core EEG metrics
            'arousal': arousal,
            'valence': valence,
            'focus': focus,
            'relaxation': relaxation,
            'cognitive_load': metrics['cognitive_load'],
            
            # Lyria MusicGenerationConfig parameters
            'bpm': max(60, min(200, bpm)),
            'density': max(0, min(1, density)),
            'brightness': max(0, min(1, brightness)),
            'guidance': max(0, min(6, guidance)),
            'temperature': max(0, min(3, temperature)),
            'scale': scale,
            'mute_drums': mute_drums,
            'mute_bass': mute_bass,
            'only_bass_and_drums': only_bass_and_drums,
            'music_generation_mode': music_mode,
            
            # Raw band powers for reference
            'delta': delta,
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        }
        
        return music_params
    
    def _select_scale(self, valence: float, arousal: float) -> str:
        """
        Select musical scale/tonality based on emotional state.
        
        Valence: positive → major keys, negative → minor keys
        Arousal: high → brighter keys, low → darker keys
        
        Returns:
            Lyria Scale enum string
        """
        # Scale mapping by emotional quadrant
        # High valence + High arousal: Bright major keys
        # High valence + Low arousal: Warm major keys
        # Low valence + High arousal: Dramatic minor keys
        # Low valence + Low arousal: Melancholic minor keys
        
        if valence > 0.6:
            # Positive emotional state → Major keys
            if arousal > 0.6:
                # Energetic positive: D Major (bright, triumphant)
                return 'D_MAJOR_B_MINOR'
            elif arousal > 0.4:
                # Moderate positive: G Major (warm, happy)
                return 'G_MAJOR_E_MINOR'
            else:
                # Calm positive: C Major (neutral, peaceful)
                return 'C_MAJOR_A_MINOR'
        
        elif valence > 0.4:
            # Neutral emotional state
            if arousal > 0.5:
                # Active neutral: A Major
                return 'A_MAJOR_G_FLAT_MINOR'
            else:
                # Calm neutral: F Major (pastoral)
                return 'F_MAJOR_D_MINOR'
        
        else:
            # Contemplative/melancholic → Minor-leaning keys
            if arousal > 0.6:
                # Intense contemplative: E♭ Major/C minor (dramatic)
                return 'E_FLAT_MAJOR_C_MINOR'
            elif arousal > 0.4:
                # Moderate contemplative: B♭ Major/G minor
                return 'B_FLAT_MAJOR_G_MINOR'
            else:
                # Deep contemplative: A♭ Major/F minor (melancholic)
                return 'A_FLAT_MAJOR_F_MINOR'
    
    def generate_prompt_from_state(self, music_params: Dict[str, float]) -> str:
        """
        Generate a Lyria prompt based on current brain state.
        
        Uses abstract musical descriptors instead of genres:
        - Texture descriptors (sparse, dense, layered)
        - Tonal descriptors (warm, bright, dark)
        - Movement descriptors (flowing, pulsing, static)
        - Instrument families (not specific genres)
        
        Returns:
            Text prompt for Lyria music generation
        """
        arousal = music_params['arousal']
        valence = music_params['valence']
        relaxation = music_params['relaxation']
        focus = music_params.get('focus', 0.5)
        theta = music_params.get('theta', 0.2)
        gamma = music_params.get('gamma', 0.2)
        
        descriptors = []
        
        # === ENERGY/MOVEMENT ===
        if arousal > 0.7:
            descriptors.append("driving rhythmic pulse")
        elif arousal > 0.5:
            descriptors.append("steady flowing movement")
        elif arousal > 0.3:
            descriptors.append("gentle undulating waves")
        else:
            descriptors.append("slow sustained tones")
        
        # === TEXTURE ===
        if relaxation > 0.6:
            descriptors.append("sparse open textures")
        elif focus > 0.6:
            descriptors.append("tight precise patterns")
        else:
            descriptors.append("layered evolving textures")
        
        # === TONAL QUALITY ===
        if valence > 0.6 and gamma > 0.25:
            descriptors.append("bright shimmering tones")
        elif valence > 0.5:
            descriptors.append("warm rich harmonics")
        elif valence > 0.3:
            descriptors.append("neutral balanced tones")
        else:
            descriptors.append("deep resonant frequencies")
        
        # === HARMONIC MOVEMENT ===
        if theta > 0.3:
            descriptors.append("drifting harmonic progressions")
        elif focus > 0.6:
            descriptors.append("structured chord changes")
        else:
            descriptors.append("subtle harmonic shifts")
        
        # === INSTRUMENT TEXTURE (abstract) ===
        if music_params.get('mute_drums', False):
            if relaxation > 0.5:
                descriptors.append("sustained pads and drones")
            else:
                descriptors.append("melodic instruments without percussion")
        elif music_params.get('only_bass_and_drums', False):
            descriptors.append("minimal bass and rhythmic elements")
        else:
            if arousal > 0.6:
                descriptors.append("full ensemble with rhythmic drive")
            else:
                descriptors.append("balanced instrumental blend")
        
        prompt = ", ".join(descriptors)
        
        return prompt
    
    def get_weighted_prompts(self, music_params: Dict[str, float]) -> list:
        """
        Generate weighted prompts for Lyria based on EEG state.
        
        Returns list of (text, weight) tuples for nuanced control.
        """
        prompts = []
        
        arousal = music_params['arousal']
        valence = music_params['valence']
        relaxation = music_params['relaxation']
        theta = music_params.get('theta', 0.2)
        alpha = music_params.get('alpha', 0.2)
        beta = music_params.get('beta', 0.2)
        gamma = music_params.get('gamma', 0.2)
        
        # Main prompt from state
        main_prompt = self.generate_prompt_from_state(music_params)
        prompts.append((main_prompt, 1.0))
        
        # Add band-specific texture prompts with weights
        if alpha > 0.3:
            prompts.append(("smooth flowing melodies", alpha))
        
        if theta > 0.25:
            prompts.append(("dreamy ethereal atmosphere", theta * 1.5))
        
        if beta > 0.3:
            prompts.append(("precise articulated notes", beta))
        
        if gamma > 0.25:
            prompts.append(("sparkling high frequency details", gamma * 2))
        
        # Emotional color
        if valence > 0.6:
            prompts.append(("uplifting ascending phrases", valence - 0.4))
        elif valence < 0.4:
            prompts.append(("introspective descending motifs", 0.6 - valence))
        
        return prompts


class MuseLSLStream:
    """Real-time MUSE streaming via LSL (Lab Streaming Layer)"""
    
    def __init__(self, adapter: MuseEEGAdapter):
        self.adapter = adapter
        self.inlet = None
        self.is_streaming = False
        
    def connect(self) -> bool:
        """Connect to MUSE LSL stream"""
        try:
            from pylsl import StreamInlet, resolve_byprop
            
            logger.info("🔍 Looking for MUSE LSL stream...")
            streams = resolve_byprop('type', 'EEG', timeout=5)
            
            if not streams:
                logger.error("❌ No EEG stream found. Make sure MUSE is connected via muselsl.")
                logger.info("   Run: muselsl stream")
                return False
            
            self.inlet = StreamInlet(streams[0])
            logger.info(f"✅ Connected to MUSE stream: {streams[0].name()}")
            return True
            
        except ImportError:
            logger.error("❌ pylsl not installed. Run: pip install pylsl")
            return False
        except Exception as e:
            logger.error(f"❌ Error connecting to MUSE: {e}")
            return False
    
    def start_streaming(self, callback: Callable[[Dict[str, float]], None], 
                       window_size: int = 256):
        """
        Start streaming EEG data and call callback with music parameters.
        
        Args:
            callback: Function to call with music_params dict
            window_size: Number of samples per analysis window
        """
        if not self.inlet:
            if not self.connect():
                return
        
        self.is_streaming = True
        buffer = []
        
        logger.info("🎵 Starting MUSE EEG stream...")
        
        while self.is_streaming:
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample:
                    buffer.append(sample[:4])  # Only first 4 channels
                    
                    if len(buffer) >= window_size:
                        # Convert to numpy array
                        eeg_data = np.array(buffer).T  # (channels, samples)
                        
                        # Extract band powers and map to music
                        band_powers = self.adapter.extract_band_powers(eeg_data)
                        music_params = self.adapter.map_to_music_params(band_powers)
                        
                        # Call callback
                        callback(music_params)
                        
                        # Keep half the buffer for overlap
                        buffer = buffer[window_size // 2:]
                        
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                time.sleep(0.1)
    
    def stop_streaming(self):
        """Stop the EEG stream"""
        self.is_streaming = False
        logger.info("⏹️ MUSE stream stopped")


class MuseCSVStream:
    """Playback MUSE data from CSV file"""
    
    def __init__(self, adapter: MuseEEGAdapter):
        self.adapter = adapter
        self.is_streaming = False
        
    def stream_from_csv(self, csv_path: str, callback: Callable[[Dict[str, float]], None],
                       window_size: int = 256, delay: float = 1.0):
        """
        Stream EEG data from CSV file.
        
        Args:
            csv_path: Path to CSV file with EEG data
            callback: Function to call with music_params dict
            window_size: Number of samples per analysis window
            delay: Delay between callbacks in seconds
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded {len(df)} samples from {csv_path}")
        except Exception as e:
            logger.error(f"❌ Error reading CSV: {e}")
            return
        
        # Detect column format
        if 'TP9' in df.columns:
            # Standard MUSE format
            channels = ['TP9', 'AF7', 'AF8', 'TP10']
        elif 'EEG.TP9' in df.columns:
            # Muse Monitor format
            channels = ['EEG.TP9', 'EEG.AF7', 'EEG.AF8', 'EEG.TP10']
        elif 'RAW_TP9' in df.columns:
            # Raw format
            channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        else:
            # Try first 4 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            if len(numeric_cols) >= 4:
                channels = list(numeric_cols)
                logger.warning(f"⚠️ Using columns: {channels}")
            else:
                logger.error("❌ Could not detect EEG channels in CSV")
                return
        
        self.is_streaming = True
        logger.info(f"🎵 Starting CSV playback with {delay}s delay...")
        
        for i in range(0, len(df) - window_size, window_size // 2):
            if not self.is_streaming:
                break
            
            try:
                # Extract window
                window = df[channels].iloc[i:i + window_size].values.T
                
                # Extract band powers and map to music
                band_powers = self.adapter.extract_band_powers(window)
                music_params = self.adapter.map_to_music_params(band_powers)
                
                # Log state
                logger.info(
                    f"🧠 Arousal={music_params['arousal']:.2f} | "
                    f"Valence={music_params['valence']:.2f} | "
                    f"BPM={music_params['bpm']}"
                )
                
                # Call callback
                callback(music_params)
                
                time.sleep(delay)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Playback stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error processing window: {e}")
                continue
        
        self.is_streaming = False
        logger.info("✅ CSV playback complete")
    
    def stop_streaming(self):
        """Stop CSV playback"""
        self.is_streaming = False


def test_muse_adapter():
    """Test the MUSE adapter with synthetic data"""
    print("🧠 MUSE EEG Adapter Test")
    print("=" * 50)
    
    adapter = MuseEEGAdapter()
    
    # Generate synthetic EEG data (simulating different states)
    test_states = {
        'Relaxed (high alpha)': {
            'alpha_boost': 2.0, 'beta_boost': 0.5
        },
        'Focused (high beta)': {
            'alpha_boost': 0.5, 'beta_boost': 2.0
        },
        'Meditative (high theta)': {
            'alpha_boost': 1.0, 'beta_boost': 0.3, 'theta_boost': 2.0
        },
    }
    
    for state_name, boosts in test_states.items():
        print(f"\n{state_name}:")
        
        # Generate synthetic signal
        t = np.linspace(0, 1, 256)
        signal = np.zeros((4, 256))
        
        for ch in range(4):
            # Base frequencies
            signal[ch] += 0.5 * np.sin(2 * np.pi * 2 * t)  # Delta
            signal[ch] += 0.5 * np.sin(2 * np.pi * 6 * t) * boosts.get('theta_boost', 1.0)  # Theta
            signal[ch] += 1.0 * np.sin(2 * np.pi * 10 * t) * boosts.get('alpha_boost', 1.0)  # Alpha
            signal[ch] += 0.5 * np.sin(2 * np.pi * 20 * t) * boosts.get('beta_boost', 1.0)  # Beta
            signal[ch] += 0.2 * np.sin(2 * np.pi * 40 * t)  # Gamma
            signal[ch] += np.random.randn(256) * 0.1  # Noise
        
        # Extract and map
        band_powers = adapter.extract_band_powers(signal)
        music_params = adapter.map_to_music_params(band_powers)
        prompt = adapter.generate_prompt_from_state(music_params)
        
        print(f"  Bands: α={band_powers['alpha']:.2f}, β={band_powers['beta']:.2f}, θ={band_powers['theta']:.2f}")
        print(f"  Arousal={music_params['arousal']:.2f}, Valence={music_params['valence']:.2f}")
        print(f"  BPM={music_params['bpm']}, Density={music_params['density']:.2f}")
        print(f"  🎵 Prompt: {prompt}")


class MuseConnectionManager:
    """
    Gestor de conexión MUSE que maneja todos los sensores disponibles:
    - EEG (4 canales, 256 Hz)
    - Accelerometer (3 ejes X/Y/Z, 52 Hz, unidades: g)
    - Gyroscope (3 ejes X/Y/Z, 52 Hz, unidades: dps)
    
    Características:
    - Descubrimiento automático de streams LSL
    - Conexión/desconexión por sensor
    - Reconexión automática con backoff
    - Buffers circulares con historial configurable
    - Callbacks por sensor para consumidores
    - Reporte de estado y estadísticas en tiempo real
    """
    
    SENSOR_TYPES = {
        'eeg':  {'lsl_type': 'EEG',  'channels': 4, 'rate': 256, 'units': 'uV'},
        'acc':  {'lsl_type': 'ACC',  'channels': 3, 'rate': 52,  'units': 'g'},
        'gyro': {'lsl_type': 'GYRO', 'channels': 3, 'rate': 52,  'units': 'dps'},
    }
    
    def __init__(self, history_seconds=10):
        self.history_seconds = history_seconds
        self.inlets = {}
        self.threads = {}
        self.running = False
        self.connected = {}
        self.sample_counts = {}
        self.last_sample_ts = {}
        self.start_time = None
        
        self._buffers = {}
        self._callbacks = {}
        self._lock = threading.Lock()
        
        for sensor in self.SENSOR_TYPES:
            self._buffers[sensor] = deque(maxlen=history_seconds * self.SENSOR_TYPES[sensor]['rate'])
            self._callbacks[sensor] = None
            self.connected[sensor] = False
            self.sample_counts[sensor] = 0
            self.last_sample_ts[sensor] = None
    
    def discover(self) -> Dict[str, bool]:
        """Descubre qué streams LSL de MUSE están disponibles."""
        from pylsl import resolve_byprop
        available = {}
        for sensor, info in self.SENSOR_TYPES.items():
            streams = resolve_byprop('type', info['lsl_type'], timeout=3)
            available[sensor] = len(streams) > 0
            if streams:
                logger.info(f"  ✅ {sensor.upper()}: {streams[0].name()} ({streams[0].nominal_srate()} Hz, {streams[0].channel_count()} ch)")
            else:
                logger.info(f"  ❌ {sensor.upper()}: no stream found")
        return available
    
    def connect_sensor(self, sensor: str) -> bool:
        """Conecta a un sensor específico via LSL."""
        if sensor not in self.SENSOR_TYPES:
            logger.error(f"Unknown sensor: {sensor}")
            return False
        
        info = self.SENSOR_TYPES[sensor]
        from pylsl import StreamInlet, resolve_byprop
        
        streams = resolve_byprop('type', info['lsl_type'], timeout=5)
        if not streams:
            logger.warning(f"⚠️ No {sensor.upper()} stream found")
            self.connected[sensor] = False
            return False
        
        self.inlets[sensor] = StreamInlet(streams[0])
        self.connected[sensor] = True
        logger.info(f"✅ {sensor.upper()} connected: {streams[0].name()} ({streams[0].nominal_srate()} Hz)")
        return True
    
    def connect_all(self) -> Dict[str, bool]:
        """Conecta a todos los sensores disponibles."""
        logger.info("🔍 Discovering MUSE streams...")
        available = self.discover()
        results = {}
        for sensor in self.SENSOR_TYPES:
            if available[sensor]:
                results[sensor] = self.connect_sensor(sensor)
            else:
                results[sensor] = False
        return results
    
    def set_callback(self, sensor: str, callback: Callable[[list, float], None]):
        """Registra un callback para un sensor. Se llama con (sample_list, timestamp)."""
        if sensor in self.SENSOR_TYPES:
            self._callbacks[sensor] = callback
    
    def start(self):
        """Inicia la recolección de datos en hilos separados."""
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        
        for sensor in self.SENSOR_TYPES:
            if self.connected.get(sensor) and self.inlets.get(sensor):
                t = threading.Thread(target=self._collect_loop, args=(sensor,), daemon=True)
                self.threads[sensor] = t
                t.start()
                logger.info(f"▶️ {sensor.upper()} collection started")
    
    def stop(self):
        """Detiene toda la recolección."""
        self.running = False
        for sensor in self.threads:
            self.connected[sensor] = False
        for t in self.threads.values():
            if t.is_alive():
                t.join(timeout=2)
        self.threads.clear()
        self.inlets.clear()
        logger.info("⏹️ All sensors stopped")
    
    def _collect_loop(self, sensor: str):
        """Hilo de recolección para un sensor."""
        inlet = self.inlets[sensor]
        while self.running and self.connected.get(sensor):
            try:
                sample, ts = inlet.pull_sample(timeout=1.0)
                if sample:
                    with self._lock:
                        self._buffers[sensor].append({
                            'data': list(sample),
                            'ts': ts or time.time()
                        })
                        self.sample_counts[sensor] += 1
                        self.last_sample_ts[sensor] = ts or time.time()
                    
                    cb = self._callbacks.get(sensor)
                    if cb:
                        cb(list(sample), ts or time.time())
            except Exception as e:
                logger.error(f"❌ {sensor.upper()} collection error: {e}")
                time.sleep(0.5)
                # Intentar reconectar
                if self.running:
                    logger.info(f"🔄 Reconnecting {sensor.upper()}...")
                    if self.connect_sensor(sensor):
                        inlet = self.inlets[sensor]
                    else:
                        self.connected[sensor] = False
                        break
    
    def get_buffer(self, sensor: str, n: int = None) -> list:
        """Obtiene las últimas n muestras de un sensor."""
        with self._lock:
            buf = list(self._buffers.get(sensor, []))
        if n:
            return buf[-n:]
        return buf
    
    def get_status(self) -> dict:
        """Retorna el estado completo de todas las conexiones."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        status = {
            'running': self.running,
            'elapsed': elapsed,
            'sensors': {}
        }
        for sensor in self.SENSOR_TYPES:
            info = self.SENSOR_TYPES[sensor]
            buf = self._buffers.get(sensor, [])
            with self._lock:
                buf_list = list(buf)
            
            sensor_status = {
                'connected': self.connected.get(sensor, False),
                'type': info['lsl_type'],
                'channels': info['channels'],
                'nominal_rate': info['rate'],
                'units': info['units'],
                'samples_received': self.sample_counts.get(sensor, 0),
                'measured_rate': self.sample_counts.get(sensor, 0) / elapsed if elapsed > 0 else 0,
                'buffer_size': len(buf_list),
                'last_sample_age': time.time() - self.last_sample_ts[sensor] if self.last_sample_ts.get(sensor) else None,
            }
            
            if buf_list:
                data = np.array([s['data'] for s in buf_list])
                sensor_status['latest'] = data[-1].tolist()
                sensor_status['mean'] = np.mean(data, axis=0).tolist()
                sensor_status['std'] = np.std(data, axis=0).tolist()
                sensor_status['min'] = np.min(data, axis=0).tolist()
                sensor_status['max'] = np.max(data, axis=0).tolist()
                sensor_status['range'] = (np.max(data, axis=0) - np.min(data, axis=0)).tolist()
                magnitudes = np.linalg.norm(data, axis=1)
                sensor_status['magnitude_mean'] = float(np.mean(magnitudes))
                sensor_status['magnitude_std'] = float(np.std(magnitudes))
                sensor_status['magnitude_latest'] = float(magnitudes[-1])
            
            status['sensors'][sensor] = sensor_status
        return status
    
    def get_series(self, sensor: str, n: int = 100) -> dict:
        """Obtiene series temporales para graficar: {x: [], y: [], z: []} o {ch0: [], ...}."""
        buf = self.get_buffer(sensor, n)
        if not buf:
            return {}
        data = np.array([s['data'] for s in buf])
        series = {}
        for i in range(data.shape[1]):
            axis = ['x', 'y', 'z'][i] if data.shape[1] <= 3 else f'ch{i}'
            series[axis] = data[:, i].tolist()
        return series


if __name__ == "__main__":
    test_muse_adapter()
