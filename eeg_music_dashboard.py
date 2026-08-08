#!/usr/bin/env python3
"""
EEG Music Dashboard - Interfaz web para visualización EEG + generación de música Lyria
Dashboard interactivo con controles para iniciar/parar generación de música
"""

import os
import asyncio
import json
import time
import wave
import numpy as np
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
import logging
import threading

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
SAMPLE_RATE_AUDIO = 48000
SAMPLE_RATE_EEG = 256
WINDOW_SIZE = 256
OUTPUT_DIR = "output"
CHANNELS = 2

EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


class EEGProcessor:
    """Procesador de señales EEG con detección de cambios y sensibilidad configurable"""
    
    def __init__(self, sensitivity=2.5):
        self.smoothing = 3
        self.band_history = {band: deque(maxlen=self.smoothing) for band in EEG_BANDS}
        self.sensitivity = sensitivity
        self.prev_metrics = None
        self.baseline_metrics = None
        self.baseline_samples = []
        self.current_style_idx = 0
        
        # Sensibilidad por banda (configurable desde UI)
        self.band_sensitivity = {
            'delta': 1.0,
            'theta': 1.0,
            'alpha': 1.0,
            'beta': 1.0,
            'gamma': 1.0
        }
        
        # Tonality state — stable, only changes on sustained mental state shifts
        self._current_key = 'C_MAJOR_A_MINOR'
        self._key_candidate = None
        self._key_candidate_since = 0
        self._key_hysteresis_seconds = 8
        
        # Genre state — stable, only changes on sustained mood shifts
        self._current_genre = 'cool jazz, laid back, mellow trumpet, instrumental'
        self._genre_candidate = None
        self._genre_candidate_since = 0
        self._genre_hysteresis_seconds = 12
        self._manual_genre = None  # When set, overrides EEG-driven genre
        
        # Per-band baseline — captured during calibration
        self._band_baseline = None  # {band: (mean, std)}
        self._calibrating = False
        self._calibration_samples = []
        self._calibration_start = 0
    
    def update_settings(self, settings: dict):
        """Actualizar configuración desde la UI"""
        if 'band_sensitivity' in settings:
            self.band_sensitivity.update(settings['band_sensitivity'])
        if 'smoothing' in settings:
            new_smoothing = max(1, min(20, settings['smoothing']))
            if new_smoothing != self.smoothing:
                self.smoothing = new_smoothing
                self.band_history = {band: deque(maxlen=self.smoothing) for band in EEG_BANDS}
        if 'global_sensitivity' in settings:
            self.sensitivity = settings['global_sensitivity']
    
    def start_calibration(self):
        """Start 10-second baseline capture"""
        self._calibrating = True
        self._calibration_samples = []
        self._calibration_start = time.time()
        logger.info("📊 Starting EEG calibration (10s)...")
    
    def feed_calibration(self, bands: dict):
        """Feed band powers during calibration"""
        if self._calibrating:
            self._calibration_samples.append(bands.copy())
    
    def finish_calibration(self) -> bool:
        """Compute per-band baseline from calibration samples. Returns True if enough data."""
        if len(self._calibration_samples) < 5:
            logger.warning(f"⚠️ Not enough calibration samples ({len(self._calibration_samples)})")
            self._calibrating = False
            return False
        
        self._band_baseline = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            values = [s[band] for s in self._calibration_samples if band in s]
            if values:
                mean = np.mean(values)
                std = np.std(values) if len(values) > 1 else 0.01
                self._band_baseline[band] = (mean, max(std, 0.001))
        
        self._calibrating = False
        logger.info(f"✅ Baseline captured: {self._band_baseline}")
        return True
    
    @property
    def is_calibrated(self) -> bool:
        return self._band_baseline is not None
    
    @property
    def calibration_progress(self) -> float:
        if not self._calibrating:
            return 0
        return min(1.0, (time.time() - self._calibration_start) / 10.0)
    
    def get_deviations(self, bands: dict) -> dict:
        """Get z-score deviations from personal baseline for each band.
        Positive = above baseline, negative = below baseline.
        Returns dict of {band: z_score} clamped to [-3, 3]."""
        if not self._band_baseline:
            return {band: 0.0 for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
        
        deviations = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band in self._band_baseline:
                mean, std = self._band_baseline[band]
                z = (bands.get(band, mean) - mean) / std
                deviations[band] = np.clip(z, -3, 3)
            else:
                deviations[band] = 0.0
        return deviations
    
    def extract_bands(self, eeg_data: np.ndarray) -> dict:
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        band_powers = {}
        n_samples = eeg_data.shape[1]
        
        fft_vals = np.fft.fft(eeg_data, axis=1)
        freqs = np.fft.fftfreq(n_samples, 1.0 / SAMPLE_RATE_EEG)
        
        for band_name, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            power = np.mean(np.abs(fft_vals[:, mask]) ** 2)
            band_powers[band_name] = power
        
        total = sum(band_powers.values())
        if total > 0:
            band_powers = {k: v / total for k, v in band_powers.items()}
        
        # Aplicar sensibilidad por banda
        for band, power in band_powers.items():
            adjusted_power = power * self.band_sensitivity.get(band, 1.0)
            self.band_history[band].append(adjusted_power)
            band_powers[band] = np.mean(list(self.band_history[band]))
        
        return band_powers
    
    def calculate_metrics(self, bands: dict) -> dict:
        d, t, a, b, g = [bands.get(x, 0.2) for x in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
        
        # Get z-score deviations from personal baseline
        dev = self.get_deviations(bands)
        
        # Convert z-scores to 0-1 range (0 = baseline, 0.5 = 1 std above, 1 = 3 std above)
        def z_to_01(z):
            return np.clip((z + 3) / 6, 0, 1)
        
        # Use deviations for metrics when calibrated, raw values otherwise
        if self.is_calibrated:
            d_m, t_m, a_m, b_m, g_m = z_to_01(dev['delta']), z_to_01(dev['theta']), z_to_01(dev['alpha']), z_to_01(dev['beta']), z_to_01(dev['gamma'])
        else:
            d_m, t_m, a_m, b_m, g_m = d, t, a, b, g
        
        arousal = np.clip(0.1*d_m + 0.2*t_m + 0.3*a_m + 0.5*b_m + 0.7*g_m, 0, 1)
        valence = np.clip((a_m - 0.3*t_m - 0.2*abs(b_m-t_m) + 0.5), 0, 1)
        focus = np.clip(b_m / (t_m + 0.01) / 3, 0, 1)
        relaxation = np.clip(a_m / (b_m + 0.01) / 2, 0, 1)
        
        metrics = {
            'arousal': arousal, 'valence': valence, 'focus': focus, 'relaxation': relaxation,
            'calibrated': self.is_calibrated,
            'deviations': dev,
            **bands
        }
        
        # Detectar cambios significativos (relative to previous, not baseline)
        metrics['significant_change'] = False
        if self.prev_metrics:
            for key in ['arousal', 'valence']:
                if abs(metrics[key] - self.prev_metrics[key]) > 0.05:
                    metrics['significant_change'] = True
                    break
        
        self.prev_metrics = metrics.copy()
        return metrics
    
    def map_to_lyria(self, bands: dict, metrics: dict) -> dict:
        """
        Neuroscience-grounded continuous mapping.
        
        Each band controls multiple musical dimensions simultaneously:
        - Delta (0.5-4Hz): Deep/unconscious → register, sustain, drone depth
        - Theta (4-8Hz): Creativity/DMN → harmonic complexity, rubato, space
        - Alpha (8-13Hz): Flow/relaxed → groove, legato, consonance
        - Beta (13-30Hz): Focus/motor → rhythmic precision, density, tempo
        - Gamma (30-100Hz): Binding/insight → brightness, ornamentation, tension
        """
        arousal = metrics.get('arousal_adjusted', metrics['arousal'])
        valence = metrics.get('valence_adjusted', metrics['valence'])
        focus = metrics.get('focus_adjusted', metrics['focus'])
        relaxation = metrics.get('relaxation_adjusted', metrics['relaxation'])
        
        delta = bands.get('delta', 0.2)
        theta = bands.get('theta', 0.2)
        alpha = bands.get('alpha', 0.2)
        beta = bands.get('beta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        # === CONTINUOUS PARAMETERS ===
        
        # BPM: driven by arousal + beta (motor cortex activation)
        # Low arousal + high delta = slow (60-80 BPM)
        # High arousal + high beta = fast (120-160 BPM)
        beta_drive = beta / (alpha + 0.01)  # Beta vs alpha = tension vs release
        bpm = int(np.clip(60 + arousal * 80 + beta_drive * 20 - delta * 30, 50, 170))
        
        # Density: how many notes per bar
        # High beta = dense, high delta = sparse
        density = np.clip(0.15 + beta * 0.6 + arousal * 0.3 - delta * 0.5, 0.05, 1.0)
        
        # Brightness: spectral content
        # Gamma adds harmonics, valence shifts bright/dark
        brightness = np.clip(0.2 + gamma * 0.8 + valence * 0.3 - delta * 0.2, 0, 1)
        
        # Groove/Swing: alpha-driven (flow state = natural groove)
        # High alpha = more swing/shuffle, low alpha = straight/grid
        groove = np.clip(alpha * 1.5 - beta * 0.3, 0, 1)
        
        # Harmonic complexity: theta-driven (default mode = wandering)
        # High theta = extended chords, modal interchange, surprising turns
        harmonic_complexity = np.clip(0.1 + theta * 1.2 + (1 - focus) * 0.3, 0, 1)
        
        # Legato/Staccato: alpha vs beta
        # Alpha = smooth connected, beta = sharp detached
        legato = np.clip(alpha / (beta + 0.01) / 3, 0, 1)
        
        # Reverb/Space: relaxation + theta (spaciousness)
        space = np.clip(relaxation * 0.6 + theta * 0.5, 0, 1)
        
        # Register/Tessitura: delta pulls low, gamma pulls high
        register = np.clip(0.5 - delta * 0.8 + gamma * 0.5, 0, 1)  # 0=low, 1=high
        
        # Dynamics/Expression: arousal drives loudness variation
        dynamics = np.clip(0.2 + arousal * 0.8, 0, 1)
        
        # Guidance (CFG): focus + arousal push style adherence
        guidance = 2.5 + focus * 2.5 + arousal * 1.5
        
        # Temperature: theta/alpha contrast drives creativity
        temperature = np.clip(0.4 + theta * 1.5 + arousal * 0.5 - delta * 0.3, 0.3, 2.0)
        
        # === TONALITY (stable, hysteresis) ===
        scale = self._resolve_key(valence, arousal, focus, delta, theta, gamma, beta)
        
        # === GENRE (stable, hysteresis) ===
        if self._manual_genre:
            genre = self._manual_genre
        else:
            genre = self._resolve_genre(valence, arousal, theta, alpha, gamma, beta)
        
        return {
            'bpm': bpm,
            'density': density,
            'brightness': brightness,
            'groove': groove,
            'harmonic_complexity': harmonic_complexity,
            'legato': legato,
            'space': space,
            'register': register,
            'dynamics': dynamics,
            'guidance': guidance,
            'temperature': temperature,
            'scale': scale,
            'genre': genre,
            'mute_drums': False,
            'mute_bass': False,
            **metrics
        }
    
    def _resolve_key(self, valence, arousal, focus, delta, theta, gamma, beta) -> str:
        """
        Map mental state to musical key. Stable with hysteresis.
        
        Mental state → Key:
        - Calm + positive (alpha dominant, high valence) → C Major (pure, open)
        - Calm + neutral (balanced, moderate valence) → G Major (warm, pastoral)
        - Calm + dark (delta dominant, low valence) → A Minor (introspective)
        - Creative + positive (theta dominant, high valence) → D Major (bright, expansive)
        - Creative + neutral (theta dominant, moderate valence) → E Minor (mysterious, flowing)
        - Creative + dark (theta + delta, low valence) → B Minor (deep, searching)
        - Focused + positive (beta dominant, high valence) → F Major (confident, warm)
        - Focused + neutral (beta dominant) → D Minor (driven, determined)
        - Focused + dark (beta + low valence) → E-flat Major (tense, dramatic)
        - Peak/insight (gamma dominant) → current key stays (don't disrupt flow)
        """
        # Determine target key from mental state
        if gamma > 0.35:
            target = self._current_key  # Peak states don't change key
        elif valence > 0.6 and arousal < 0.5:
            target = 'C_MAJOR_A_MINOR'      # Calm + positive
        elif valence > 0.6 and arousal >= 0.5:
            target = 'D_MAJOR_B_MINOR'       # Energetic + positive
        elif valence < 0.4 and delta > 0.3:
            target = 'A_FLAT_MAJOR_F_MINOR'  # Deep + dark
        elif valence < 0.4 and theta > 0.3:
            target = 'E_FLAT_MAJOR_C_MINOR'  # Creative + dark
        elif valence < 0.4:
            target = 'B_FLAT_MAJOR_G_MINOR'  # Dark + tense
        elif theta > 0.3 and valence > 0.5:
            target = 'G_MAJOR_E_MINOR'       # Creative + positive
        elif beta > 0.3 and focus > 0.5:
            target = 'F_MAJOR_D_MINOR'       # Focused + driven
        else:
            target = 'C_MAJOR_A_MINOR'       # Default: neutral
        
        # Hysteresis: only change if target persists for N seconds
        now = time.time()
        if target != self._current_key:
            if target != self._key_candidate:
                self._key_candidate = target
                self._key_candidate_since = now
            elif now - self._key_candidate_since >= self._key_hysteresis_seconds:
                self._current_key = target
                self._key_candidate = None
        else:
            self._key_candidate = None
        
        return self._current_key
    
    def _resolve_genre(self, valence, arousal, theta=0.2, alpha=0.2, gamma=0.2, beta=0.2) -> str:
        """
        Map sustained mental state to genre. Oscillates between jazz and futuristic electronic.
        
        Theta (creativity/DMN) → futuristic electronic
        Alpha (flow/relaxed) → jazz
        Gamma (insight) → experimental electronic
        Beta (focus) → bebop jazz
        
        Valence × Arousal sets the energy within the genre.
        """
        # Theta/alpha ratio determines jazz vs electronic axis
        theta_alpha = theta / (alpha + 0.01)  # >1 = more creative/dreamy, <1 = more flow/relaxed
        gamma_beta = gamma / (beta + 0.01)   # >1 = more insight/sparkle, <1 = more focused/precise
        
        # Pick genre family: jazz vs electronic
        if theta_alpha > 1.5 or gamma > 0.3:
            # Futuristic electronic territory
            if arousal > 0.6:
                target = "futuristic electronic, driving synth arpeggios, energetic"
            elif valence < 0.4:
                target = "dark ambient electronic, deep drones, atmospheric"
            else:
                target = "chillwave, dreamy synth pads, electronic textures"
        elif gamma_beta > 1.5:
            # Experimental/IDM territory
            target = "experimental electronic, glitchy, intricate rhythms"
        elif theta_alpha < 0.6 and beta > 0.2:
            # Bebop/precise jazz territory
            target = "bebop jazz, fast tempo, virtuosic saxophone, instrumental"
        elif arousal > 0.6:
            # High energy jazz
            target = "upbeat jazz swing, energetic brass, walking bass"
        elif valence < 0.4:
            # Dark jazz
            target = "modal jazz, spacious, introspective piano, instrumental"
        elif valence > 0.6:
            # Warm jazz
            target = "smooth jazz, saxophone and piano, instrumental"
        else:
            # Default: cool jazz
            target = "cool jazz, laid back, mellow trumpet, instrumental"
        
        # Hysteresis: only change if target persists
        now = time.time()
        if target != self._current_genre:
            if target != self._genre_candidate:
                self._genre_candidate = target
                self._genre_candidate_since = now
            elif now - self._genre_candidate_since >= self._genre_hysteresis_seconds:
                self._current_genre = target
                self._genre_candidate = None
        else:
            self._genre_candidate = None
        
        return self._current_genre
    
    def generate_prompts(self, params: dict) -> list:
        prompts = []
        arousal = params.get('arousal_adjusted', params['arousal'])
        valence = params.get('valence_adjusted', params['valence'])
        relaxation = params.get('relaxation_adjusted', params.get('relaxation', 0.5))
        focus = params.get('focus_adjusted', params.get('focus', 0.5))
        
        delta = params.get('delta', 0.2)
        theta = params.get('theta', 0.2)
        alpha = params.get('alpha', 0.2)
        beta = params.get('beta', 0.2)
        gamma = params.get('gamma', 0.2)
        
        if arousal > 0.75:
            prompts.append(("driving drums, energetic pulse, tight groove", 1.0))
        elif arousal > 0.55:
            prompts.append(("steady rhythm, flowing movement, moderate energy", 1.0))
        elif arousal > 0.35:
            prompts.append(("gentle pulse, soft dynamics, relaxed tempo", 1.0))
        else:
            prompts.append(("slow sustained drones, minimal rhythm, spacious", 1.0))
            
        if valence > 0.65:
            prompts.append(("bright joyful and uplifting harmonies", valence))
        elif valence < 0.35:
            prompts.append(("contemplative mysterious and deeply melancholic mood", 0.7 - valence))
            
        if focus > 0.65:
            prompts.append(("tight precise and highly structured patterns", focus))
        elif relaxation > 0.65:
            prompts.append(("spacious drifting and ethereal ambient textures", relaxation))
            
        if theta > 0.35:
            prompts.append(("dreamy surreal otherworldly atmosphere", theta * 1.5))
        if gamma > 0.25:
            prompts.append(("brilliant sparkling details, shimmering highs", gamma * 2.0))
        
        dominant = max(['delta', 'theta', 'alpha', 'beta', 'gamma'], key=lambda b: params.get(b, 0))
        instrument_presets = {
            'delta': ["deep contrabass and warm cello", "sub-bass pulses and massive drones"],
            'theta': ["ethereal hang drum and wooden kalimba", "drifting celestial synth pads"],
            'alpha': ["intimate grand piano and warm classical guitar", "soft electric rhodes piano and nylon strings"],
            'beta': ["crisp marimba and bright vibraphone", "energetic synthesizer bass and electric guitar leads"],
            'gamma': ["glistening bells and chimes", "sparkling digital synths and harp glissandos"]
        }
        idx = 1 if arousal > 0.55 else 0
        instrument_prompt = instrument_presets.get(dominant, ["piano", "piano"])[idx]
        prompts.append((instrument_prompt, 0.8))
        
        if arousal > 0.5:
            if valence > 0.5:
                styles = ["organic jazz fusion", "energetic electro-pop", "uplifting house groove"]
            else:
                styles = ["dramatic cinematic orchestral", "dark minimal techno", "intense cyber-punk synth"]
        else:
            if valence > 0.5:
                styles = ["peaceful neo-classical piano", "warm lo-fi chillhop", "gentle folktronica"]
            else:
                styles = ["contemplative ambient drone", "deep melancholic shoegaze", "mysterious dark ambient"]
                
        if params.get('significant_change'):
            self.current_style_idx = (self.current_style_idx + 1) % len(styles)
        else:
            self.current_style_idx = self.current_style_idx % len(styles)
            
        selected_style = styles[self.current_style_idx]
        prompts.append((selected_style, 0.5))
        
        return prompts
    
    def reset_baseline(self):
        """Reset baseline para recalibrar"""
        self.baseline_metrics = None
        self.baseline_samples = []
        self.prev_metrics = None


class MusicGenerator:
    """Generador de música local con Magenta RealTime 2"""
    
    def __init__(self, model_size='mrt2_small'):
        self.is_generating = False
        self.audio_chunks = []
        self.current_style_embedding = None
        self.start_time = None
        self.buffer_ready = False
        self.buffer_progress = 0
        self.audio_callback = None  # Callback para enviar audio al navegador
        self._gen_temperature = 1.2
        self._gen_top_k = 40
        self._gen_cfg_musiccoca = 3.0
        self._gen_drums = None
        self._target_style_embedding = None
        self._style_blend = 0.4  # Faster style movement (was 0.15, too slow)
        self._vibe_history = deque(maxlen=10)
        self._tflite_lock = threading.Lock()  # Only for MusicCoCa TFLite interpreters
        self._gen_lock = threading.Lock()  # Separate lock for MLX generation (not needed but kept for safety)
        self._throttled = False  # Backend throttle flag set by frontend
        self._last_vibe_prompt = None  # Cache to skip re-embedding when prompt unchanged
        self._send_queue = None  # Will be set to a thread-safe queue for audio sends
        self.audio_bin_callback = None  # Binary audio send callback
        
        logger.info(f"🧠 Cargando modelo local Magenta RealTime 2 ({model_size})...")
        from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
        self.mrt = MagentaRT2SystemMlxfn(size=model_size)
        self.model_size = model_size
        logger.info(f"✅ Modelo {model_size} cargado.")
        self.state = None
        
    async def start(self):
        """Iniciar generación de música local"""
        if self.is_generating:
            return False
        
        self.is_generating = True
        self.start_time = time.time()
        self.audio_chunks = []
        self.buffer_ready = False
        self.buffer_progress = 0
        self.state = None  # Reset state para una nueva sesión
        self._silent_streak = 0  # Track consecutive silent chunks
        self._total_bytes = 0  # Running byte counter for buffer progress
        
        # Codificar estilo inicial por defecto
        self.current_style_embedding = self.mrt.embed_style("gentle flowing atmospheric music, ambient, calm")
        
        logger.info("🎵 Generación local de Magenta iniciada")
        return True
        
    async def stop(self):
        """Detener generación de música local"""
        if not self.is_generating:
            return None
            
        self.is_generating = False
        
        if self.audio_chunks:
            return self._save_audio()
        return None
        
    def _save_audio(self):
        """Guardar audio a archivo"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"eeg_music_{timestamp}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        audio_data = b''.join(self.audio_chunks)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE_AUDIO)
            wav_file.writeframes(audio_data)
        
        duration = len(audio_data) / (SAMPLE_RATE_AUDIO * 2 * CHANNELS)
        logger.info(f"💾 Audio guardado: {filepath} ({duration:.1f}s)")
        return filepath
        
    def _build_vibe_prompt(self, params: dict) -> str:
        """Build prompt from stable genre + continuous band modulation.
        Genre = stable identity (hysteresis). Bands = continuous variation within that genre."""
        
        genre = params.get('genre', 'lofi jazz piano, warm, gentle')
        
        # Continuous modifiers from bands — these add variety WITHIN the genre
        delta = params.get('delta', 0.2)
        theta = params.get('theta', 0.2)
        alpha = params.get('alpha', 0.2)
        beta = params.get('beta', 0.2)
        gamma = params.get('gamma', 0.2)
        
        # Pick the 1-2 most extreme band characteristics as modifiers
        modifiers = []
        
        # Delta: depth
        if delta > 0.25:
            modifiers.append("deep bass")
        
        # Theta: dreaminess
        if theta > 0.25:
            modifiers.append("dreamy")
        
        # Alpha: smoothness
        if alpha > 0.25:
            modifiers.append("smooth")
        
        # Beta: energy
        if beta > 0.25:
            modifiers.append("rhythmic")
        
        # Gamma: sparkle
        if gamma > 0.2:
            modifiers.append("bright")
        
        # Take max 2 modifiers to keep prompt concise
        modifier_str = ", ".join(modifiers[:2]) if modifiers else ""
        
        if modifier_str:
            return f"{genre}, {modifier_str}, instrumental"
        return f"{genre}, instrumental"
    
    async def update(self, params: dict, prompts: list):
        """Actualizar estilo de Magenta con mapping continuo y transiciones suaves"""
        if not self.is_generating:
            return
            
        try:
            self.current_params = params
            
            # Use the continuous parameters from map_to_lyria
            target_temp = params.get('temperature', 1.2)
            target_guidance = params.get('guidance', 3.0)
            arousal = params.get('arousal_adjusted', params.get('arousal', 0.5))
            
            # Smooth interpolation — respond to brain changes gradually
            blend = 0.3  # Slower than before for smoother transitions
            self._gen_temperature = self._gen_temperature * (1 - blend) + target_temp * blend
            self._gen_cfg_musiccoca = self._gen_cfg_musiccoca * (1 - blend) + target_guidance * blend
            
            # Top-k: derived from harmonic_complexity (more complexity = more variety)
            hc = params.get('harmonic_complexity', 0.5)
            target_top_k = int(np.clip(10 + hc * 40 + arousal * 10, 5, 50))
            self._gen_top_k = int(self._gen_top_k * (1 - blend) + target_top_k * blend)
            
            # Drums: always on
            self._gen_drums = [1]
            
            # --- Build vibe prompt and compute target style embedding ---
            vibe_prompt = self._build_vibe_prompt(params)
            self._vibe_history.append(vibe_prompt)
            
            # Skip re-embedding if prompt hasn't changed — saves CPU for generation
            if vibe_prompt == self._last_vibe_prompt:
                return
            self._last_vibe_prompt = vibe_prompt
            
            # Compute target embedding in a background thread to avoid blocking audio.
            # Guarded by a lock since the underlying TFLite interpreters are not
            # thread-safe and would race against generate()'s style tokenization.
            def _embed_style_locked(prompt):
                with self._tflite_lock:
                    return self.mrt.embed_style(prompt)
            
            loop = asyncio.get_event_loop()
            self._target_style_embedding = await loop.run_in_executor(
                None, _embed_style_locked, vibe_prompt
            )
            
            # Interpolate style embedding smoothly (blend toward target)
            if self.current_style_embedding is not None and self._target_style_embedding is not None:
                import mlx.core as mx
                current_np = np.array(self.current_style_embedding)
                target_np = np.array(self._target_style_embedding)
                blended = current_np * (1 - self._style_blend) + target_np * self._style_blend
                self.current_style_embedding = mx.array(blended.tolist())
            else:
                self.current_style_embedding = self._target_style_embedding
            
        except Exception as e:
            logger.error(f"Error actualizando estilo de Magenta: {e}")
            
    async def receive_audio(self):
        """Bucle continuo de generación de audio y envío al navegador"""
        import base64
        import asyncio
        chunk_count = 0
        
        frames_per_step = 25
        step_duration = frames_per_step * 0.04
        
        try:
            logger.info(f"🎧 Iniciando bucle local de generación ({frames_per_step} frames = {step_duration:.1f}s/audio)...")
            while self.is_generating:
                t_start = time.time()
                
                waveform, self.state = self.mrt.generate(
                    style=self.current_style_embedding,
                    drums=self._gen_drums,
                    frames=frames_per_step,
                    state=self.state,
                    temperature=self._gen_temperature,
                    top_k=self._gen_top_k,
                    cfg_musiccoca=self._gen_cfg_musiccoca,
                )
                
                generation_time = time.time() - t_start
                t_post = time.time()
                realtime_ratio = step_duration / generation_time if generation_time > 0 else 0
                
                # Detect silence — only reset state after consecutive silent chunks
                # to avoid micro-cuts from single quiet frames
                samples_float = waveform.samples
                rms = np.sqrt(np.mean(samples_float ** 2))
                if rms < 0.005:  # Very quiet (normalized -1 to 1 range)
                    self._silent_streak = getattr(self, '_silent_streak', 0) + 1
                    if self._silent_streak >= 3:
                        logger.warning(f"⚠️ {self._silent_streak} consecutive silent chunks (RMS={rms:.6f}), resetting state")
                        self.state = None
                        self._silent_streak = 0
                    # Still send the chunk to maintain stream continuity (it's quiet, not absent)
                else:
                    self._silent_streak = 0
                
                samples_int16 = (waveform.samples * 32767.0).astype(np.int16)
                chunk_data = samples_int16.tobytes()
                
                chunk_count += 1
                self.audio_chunks.append(chunk_data)
                # Limit memory: keep last 100 chunks for save-on-stop
                if len(self.audio_chunks) > 100:
                    self.audio_chunks = self.audio_chunks[-100:]
                
                if chunk_count % 50 == 0:
                    total_loop = time.time() - t_start
                    post_time = time.time() - t_post
                    logger.info(f"🎵 Chunk {chunk_count}: {generation_time:.2f}s gen / {step_duration:.1f}s audio (ratio {realtime_ratio:.2f}x) | post={post_time:.2f}s total={total_loop:.2f}s | temp={self._gen_temperature:.2f} top_k={self._gen_top_k} cfg={self._gen_cfg_musiccoca:.2f} drums={self._gen_drums}")
                    
                if self.audio_bin_callback:
                    # Send raw binary — no base64, no JSON, 33% less data
                    asyncio.ensure_future(self.audio_bin_callback(
                        chunk_count, chunk_data, SAMPLE_RATE_AUDIO, CHANNELS
                    ))
                elif self.audio_callback:
                    audio_b64 = base64.b64encode(chunk_data).decode('utf-8')
                    asyncio.ensure_future(self.audio_callback({
                        'type': 'audio',
                        'data': audio_b64,
                        'sample_rate': SAMPLE_RATE_AUDIO,
                        'channels': CHANNELS,
                        'chunk_id': chunk_count,
                    }))
                
                # Track buffer progress without O(n) sum every iteration
                self._total_bytes = getattr(self, '_total_bytes', 0) + len(chunk_data)
                target_bytes = 20 * SAMPLE_RATE_AUDIO * CHANNELS * 2
                self.buffer_progress = min(100, (self._total_bytes / target_bytes) * 100)
                if self.buffer_progress >= 100:
                    self.buffer_ready = True
                
                # If frontend says we have enough buffered, wait before generating more
                # But auto-reset after 10s to prevent permanent stall if frontend disconnects
                if self._throttled:
                    if not hasattr(self, '_throttle_start'):
                        self._throttle_start = time.time()
                    if time.time() - self._throttle_start > 10:
                        self._throttled = False
                        delattr(self, '_throttle_start')
                        logger.info("🔄 Throttle timeout — resuming generation")
                    else:
                        await asyncio.sleep(0.5)
                else:
                    if hasattr(self, '_throttle_start'):
                        delattr(self, '_throttle_start')
                    await asyncio.sleep(0)  # Minimal yield to let event loop process sends
                
            logger.info(f"Bucle local finalizado. Total de chunks generados: {chunk_count}")
        except Exception as e:
            logger.error(f"Error en bucle de generación de audio: {e}")
            import traceback
            traceback.print_exc()


class LyriaOnlineGenerator:
    """Generador de música online con Google Lyria 3 Clip API"""
    
    def __init__(self):
        self.is_generating = False
        self.audio_chunks = []
        self.client = None
        self.buffer_ready = False
        self.buffer_progress = 0
        self.current_params = None
        self.current_prompts = [("gentle flowing atmospheric music, ambient, calm", 1.0)]
        self.audio_callback = None
        
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        from google import genai
        self.client = genai.Client(api_key=api_key)
        logger.info("✅ Lyria 3 online client initialized")
    
    async def start(self):
        if self.is_generating:
            return False
        
        self.is_generating = True
        self.audio_chunks = []
        self.buffer_ready = False
        self.buffer_progress = 0
        
        logger.info("🎵 Lyria 3 online generation started")
        return True
    
    async def stop(self):
        if not self.is_generating:
            return None
        
        self.is_generating = False
        
        if self.audio_chunks:
            return self._save_audio()
        return None
    
    def _save_audio(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"eeg_music_{timestamp}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        audio_data = b''.join(self.audio_chunks)
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"💾 Audio guardado: {filepath} ({len(audio_data)} bytes)")
        return filepath
    
    async def update(self, params, prompts):
        if not self.is_generating:
            return
        self.current_params = params
        self.current_prompts = prompts
    
    def _build_prompt(self):
        parts = [(text, weight) for text, weight in self.current_prompts if weight > 0.3]
        if not parts:
            return "calm atmospheric instrumental music, no vocals"
        
        # Separar por categoría basándose en el contenido del prompt
        mood_parts = []
        instrument_parts = []
        style_parts = []
        atmosphere_parts = []
        
        for text, weight in parts:
            text_lower = text.lower()
            if any(w in text_lower for w in ['drum', 'pulse', 'groove', 'rhythm', 'tempo', 'drone', 'spacious', 'sustained']):
                mood_parts.append(text)
            elif any(w in text_lower for w in ['bright', 'joyful', 'melancholic', 'mysterious', 'uplifting', 'contemplative']):
                mood_parts.append(text)
            elif any(w in text_lower for w in ['tight', 'precise', 'structured', 'drifting', 'ethereal', 'ambient']):
                atmosphere_parts.append(text)
            elif any(w in text_lower for w in ['dreamy', 'surreal', 'otherworldly', 'sparkling', 'shimmering', 'brilliant', 'glistening']):
                atmosphere_parts.append(text)
            elif any(w in text_lower for w in ['piano', 'guitar', 'synth', 'bass', 'cello', 'drum', 'marimba', 'vibraphone', 'bell', 'chime', 'harp', 'kalimba', 'rhodes', 'contrabass']):
                instrument_parts.append(text)
            else:
                style_parts.append(text)
        
        # Construir prompt estructurado tipo [0:00 - 0:30] para Lyria 3
        prompt_sections = []
        prompt_sections.append("instrumental only, no vocals")
        
        if atmosphere_parts:
            prompt_sections.append(", ".join(atmosphere_parts))
        if mood_parts:
            prompt_sections.append(", ".join(mood_parts))
        if instrument_parts:
            prompt_sections.append(", ".join(instrument_parts))
        if style_parts:
            prompt_sections.append(", ".join(style_parts))
        
        return ", ".join(prompt_sections)
    
    async def receive_audio(self):
        import base64
        clip_count = 0
        
        try:
            logger.info("🎧 Iniciando generación de clips Lyria 3...")
            
            while self.is_generating:
                prompt = self._build_prompt()
                logger.info(f"🎵 Generating clip {clip_count + 1} with Lyria 3 Clip...")
                logger.info(f"   Prompt: {prompt[:80]}...")
                
                t_start = time.time()
                
                try:
                    loop = asyncio.get_event_loop()
                    interaction = await loop.run_in_executor(
                        None,
                        lambda: self.client.interactions.create(
                            model="lyria-3-clip-preview",
                            input=prompt,
                        )
                    )
                    
                    gen_time = time.time() - t_start
                    logger.info(f"   Generated in {gen_time:.1f}s")
                    
                    if interaction.output_audio and interaction.output_audio.data:
                        audio_bytes = base64.b64decode(interaction.output_audio.data)
                        self.audio_chunks.append(audio_bytes)
                        clip_count += 1
                        
                        if self.audio_callback:
                            audio_b64 = interaction.output_audio.data
                            await self.audio_callback({
                                'type': 'audio_mp3',
                                'data': audio_b64,
                                'clip': clip_count,
                            })
                        
                        total_mb = sum(len(c) for c in self.audio_chunks) / (1024 * 1024)
                        self.buffer_progress = min(100, clip_count * 25)
                        if self.buffer_progress >= 100:
                            self.buffer_ready = True
                        
                        logger.info(f"   Clip {clip_count}: {len(audio_bytes)} bytes ({total_mb:.1f}MB total)")
                    
                except Exception as e:
                    logger.error(f"Error generating clip: {e}")
                    await asyncio.sleep(2)
            
            logger.info(f"Lyria 3 generation stopped. {clip_count} clips generated.")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.is_generating:
                logger.error(f"Error en Lyria 3: {e}")
                import traceback
                traceback.print_exc()


# HTML del Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠🎵 EEG Music Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 20px; font-size: 2em; }
        h1 span { font-size: 0.5em; opacity: 0.7; display: block; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .card h2 {
            font-size: 1.1em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
        }
        .status.connected { background: #44ff44; }
        .status.generating { background: #ffaa00; animation: pulse 1s infinite; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.2); }
        }
        
        .band-bar {
            display: flex;
            align-items: center;
            margin: 8px 0;
            gap: 10px;
        }
        .band-label { width: 70px; font-size: 0.85em; }
        .band-track {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        .band-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .band-value { width: 45px; text-align: right; font-family: monospace; font-size: 0.85em; }
        
        .delta .band-fill { background: linear-gradient(90deg, #9b59b6, #8e44ad); }
        .theta .band-fill { background: linear-gradient(90deg, #3498db, #2980b9); }
        .alpha .band-fill { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .beta .band-fill { background: linear-gradient(90deg, #f1c40f, #f39c12); }
        .gamma .band-fill { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .metric {
            text-align: center;
            padding: 12px 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .metric-value { font-size: 1.5em; font-weight: bold; margin: 3px 0; }
        .metric-label { opacity: 0.7; font-size: 0.8em; }
        
        .music-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.2s;
            flex: 1;
        }
        button:hover { transform: scale(1.02); }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .btn-start { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }
        .btn-stop { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
        .btn-reset { background: rgba(255,255,255,0.2); color: white; }
        
        .music-status {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .music-param {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .music-param:last-child { border-bottom: none; }
        .param-label { opacity: 0.7; }
        .param-value { font-weight: bold; font-family: monospace; }
        
        .prompt-display {
            background: rgba(52, 152, 219, 0.2);
            border: 1px solid rgba(52, 152, 219, 0.4);
            border-radius: 10px;
            padding: 12px;
            margin-top: 10px;
            font-style: italic;
            font-size: 0.9em;
        }
        
        .buffer-progress {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        .buffer-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s;
        }
        
        .state-indicator {
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        .state-emoji { font-size: 2.5em; display: block; margin-bottom: 8px; }
        
        .chart-container { height: 150px; margin-top: 10px; }
        
        .log-container {
            max-height: 150px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 10px;
            font-family: monospace;
            font-size: 0.8em;
        }
        .log-entry { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-time { color: #888; }
        .log-change { color: #f1c40f; }
        
        .slider-group { display: flex; flex-direction: column; gap: 8px; }
        .slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-label { width: 80px; font-size: 0.9em; }
        .slider-value { width: 45px; text-align: right; font-family: monospace; font-size: 0.85em; }
        input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #3498db;
            border-radius: 50%;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            background: #2980b9;
            transform: scale(1.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠🎵 EEG Music Dashboard <span>Brain-controlled music generation</span></h1>
        
        <div class="grid">
            <!-- EEG Bands -->
            <div class="card">
                <h2><span class="status" id="eeg-status"></span> EEG Bands</h2>
                
                <div class="band-bar delta">
                    <span class="band-label">δ Delta</span>
                    <div class="band-track"><div class="band-fill" id="delta-bar"></div></div>
                    <span class="band-value" id="delta-val">0.00</span>
                </div>
                <div class="band-bar theta">
                    <span class="band-label">θ Theta</span>
                    <div class="band-track"><div class="band-fill" id="theta-bar"></div></div>
                    <span class="band-value" id="theta-val">0.00</span>
                </div>
                <div class="band-bar alpha">
                    <span class="band-label">α Alpha</span>
                    <div class="band-track"><div class="band-fill" id="alpha-bar"></div></div>
                    <span class="band-value" id="alpha-val">0.00</span>
                </div>
                <div class="band-bar beta">
                    <span class="band-label">β Beta</span>
                    <div class="band-track"><div class="band-fill" id="beta-bar"></div></div>
                    <span class="band-value" id="beta-val">0.00</span>
                </div>
                <div class="band-bar gamma">
                    <span class="band-label">γ Gamma</span>
                    <div class="band-track"><div class="band-fill" id="gamma-bar"></div></div>
                    <span class="band-value" id="gamma-val">0.00</span>
                </div>
                
                <div class="chart-container">
                    <canvas id="bandChart"></canvas>
                </div>
            </div>
            
            <!-- Metrics -->
            <div class="card">
                <h2>🎯 Brain State</h2>
                
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Arousal</div>
                        <div class="metric-value" id="arousal-val" style="color:#e74c3c">0.00</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Valence</div>
                        <div class="metric-value" id="valence-val" style="color:#2ecc71">0.00</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Focus</div>
                        <div class="metric-value" id="focus-val" style="color:#f1c40f">0.00</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Relaxation</div>
                        <div class="metric-value" id="relax-val" style="color:#3498db">0.00</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Dominant</div>
                        <div class="metric-value" id="dominant-val" style="color:#9b59b6">α</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Change</div>
                        <div class="metric-value" id="change-val">-</div>
                    </div>
                </div>
                
                <div class="state-indicator">
                    <span class="state-emoji" id="state-emoji">🧘</span>
                    <span id="state-text">Connecting...</span>
                </div>
            </div>
            
            <!-- Music Controls -->
            <div class="card">
                <h2><span class="status" id="music-status"></span> Music Control</h2>
                
                <div style="margin-bottom:15px;">
                    <label style="font-size:0.85em; opacity:0.7; display:block; margin-bottom:6px;">Music Engine</label>
                    <div style="display:flex; gap:8px;">
                        <button id="engine-local" onclick="selectEngine('local')" style="flex:1; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15); background:rgba(78,205,196,0.2); color:#e6edf3; cursor:pointer; font-size:0.85em;">
                            🧠 Local (Magenta)
                        </button>
                        <button id="engine-online" onclick="selectEngine('online')" style="flex:1; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.05); color:#e6edf3; cursor:pointer; font-size:0.85em;">
                            ☁️ Online (Lyria)
                        </button>
                    </div>
                </div>
                
                <div id="model-selector-div" style="margin-bottom:15px;">
                    <label style="font-size:0.85em; opacity:0.7; display:block; margin-bottom:6px;">Model Size (Local)</label>
                    <select id="model-select" style="width:100%; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.05); color:#e6edf3; font-size:0.85em;">
                        <option value="mrt2_small">⚡ Fast — mrt2_small (230M, ~2.3x realtime)</option>
                        <option value="mrt2_base_fast">🚀 Base Fast — 4-bit + 1 CFG (~2x realtime, best quality/speed)</option>
                        <option value="mrt2_base">🎨 Quality — mrt2_base (2.4B, ~0.7x realtime, cuts)</option>
                    </select>
                </div>
                
                <div class="music-controls">
                    <button class="btn-start" id="btn-start" onclick="startMusic()" disabled>▶️ Start Music</button>
                    <button class="btn-stop" id="btn-stop" onclick="stopMusic()" disabled>⏹️ Stop</button>
                </div>
                
                <button class="btn-reset" id="btn-calibrate" onclick="startCalibration()" style="width:100%; margin-bottom:10px;">📊 Calibrate Baseline (10s)</button>
                
                <div style="margin-bottom:10px;">
                    <label style="font-size:0.8em; opacity:0.7; display:block; margin-bottom:4px;">Genre Override</label>
                    <select id="genre-select" onchange="setGenre(this.value)" style="width:100%; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.05); color:#e6edf3; font-size:0.85em;">
                        <option value="">Auto (EEG-driven)</option>
                        <option value="cool jazz, laid back, mellow trumpet, instrumental">Cool Jazz</option>
                        <option value="experimental piano, prepared piano, extended techniques, classical">Experimental Piano</option>
                        <option value="contemporary classical, strings and piano, orchestral">Classical</option>
                        <option value="futuristic electronic, driving synth arpeggios, energetic">Futuristic Electronic</option>
                        <option value="ambient, atmospheric pads, evolving textures">Ambient</option>
                        <option value="bebop jazz, fast tempo, virtuosic saxophone, instrumental">Bebop</option>
                        <option value="modal jazz, spacious, introspective piano, instrumental">Modal Jazz</option>
                        <option value="lofi hip hop, dusty piano, vinyl crackle, chill">Lo-fi Hip Hop</option>
                    </select>
                </div>
                
                <button class="btn-reset" onclick="resetBaseline()" style="width:100%; margin-bottom:10px;">🔄 Reset Baseline</button>
                <button class="btn-reset" id="btn-muse" onclick="startMuseStream()" style="width:100%; margin-bottom:15px;">📡 Start Muse Stream</button>
                <div id="muse-status" style="text-align:center; font-size:0.8em; margin-bottom:10px; opacity:0.7;">Muse: Not connected</div>
                
                <div class="music-status">
                    <div class="music-param">
                        <span class="param-label">Status</span>
                        <span class="param-value" id="gen-status">Stopped</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Time</span>
                        <span class="param-value" id="gen-time">0:00</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Buffer</span>
                        <span class="param-value" id="buffer-status">0%</span>
                    </div>
                    <div class="buffer-progress">
                        <div class="buffer-fill" id="buffer-bar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            
            <!-- Music Parameters -->
            <div class="card">
                <h2>🎛️ Music Parameters</h2>
                
                <div class="music-status">
                    <div class="music-param">
                        <span class="param-label">BPM</span>
                        <span class="param-value" id="param-bpm">80</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Scale</span>
                        <span class="param-value" id="param-scale">C Major</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Density</span>
                        <span class="param-value" id="param-density">0.50</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Brightness</span>
                        <span class="param-value" id="param-brightness">0.50</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Drums</span>
                        <span class="param-value" id="param-drums">ON</span>
                    </div>
                    <div class="music-param">
                        <span class="param-label">Bass</span>
                        <span class="param-value" id="param-bass">ON</span>
                    </div>
                </div>
                
                <div class="prompt-display" id="current-prompt">
                    Waiting to start...
                </div>
            </div>
            
            <!-- Sensitivity Controls -->
            <div class="card">
                <h2>🎚️ Band Sensitivity</h2>
                <p style="opacity:0.7; font-size:0.85em; margin-bottom:15px;">Adjust how much each band affects the music</p>
                
                <div class="slider-group">
                    <div class="slider-row">
                        <span class="slider-label" style="color:#9b59b6">δ Delta</span>
                        <input type="range" min="0" max="300" value="100" id="sens-delta" oninput="updateSensitivity()">
                        <span class="slider-value" id="sens-delta-val">1.0x</span>
                    </div>
                    <div class="slider-row">
                        <span class="slider-label" style="color:#3498db">θ Theta</span>
                        <input type="range" min="0" max="300" value="100" id="sens-theta" oninput="updateSensitivity()">
                        <span class="slider-value" id="sens-theta-val">1.0x</span>
                    </div>
                    <div class="slider-row">
                        <span class="slider-label" style="color:#2ecc71">α Alpha</span>
                        <input type="range" min="0" max="300" value="100" id="sens-alpha" oninput="updateSensitivity()">
                        <span class="slider-value" id="sens-alpha-val">1.0x</span>
                    </div>
                    <div class="slider-row">
                        <span class="slider-label" style="color:#f1c40f">β Beta</span>
                        <input type="range" min="0" max="300" value="100" id="sens-beta" oninput="updateSensitivity()">
                        <span class="slider-value" id="sens-beta-val">1.0x</span>
                    </div>
                    <div class="slider-row">
                        <span class="slider-label" style="color:#e74c3c">γ Gamma</span>
                        <input type="range" min="0" max="300" value="100" id="sens-gamma" oninput="updateSensitivity()">
                        <span class="slider-value" id="sens-gamma-val">1.0x</span>
                    </div>
                </div>
                
                <div style="margin-top:20px; padding-top:15px; border-top:1px solid rgba(255,255,255,0.1);">
                    <h3 style="font-size:1em; margin-bottom:10px;">⚖️ Stability</h3>
                    <div class="slider-row">
                        <span class="slider-label">Smoothing</span>
                        <input type="range" min="1" max="20" value="5" id="smoothing" oninput="updateSensitivity()">
                        <span class="slider-value" id="smoothing-val">5</span>
                    </div>
                    <div class="slider-row">
                        <span class="slider-label">Global Sens.</span>
                        <input type="range" min="50" max="500" value="250" id="global-sens" oninput="updateSensitivity()">
                        <span class="slider-value" id="global-sens-val">2.5x</span>
                    </div>
                </div>
                
                <button class="btn-secondary" onclick="resetSliders()" style="width:100%; margin-top:15px;">↺ Reset to Default</button>
            </div>
            
            <!-- Log -->
            <div class="card">
                <h2>📋 Activity Log</h2>
                <div class="log-container" id="log-container"></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let isGenerating = false;
        let startTime = null;
        let bandHistory = { delta: [], theta: [], alpha: [], beta: [], gamma: [] };
        const maxHistory = 60;
        
        // Chart
        const ctx = document.getElementById('bandChart').getContext('2d');
        const bandChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(maxHistory).fill(''),
                datasets: [
                    { label: 'Delta', data: [], borderColor: '#9b59b6', tension: 0.4, pointRadius: 0, borderWidth: 2 },
                    { label: 'Theta', data: [], borderColor: '#3498db', tension: 0.4, pointRadius: 0, borderWidth: 2 },
                    { label: 'Alpha', data: [], borderColor: '#2ecc71', tension: 0.4, pointRadius: 0, borderWidth: 2 },
                    { label: 'Beta', data: [], borderColor: '#f1c40f', tension: 0.4, pointRadius: 0, borderWidth: 2 },
                    { label: 'Gamma', data: [], borderColor: '#e74c3c', tension: 0.4, pointRadius: 0, borderWidth: 2 },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 0.6, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#888' } },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        function connect() {
            ws = new WebSocket('ws://localhost:8767');
            
            ws.onopen = () => {
                document.getElementById('eeg-status').classList.add('connected');
                addLog('Conectado al servidor');
            };
            
            ws.onclose = () => {
                document.getElementById('eeg-status').classList.remove('connected');
                addLog('Desconectado - reconectando...');
                setTimeout(connect, 2000);
            };
            
            ws.binaryType = 'arraybuffer';
            ws.onmessage = (event) => {
                // Binary frames = raw audio chunks (4-byte chunk_id + int16 samples)
                if (event.data instanceof ArrayBuffer) {
                    handleBinaryAudio(event.data);
                    return;
                }
                const data = JSON.parse(event.data);
                
                if (data.type === 'eeg') {
                    updateEEG(data);
                } else if (data.type === 'audio') {
                    handleAudioChunk(data);
                } else if (data.type === 'audio_mp3') {
                    handleMP3Clip(data);
                } else if (data.type === 'music_status') {
                    updateMusicStatus(data);
                } else if (data.type === 'music_started') {
                    isGenerating = true;
                    startTime = Date.now();
                    initAudio();
                    document.getElementById('btn-start').disabled = true;
                    document.getElementById('btn-stop').disabled = false;
                    document.getElementById('music-status').classList.add('generating');
                    document.getElementById('gen-status').textContent = 'Buffering...';
                    addLog('🎵 Music started - buffering audio...', 'change');
                } else if (data.type === 'muse_status') {
                    updateMuseStatus(data.connected);
                } else if (data.type === 'calibration_done') {
                    const btn = document.getElementById('btn-calibrate');
                    btn.disabled = false;
                    btn.textContent = data.success ? '📊 Recalibrate Baseline' : '📊 Calibrate Baseline (10s)';
                    if (data.success) addLog('✅ Baseline captured!', 'change');
                } else if (data.type === 'music_stopped') {
                    isGenerating = false;
                    stopAudio();
                    document.getElementById('btn-start').disabled = false;
                    document.getElementById('btn-stop').disabled = true;
                    document.getElementById('music-status').classList.remove('generating');
                    document.getElementById('gen-status').textContent = 'Stopped';
                    if (data.file) {
                        addLog('💾 Audio saved: ' + data.file, 'change');
                    }
                } else if (data.type === 'log') {
                    addLog(data.message, data.level);
                }
            };
        }
        
        let lastChartUpdate = 0;
        function updateEEG(data) {
            const bands = data.bands;
            const metrics = data.metrics;
            const music = data.music_params;
            
            // Always accumulate band history (cheap)
            for (const band of ['delta', 'theta', 'alpha', 'beta', 'gamma']) {
                const val = bands[band] || 0;
                bandHistory[band].push(val);
                if (bandHistory[band].length > maxHistory) bandHistory[band].shift();
            }
            
            // Update bars + text on every message (~1/s, very cheap)
            for (const band of ['delta', 'theta', 'alpha', 'beta', 'gamma']) {
                const val = bands[band] || 0;
                document.getElementById(`${band}-bar`).style.width = `${val * 100}%`;
                document.getElementById(`${band}-val`).textContent = val.toFixed(2);
            }
            
            const adj_a = metrics.arousal_adjusted || metrics.arousal;
            const adj_v = metrics.valence_adjusted || metrics.valence;
            document.getElementById('arousal-val').textContent = adj_a.toFixed(2);
            document.getElementById('valence-val').textContent = adj_v.toFixed(2);
            document.getElementById('focus-val').textContent = (metrics.focus_adjusted || metrics.focus).toFixed(2);
            document.getElementById('relax-val').textContent = (metrics.relaxation_adjusted || metrics.relaxation).toFixed(2);
            
            const bandMap = {delta: 'δ', theta: 'θ', alpha: 'α', beta: 'β', gamma: 'γ'};
            const dominant = Object.keys(bands).reduce((a, b) => bands[a] > bands[b] ? a : b);
            document.getElementById('dominant-val').textContent = bandMap[dominant];
            document.getElementById('change-val').textContent = metrics.significant_change ? '🔄' : '-';
            updateState(metrics);
            
            if (music) {
                document.getElementById('param-bpm').textContent = music.bpm;
                document.getElementById('param-scale').textContent = formatScale(music.scale);
                document.getElementById('param-density').textContent = music.density.toFixed(2);
                document.getElementById('param-brightness').textContent = music.brightness.toFixed(2);
                document.getElementById('param-drums').textContent = music.mute_drums ? 'OFF' : 'ON';
                document.getElementById('param-bass').textContent = music.mute_bass ? 'OFF' : 'ON';
            }
            
            if (data.prompt) {
                document.getElementById('current-prompt').textContent = data.prompt;
            }
            
            // Update chart every 2s (Chart.js re-render is expensive)
            const now = Date.now();
            if (now - lastChartUpdate >= 2000) {
                lastChartUpdate = now;
                bandChart.data.datasets[0].data = bandHistory.delta;
                bandChart.data.datasets[1].data = bandHistory.theta;
                bandChart.data.datasets[2].data = bandHistory.alpha;
                bandChart.data.datasets[3].data = bandHistory.beta;
                bandChart.data.datasets[4].data = bandHistory.gamma;
                bandChart.update('none');
            }
        }
        
        function updateMusicStatus(data) {
            document.getElementById('buffer-status').textContent = data.buffer_progress.toFixed(0) + '%';
            document.getElementById('buffer-bar').style.width = data.buffer_progress + '%';
            
            if (data.buffer_ready) {
                document.getElementById('gen-status').textContent = '🔊 Playing';
            }
        }
        
        function updateState(metrics) {
            const relax = metrics.relaxation_adjusted || metrics.relaxation;
            const focus = metrics.focus_adjusted || metrics.focus;
            const arousal = metrics.arousal_adjusted || metrics.arousal;
            
            let emoji = '🧘', text = 'Neutral';
            
            if (relax > 0.6) { emoji = '😌'; text = 'Relaxed'; }
            else if (focus > 0.6) { emoji = '🎯'; text = 'Focused'; }
            else if (arousal > 0.7) { emoji = '⚡'; text = 'High Energy'; }
            else if (arousal < 0.2) { emoji = '😴'; text = 'Low Energy'; }
            
            document.getElementById('state-emoji').textContent = emoji;
            document.getElementById('state-text').textContent = text;
        }
        
        function formatScale(scale) {
            const map = {
                'C_MAJOR_A_MINOR': 'C Major',
                'D_MAJOR_B_MINOR': 'D Major',
                'G_MAJOR_E_MINOR': 'G Major',
                'F_MAJOR_D_MINOR': 'F Major',
                'A_MAJOR_G_FLAT_MINOR': 'A Major',
                'E_FLAT_MAJOR_C_MINOR': 'E♭ Major',
                'B_FLAT_MAJOR_G_MINOR': 'B♭ Major',
                'A_FLAT_MAJOR_F_MINOR': 'A♭ Major'
            };
            return map[scale] || scale;
        }
        
        let selectedEngine = 'local';
        
        function selectEngine(engine) {
            selectedEngine = engine;
            document.getElementById('engine-local').style.background = engine === 'local' ? 'rgba(78,205,196,0.2)' : 'rgba(255,255,255,0.05)';
            document.getElementById('engine-online').style.background = engine === 'online' ? 'rgba(78,205,196,0.2)' : 'rgba(255,255,255,0.05)';
            document.getElementById('model-selector-div').style.display = engine === 'local' ? 'block' : 'none';
            addLog('Engine: ' + (engine === 'local' ? 'Local (Magenta)' : 'Online (Lyria)'));
        }
        
        function startMusic() {
            const modelSize = document.getElementById('model-select') ? document.getElementById('model-select').value : 'mrt2_small';
            ws.send(JSON.stringify({ action: 'start_music', engine: selectedEngine, model_size: modelSize }));
            addLog(`Starting music generation... (${selectedEngine}${selectedEngine === 'local' ? ', ' + modelSize : ''})`);
        }
        
        function stopMusic() {
            ws.send(JSON.stringify({ action: 'stop_music' }));
            addLog('Stopping music...');
        }
        
        function resetBaseline() {
            ws.send(JSON.stringify({ action: 'reset_baseline' }));
            addLog('🔄 Resetting baseline...', 'change');
        }
        
        function startCalibration() {
            const btn = document.getElementById('btn-calibrate');
            btn.disabled = true;
            btn.textContent = '📊 Calibrating...';
            ws.send(JSON.stringify({ action: 'calibrate' }));
            addLog('📊 Calibrating baseline — stay still and relaxed for 10s...', 'change');
        }
        
        function setGenre(value) {
            ws.send(JSON.stringify({ action: 'set_genre', genre: value }));
            if (value) {
                addLog(`🎵 Genre override: ${value}`, 'change');
            } else {
                addLog('🎵 Genre: Auto (EEG-driven)', 'change');
            }
        }
        
        let museStreamRunning = false;
        let museConnected = false;
        function startMuseStream() {
            if (museStreamRunning) {
                addLog('Muse stream already running');
                return;
            }
            museStreamRunning = true;
            const btn = document.getElementById('btn-muse');
            btn.disabled = true;
            btn.textContent = '📡 Starting Muse...';
            ws.send(JSON.stringify({ action: 'start_muse' }));
            addLog('📡 Starting muselsl stream...', 'change');
        }
        
        function updateMuseStatus(connected) {
            museConnected = connected;
            const el = document.getElementById('muse-status');
            const btn = document.getElementById('btn-muse');
            if (connected) {
                el.textContent = 'Muse: ✅ Connected';
                el.style.color = '#4ecdc4';
                el.style.opacity = '1';
                document.getElementById('btn-start').disabled = false;
                btn.disabled = true;
                btn.textContent = '📡 Muse Active';
            } else {
                el.textContent = 'Muse: ❌ Not connected';
                el.style.color = '#e74c3c';
                el.style.opacity = '1';
                document.getElementById('btn-start').disabled = true;
                btn.disabled = false;
                btn.textContent = '📡 Start Muse Stream';
                museStreamRunning = false;
            }
        }
        
        // ============ SENSITIVITY CONTROLS ============
        let settingsDebounce = null;
        function updateSensitivity() {
            const settings = {
                band_sensitivity: {
                    delta: document.getElementById('sens-delta').value / 100,
                    theta: document.getElementById('sens-theta').value / 100,
                    alpha: document.getElementById('sens-alpha').value / 100,
                    beta: document.getElementById('sens-beta').value / 100,
                    gamma: document.getElementById('sens-gamma').value / 100
                },
                smoothing: parseInt(document.getElementById('smoothing').value),
                global_sensitivity: document.getElementById('global-sens').value / 100
            };
            
            // Update display values immediately
            document.getElementById('sens-delta-val').textContent = settings.band_sensitivity.delta.toFixed(1) + 'x';
            document.getElementById('sens-theta-val').textContent = settings.band_sensitivity.theta.toFixed(1) + 'x';
            document.getElementById('sens-alpha-val').textContent = settings.band_sensitivity.alpha.toFixed(1) + 'x';
            document.getElementById('sens-beta-val').textContent = settings.band_sensitivity.beta.toFixed(1) + 'x';
            document.getElementById('sens-gamma-val').textContent = settings.band_sensitivity.gamma.toFixed(1) + 'x';
            document.getElementById('smoothing-val').textContent = settings.smoothing;
            document.getElementById('global-sens-val').textContent = settings.global_sensitivity.toFixed(1) + 'x';
            
            // Debounce sending to server (300ms after last slider movement)
            if (settingsDebounce) clearTimeout(settingsDebounce);
            settingsDebounce = setTimeout(() => {
                ws.send(JSON.stringify({ action: 'update_settings', settings: settings }));
                settingsDebounce = null;
            }, 300);
        }
        
        function resetSliders() {
            document.getElementById('sens-delta').value = 100;
            document.getElementById('sens-theta').value = 100;
            document.getElementById('sens-alpha').value = 100;
            document.getElementById('sens-beta').value = 100;
            document.getElementById('sens-gamma').value = 100;
            document.getElementById('smoothing').value = 5;
            document.getElementById('global-sens').value = 250;
            updateSensitivity();
            addLog('↺ Sliders reset to default');
        }
        
        function addLog(message, level = '') {
            const container = document.getElementById('log-container');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry' + (level === 'change' ? ' log-change' : '');
            entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }
        
        // Update time
        setInterval(() => {
            if (isGenerating && startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                document.getElementById('gen-time').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }
        }, 1000);
        
        // ============ WEB AUDIO API FOR PLAYBACK ============
        let audioContext = null;
        let audioQueue = [];
        let isPlaying = false;
        let nextPlayTime = 0;
        const BUFFER_SECONDS = 30; // Pre-buffer before playback (30s = enough headroom for jitter)
        let totalBufferedSeconds = 0;
        let playbackStarted = false;
        let chunkCount = 0;
        let gapCount = 0;
        let lastChunkTime = 0;
        let backendThrottled = false;
        const AUDIO_DEBUG = true; // Set to true for audio pipeline debugging logs
        
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 48000
                });
                addLog('🔊 Audio initialized');
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        
        function handleBinaryAudio(arrayBuffer) {
            if (!audioContext) initAudio();
            
            const now = performance.now();
            chunkCount++;
            const interChunkMs = lastChunkTime > 0 ? (now - lastChunkTime).toFixed(0) : '-';
            lastChunkTime = now;
            
            // Skip 4-byte header (chunk_id), rest is Int16 stereo PCM
            const int16View = new Int16Array(arrayBuffer, 4);
            const numSamples = int16View.length / 2;
            const audioBuffer = audioContext.createBuffer(2, numSamples, 48000);
            const leftChannel = audioBuffer.getChannelData(0);
            const rightChannel = audioBuffer.getChannelData(1);
            
            for (let i = 0; i < numSamples; i++) {
                leftChannel[i] = int16View[i * 2] / 32768.0;
                rightChannel[i] = int16View[i * 2 + 1] / 32768.0;
            }
            
            // Add to queue
            audioQueue.push(audioBuffer);
            totalBufferedSeconds += audioBuffer.duration;
            
            // Logging: chunk arrival stats (only in debug mode)
            if (AUDIO_DEBUG && chunkCount % 10 === 0) {
                const scheduledAhead = nextPlayTime > 0 ? (nextPlayTime - audioContext.currentTime).toFixed(2) : '0';
                console.log(`[AUDIO] chunk#${chunkCount} | inter-chunk: ${interChunkMs}ms | queue: ${audioQueue.length} | ahead: ${scheduledAhead}s`);
            }
            
            // Start playback after buffering enough
            if (!playbackStarted && totalBufferedSeconds >= BUFFER_SECONDS) {
                playbackStarted = true;
                nextPlayTime = audioContext.currentTime + 0.1;
                addLog(`🔊 Playback started! (buffer: ${totalBufferedSeconds.toFixed(1)}s)`, 'change');
                if (AUDIO_DEBUG) console.log(`[AUDIO] === PLAYBACK STARTED === queue=${audioQueue.length} buffered=${totalBufferedSeconds.toFixed(1)}s`);
            } else if (!playbackStarted) {
                // Show buffering progress
                const pct = Math.min(100, (totalBufferedSeconds / BUFFER_SECONDS * 100).toFixed(0));
                const bufStatus = document.getElementById('buffer-status');
                if (bufStatus) bufStatus.textContent = `${pct}% (buffering ${totalBufferedSeconds.toFixed(1)}s/${BUFFER_SECONDS}s)`;
                const bufBar = document.getElementById('buffer-bar');
                if (bufBar) bufBar.style.width = pct + '%';
            }
            
            // Always ensure scheduler is running when we have audio
            if (playbackStarted) {
                startScheduler();
            }
        }
        
        let schedulerInterval = null;
        
        function scheduleBuffers() {
            if (!audioContext || !playbackStarted) return;
            
            // Auto-resume if browser suspended the context
            if (audioContext.state === 'suspended') {
            if (AUDIO_DEBUG) console.warn('[AUDIO] ⚠️ AudioContext SUSPENDED — attempting resume');
                audioContext.resume();
                return;
            }
            
            // If we fell behind realtime, catch up to avoid gaps
            if (nextPlayTime < audioContext.currentTime) {
                const gapSec = (audioContext.currentTime - nextPlayTime).toFixed(3);
                gapCount++;
            if (AUDIO_DEBUG) console.warn(`[AUDIO] ⚠️ GAP #${gapCount} detected! nextPlayTime was ${gapSec}s behind currentTime. Resetting. queue=${audioQueue.length}`);
                nextPlayTime = audioContext.currentTime + 0.02;
            }
            // Schedule only 3s ahead — keep rest in queue as jitter buffer
            const LOOKAHEAD = 3.0;
            let scheduledThisRun = 0;
            while (audioQueue.length > 0 && nextPlayTime < audioContext.currentTime + LOOKAHEAD) {
                const buffer = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                
                // No crossfade — model streaming state ensures seamless continuity
                source.connect(audioContext.destination);
                source.start(nextPlayTime);
                source.onended = function() {
                    source.disconnect();
                };
                nextPlayTime = nextPlayTime + buffer.duration;
                scheduledThisRun++;
            }
            if (AUDIO_DEBUG && scheduledThisRun > 0 && chunkCount % 10 === 0) {
                const ahead = (nextPlayTime - audioContext.currentTime).toFixed(2);
                console.log(`[SCHED] scheduled ${scheduledThisRun} | queue: ${audioQueue.length} | ahead: ${ahead}s | gaps: ${gapCount}`);
            }
            // Throttle backend: tell it to pause if queue is large, resume if small
            if (ws && ws.readyState === 1) {
                if (audioQueue.length > 40 && !backendThrottled) {
                    backendThrottled = true;
                    ws.send(JSON.stringify({ action: 'throttle', throttled: true }));
                if (AUDIO_DEBUG) console.log('[THROTTLE] telling backend to PAUSE generation');
                } else if (audioQueue.length < 15 && backendThrottled) {
                    backendThrottled = false;
                    ws.send(JSON.stringify({ action: 'throttle', throttled: false }));
                if (AUDIO_DEBUG) console.log('[THROTTLE] telling backend to RESUME generation');
                }
            }
            // Warn if queue is empty and we're still generating (only in debug mode)
            if (AUDIO_DEBUG && audioQueue.length === 0 && isGenerating && chunkCount % 10 === 0) {
                const ahead = (nextPlayTime - audioContext.currentTime).toFixed(2);
                if (parseFloat(ahead) < 1.0) {
                    console.warn(`[SCHED] ⚠️ QUEUE EMPTY! ahead=${ahead}s`);
                }
            }
        }
        
        function startScheduler() {
            if (schedulerInterval) return;
            schedulerInterval = setInterval(() => {
                if (!playbackStarted || !audioContext) {
                    clearInterval(schedulerInterval);
                    schedulerInterval = null;
                    return;
                }
                scheduleBuffers();
            }, 25);
        }
        
        // Report stats to backend every 3s for monitoring
        setInterval(() => {
            if (playbackStarted && ws && ws.readyState === 1) {
                const ahead = nextPlayTime > 0 && audioContext ? (nextPlayTime - audioContext.currentTime).toFixed(2) : '0';
                ws.send(JSON.stringify({
                    action: 'frontend_stats',
                    stats: {
                        chunkCount,
                        gapCount,
                        queueLen: audioQueue.length,
                        bufferedSec: totalBufferedSeconds.toFixed(1),
                        scheduledAhead: ahead,
                        playbackStarted,
                        ctxTime: audioContext ? audioContext.currentTime.toFixed(2) : '0',
                        ctxState: audioContext ? audioContext.state : 'none',
                    }
                }));
            }
        }, 3000);
        
        function handleMP3Clip(data) {
            if (!audioContext) initAudio();
            
            const binaryString = atob(data.data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            audioContext.decodeAudioData(bytes.buffer, function(audioBuffer) {
                audioQueue.push(audioBuffer);
                totalBufferedSeconds += audioBuffer.duration;
                addLog(`🎵 Clip ${data.clip} decoded (${audioBuffer.duration.toFixed(1)}s)`, 'change');
                
                if (!playbackStarted && totalBufferedSeconds >= 2) {
                    playbackStarted = true;
                    nextPlayTime = audioContext.currentTime + 0.1;
                    scheduleBuffers();
                    addLog('🔊 Playback started!', 'change');
                }
            }, function(err) {
                addLog('❌ MP3 decode error: ' + err, 'error');
            });
        }
        
        function stopAudio() {
            if (AUDIO_DEBUG) console.log(`[AUDIO] === STOP === total chunks: ${chunkCount} | total gaps: ${gapCount} | queue remaining: ${audioQueue.length}`);
            audioQueue = [];
            playbackStarted = false;
            totalBufferedSeconds = 0;
            chunkCount = 0;
            gapCount = 0;
            lastChunkTime = 0;
            if (schedulerInterval) {
                clearInterval(schedulerInterval);
                schedulerInterval = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
        }
        
        connect();
    </script>
</body>
</html>
"""


async def run_dashboard_server():
    """Servidor WebSocket para el dashboard"""
    import websockets
    
    processor = EEGProcessor(sensitivity=2.5)
    state = {
        'music_gen': None,
        'music_gen_local': None,  # cached by model size
        'music_gen_online': None,
        'current_engine': None,
        'local_model_size': 'mrt2_small',  # default model for local engine
    }
    
    # Conectar MUSE (inicial, pero se reconecta dinámicamente después)
    eeg_state = {
        'inlet': None,
        'reconnecting': False,
        'muse_proc': None,
    }
    try:
        from pylsl import StreamInlet, resolve_byprop
        logger.info("🔍 Buscando MUSE...")
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if streams:
            eeg_state['inlet'] = StreamInlet(streams[0])
            logger.info(f"✅ MUSE conectado: {streams[0].name()}")
        else:
            logger.warning("⚠️ MUSE no encontrado — se reconectará automáticamente cuando detecte el stream")
    except Exception as e:
        logger.warning(f"⚠️ Error MUSE: {e}")
    
    connected_clients = set()
    eeg_buffer = []
    
    async def broadcast(message):
        if connected_clients:
            msg = json.dumps(message)
            await asyncio.gather(*[_safe_send(client, msg) for client in connected_clients], return_exceptions=True)
    
    async def send_muse_status(connected):
        await broadcast({'type': 'muse_status', 'connected': connected})
    
    _pending_sends = set()
    
    async def _safe_send(client, msg):
        try:
            await client.send(msg)
        except Exception:
            pass  # Client disconnected — silently ignore
    
    async def broadcast_fire(message):
        """Fire-and-forget broadcast for large messages like audio chunks"""
        if connected_clients:
            msg = json.dumps(message)
            tasks = [asyncio.ensure_future(_safe_send(client, msg)) for client in connected_clients]
            for t in tasks:
                _pending_sends.add(t)
                t.add_done_callback(_pending_sends.discard)
            await asyncio.sleep(0)
    
    async def broadcast_audio(chunk_id, raw_bytes, sample_rate, channels):
        """Send audio as binary WebSocket frame — much faster than base64+JSON"""
        if connected_clients:
            # Build a binary message: 4-byte header (chunk_id as uint32) + raw audio bytes
            import struct
            header = struct.pack('<I', chunk_id)  # 4 bytes little-endian
            payload = header + raw_bytes
            tasks = [asyncio.ensure_future(_safe_send_bin(client, payload)) for client in connected_clients]
            for t in tasks:
                _pending_sends.add(t)
                t.add_done_callback(_pending_sends.discard)
            await asyncio.sleep(0)
    
    async def _safe_send_bin(client, payload):
        try:
            await client.send(payload)
        except Exception:
            pass
    
    async def reconnect_muse():
        """Busca MUSE periódicamente si no hay conexión activa"""
        from pylsl import StreamInlet, resolve_byprop
        nonlocal eeg_buffer
        while True:
            if eeg_state['inlet'] is None and not eeg_state['reconnecting']:
                eeg_state['reconnecting'] = True
                try:
                    logger.info("🔄 Buscando stream MUSE...")
                    streams = resolve_byprop('type', 'EEG', timeout=3)
                    if streams:
                        inlet = StreamInlet(streams[0])
                        # Verify actual data is flowing before marking as connected
                        sample, _ = inlet.pull_sample(timeout=2.0)
                        if sample is not None and any(abs(v) > 0.1 for v in sample):
                            eeg_state['inlet'] = inlet
                            logger.info(f"✅ MUSE conectado: {streams[0].name()}")
                            await broadcast({'type': 'log', 'message': 'MUSE EEG conectado', 'level': 'change'})
                            await send_muse_status(True)
                        else:
                            logger.warning("⚠️ Stream found but no data flowing — is Muse on?")
                            await send_muse_status(False)
                    else:
                        logger.info("⏳ MUSE no detectado, reintentando en 3s...")
                except Exception as e:
                    logger.warning(f"⚠️ Error reconectando MUSE: {e}")
                finally:
                    eeg_state['reconnecting'] = False
            await asyncio.sleep(3)
    
    async def handler(websocket):
        connected_clients.add(websocket)
        logger.info(f"📱 Cliente conectado ({len(connected_clients)})")
        
        # Send current Muse status to new client
        await websocket.send(json.dumps({'type': 'muse_status', 'connected': eeg_state['inlet'] is not None}))
        
        audio_task = None
        
        try:
            # Tarea para procesar EEG
            async def process_eeg():
                nonlocal eeg_buffer
                last_update = 0
                empty_pulls = 0
                
                while True:
                    try:
                        inlet = eeg_state['inlet']
                        if inlet:
                            sample, _ = inlet.pull_sample(timeout=0.1)
                            if sample:
                                eeg_buffer.append(sample[:4])
                                empty_pulls = 0
                            else:
                                empty_pulls += 1
                                if empty_pulls > 30:
                                    logger.warning("⚠️ MUSE stream perdido, esperando reconexión...")
                                    eeg_state['inlet'] = None
                                    empty_pulls = 0
                                    await send_muse_status(False)
                                    await asyncio.sleep(1)
                        else:
                            # Datos simulados mientras se reconecta MUSE
                            t = time.time()
                            sample = [np.sin(2*np.pi*10*t + i*0.5) + np.random.randn()*0.1 for i in range(4)]
                            eeg_buffer.append(sample)
                            await asyncio.sleep(1/SAMPLE_RATE_EEG)
                        
                        if len(eeg_buffer) >= WINDOW_SIZE:
                            eeg_data = np.array(eeg_buffer).T
                            bands = processor.extract_bands(eeg_data)
                            
                            # Feed calibration if active
                            if processor._calibrating:
                                processor.feed_calibration(bands)
                            
                            metrics = processor.calculate_metrics(bands)
                            music_params = processor.map_to_lyria(bands, metrics)
                            prompts = processor.generate_prompts(music_params)
                            
                            # Convertir numpy types a Python nativos para JSON
                            def to_json_safe(obj):
                                if isinstance(obj, dict):
                                    return {k: to_json_safe(v) for k, v in obj.items()}
                                elif isinstance(obj, (np.bool_, np.integer)):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, bool):
                                    return obj
                                return obj
                            
                            # Enviar a clientes (fire-and-forget to not block audio)
                            await broadcast_fire({
                                'type': 'eeg',
                                'bands': to_json_safe(bands),
                                'metrics': to_json_safe(metrics),
                                'music_params': to_json_safe(music_params),
                                'prompt': prompts[0][0] if prompts else ''
                            })
                            
                            # Actualizar música si está generando
                            now = time.time()
                            mg = state['music_gen']
                            if mg and mg.is_generating and now - last_update >= 6.0:
                                last_update = now
                                await mg.update(music_params, prompts)
                                
                                # Enviar estado del buffer
                                await broadcast({
                                    'type': 'music_status',
                                    'buffer_progress': mg.buffer_progress,
                                    'buffer_ready': mg.buffer_ready
                                })
                            
                            eeg_buffer = eeg_buffer[WINDOW_SIZE // 2:]
                            
                    except Exception as e:
                        logger.error(f"Error EEG: {e}")
                        await asyncio.sleep(0.1)
            
            eeg_task = asyncio.create_task(process_eeg())
            
            # Recibir comandos
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    action = cmd.get('action')
                    
                    if action == 'start_music':
                        engine = cmd.get('engine', 'local')
                        
                        if state['music_gen'] and state['music_gen'].is_generating:
                            logger.warning("Music already generating, ignoring start")
                            continue
                        
                        if engine == 'online':
                            if state['music_gen_online'] is None:
                                try:
                                    state['music_gen_online'] = LyriaOnlineGenerator()
                                except Exception as e:
                                    logger.error(f"Failed to init Lyria online: {e}")
                                    await broadcast({'type': 'log', 'message': f'Error: {e}', 'level': 'error'})
                                    continue
                            state['music_gen'] = state['music_gen_online']
                        else:
                            model_size = cmd.get('model_size', state.get('local_model_size', 'mrt2_small'))
                            state['local_model_size'] = model_size
                            # Reuse cached instance only if same model size
                            cached = state.get('music_gen_local')
                            if cached is None or cached.model_size != model_size:
                                state['music_gen_local'] = MusicGenerator(model_size=model_size)
                            state['music_gen'] = state['music_gen_local']
                        
                        state['current_engine'] = engine
                        mg = state['music_gen']
                        mg.audio_callback = broadcast_fire  # for metadata messages
                        mg.audio_bin_callback = broadcast_audio  # for binary audio
                        success = await mg.start()
                        if success:
                            audio_task = asyncio.create_task(mg.receive_audio())
                            await broadcast({'type': 'music_started'})
                            logger.info(f"🎵 Music started with engine: {engine}")
                    
                    elif action == 'stop_music':
                        mg = state['music_gen']
                        if mg:
                            filepath = await mg.stop()
                        else:
                            filepath = None
                        if audio_task:
                            audio_task.cancel()
                        await broadcast({'type': 'music_stopped', 'file': filepath})
                    
                    elif action == 'reset_baseline':
                        processor.reset_baseline()
                        await broadcast({'type': 'log', 'message': 'Baseline reset', 'level': 'change'})
                    
                    elif action == 'set_genre':
                        genre = cmd.get('genre', '')
                        if genre:
                            processor._manual_genre = genre
                            await broadcast({'type': 'log', 'message': f'🎵 Genre override: {genre}', 'level': 'change'})
                        else:
                            processor._manual_genre = None
                            await broadcast({'type': 'log', 'message': '🎵 Genre: Auto (EEG-driven)', 'level': 'change'})
                    
                    elif action == 'calibrate':
                        processor.start_calibration()
                        await broadcast({'type': 'log', 'message': '📊 Calibration started — stay relaxed for 10s', 'level': 'change'})
                        # Auto-finish after 10s
                        async def _finish_cal():
                            await asyncio.sleep(10)
                            if processor._calibrating:
                                success = processor.finish_calibration()
                                if success:
                                    await broadcast({'type': 'log', 'message': f'✅ Baseline captured: {processor._band_baseline}', 'level': 'change'})
                                    await broadcast({'type': 'log', 'message': 'Music will now react to YOUR deviations from baseline', 'level': 'change'})
                                else:
                                    await broadcast({'type': 'log', 'message': '⚠️ Calibration failed — not enough data', 'level': 'error'})
                                await broadcast({'type': 'calibration_done', 'success': success})
                        asyncio.create_task(_finish_cal())
                    
                    elif action == 'start_muse':
                        import subprocess
                        if eeg_state.get('muse_proc') is not None and eeg_state['muse_proc'].poll() is None:
                            await broadcast({'type': 'log', 'message': 'Muse stream already running', 'level': 'error'})
                            continue
                        try:
                            proc = subprocess.Popen(
                                ['muselsl', 'stream'],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            eeg_state['muse_proc'] = proc
                            logger.info(f"📡 muselsl stream started (PID {proc.pid})")
                            await broadcast({'type': 'log', 'message': '📡 muselsl stream started, waiting for LSL...', 'level': 'change'})
                            
                            # Wait for LSL stream to appear (up to 10s)
                            from pylsl import resolve_byprop as _resolve
                            lsl_found = False
                            for attempt in range(5):
                                await asyncio.sleep(2)
                                if proc.poll() is not None:
                                    await broadcast({'type': 'log', 'message': '⚠️ muselsl stream exited — is Muse paired via Bluetooth?', 'level': 'error'})
                                    await send_muse_status(False)
                                    break
                                streams = _resolve('type', 'EEG', timeout=1)
                                if streams:
                                    inlet = StreamInlet(streams[0])
                                    # Verify actual data flows before marking connected
                                    sample, _ = inlet.pull_sample(timeout=2.0)
                                    if sample is not None and any(abs(v) > 0.1 for v in sample):
                                        eeg_state['inlet'] = inlet
                                        lsl_found = True
                                        logger.info(f"✅ MUSE LSL stream detected with live data")
                                        await broadcast({'type': 'log', 'message': '✅ Muse connected via LSL', 'level': 'change'})
                                        await send_muse_status(True)
                                        break
                                    else:
                                        logger.warning("⚠️ LSL stream found but no data — is Muse on?")
                                        await send_muse_status(False)
                            
                            if not lsl_found and proc.poll() is None:
                                await broadcast({'type': 'log', 'message': '⏳ muselsl running but no LSL stream yet. Check Bluetooth pairing.', 'level': 'error'})
                                await send_muse_status(False)
                        except FileNotFoundError:
                            await broadcast({'type': 'log', 'message': '❌ muselsl not found. Install with: pip install muselsl', 'level': 'error'})
                        except Exception as e:
                            await broadcast({'type': 'log', 'message': f'❌ Error starting muse: {e}', 'level': 'error'})
                    
                    elif action == 'update_settings':
                        settings = cmd.get('settings', {})
                        processor.update_settings(settings)
                        logger.info(f"🎚️ Settings updated: smoothing={processor.smoothing}, sensitivity={processor.sensitivity:.1f}")
                    
                    elif action == 'frontend_stats':
                        s = cmd.get('stats', {})
                        logger.info(f"📊 [FRONTEND] chunks={s.get('chunkCount',0)} gaps={s.get('gapCount',0)} queue={s.get('queueLen',0)} buffered={s.get('bufferedSec','0')}s ahead={s.get('scheduledAhead','0')}s ctxTime={s.get('ctxTime','0')}s ctxState={s.get('ctxState','?')}")
                    
                    elif action == 'throttle':
                        throttled = cmd.get('throttled', False)
                        mg = state['music_gen']
                        if mg and hasattr(mg, '_throttled'):
                            mg._throttled = throttled
                            if throttled:
                                logger.info("⏸️ Backend throttled — pausing generation (frontend buffer full)")
                            else:
                                logger.info("▶️ Backend unthrottled — resuming generation")
                        
                except Exception as e:
                    logger.error(f"Error comando: {e}")
            
        finally:
            connected_clients.remove(websocket)
            eeg_task.cancel()
            if audio_task:
                audio_task.cancel()
            logger.info(f"📱 Cliente desconectado ({len(connected_clients)})")
    
    reconnect_task = asyncio.create_task(reconnect_muse())
    server = await websockets.serve(handler, "localhost", 8767)
    logger.info("🌐 Dashboard server en ws://localhost:8767")
    await server.wait_closed()
    reconnect_task.cancel()


def save_dashboard():
    filepath = os.path.join(os.path.dirname(__file__), "music_dashboard.html")
    with open(filepath, 'w') as f:
        f.write(DASHBOARD_HTML)
    return filepath


def main():
    import webbrowser
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        🧠🎵 EEG MUSIC DASHBOARD 🎵🧠                     ║
    ║   Visualización + Control de Música en Tiempo Real       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    html_path = save_dashboard()
    print(f"📄 Dashboard: {html_path}")
    
    print("\n🔌 Asegúrate de que MUSE está conectado: muselsl stream")
    
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    print("\n🌐 Iniciando servidor...")
    asyncio.run(run_dashboard_server())


if __name__ == "__main__":
    main()
