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
        
        arousal = np.clip(0.1*d + 0.2*t + 0.3*a + 0.5*b + 0.7*g, 0, 1)
        valence = np.clip((a - 0.3*t - 0.2*abs(b-t) + 0.5), 0, 1)
        focus = np.clip(b / (t + 0.01) / 3, 0, 1)
        relaxation = np.clip(a / (b + 0.01) / 2, 0, 1)
        
        metrics = {'arousal': arousal, 'valence': valence, 'focus': focus, 'relaxation': relaxation, **bands}
        
        # Baseline
        if self.baseline_metrics is None:
            self.baseline_samples.append(metrics.copy())
            if len(self.baseline_samples) >= 5:
                self.baseline_metrics = {
                    k: np.mean([s[k] for s in self.baseline_samples])
                    for k in ['arousal', 'valence', 'focus', 'relaxation']
                }
        
        # Cambios relativos
        if self.baseline_metrics:
            for key in ['arousal', 'valence', 'focus', 'relaxation']:
                delta = metrics[key] - self.baseline_metrics[key]
                metrics[f'{key}_adjusted'] = np.clip(
                    self.baseline_metrics[key] + delta * self.sensitivity, 0, 1
                )
        
        # Detectar cambios
        metrics['significant_change'] = False
        if self.prev_metrics:
            for key in ['arousal', 'valence']:
                if abs(metrics[key] - self.prev_metrics[key]) > 0.05:
                    metrics['significant_change'] = True
                    break
        
        self.prev_metrics = metrics.copy()
        return metrics
    
    def map_to_lyria(self, bands: dict, metrics: dict) -> dict:
        arousal = metrics.get('arousal_adjusted', metrics['arousal'])
        valence = metrics.get('valence_adjusted', metrics['valence'])
        focus = metrics.get('focus_adjusted', metrics['focus'])
        relaxation = metrics.get('relaxation_adjusted', metrics['relaxation'])
        
        delta = bands.get('delta', 0.2)
        theta = bands.get('theta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        bpm = int(60 + arousal * 140)
        density = np.clip(0.2 + arousal * 0.6 - relaxation * 0.3, 0, 1)
        brightness = np.clip(gamma * 2.0 + valence * 0.4, 0, 1)
        guidance = 1.5 + focus * 4.0
        temperature = 0.6 + theta * 2.0
        
        # Scale
        if valence > 0.6:
            scale = 'D_MAJOR_B_MINOR' if arousal > 0.6 else ('G_MAJOR_E_MINOR' if arousal > 0.4 else 'C_MAJOR_A_MINOR')
        elif valence > 0.4:
            scale = 'A_MAJOR_G_FLAT_MINOR' if arousal > 0.5 else 'F_MAJOR_D_MINOR'
        else:
            scale = 'E_FLAT_MAJOR_C_MINOR' if arousal > 0.6 else ('B_FLAT_MAJOR_G_MINOR' if arousal > 0.4 else 'A_FLAT_MAJOR_F_MINOR')
        
        return {
            'bpm': max(60, min(200, bpm)),
            'density': max(0, min(1, density)),
            'brightness': max(0, min(1, brightness)),
            'guidance': max(0, min(6, guidance)),
            'temperature': max(0, min(3, temperature)),
            'scale': scale,
            'mute_drums': relaxation > 0.65,
            'mute_bass': delta > 0.35,
            **metrics
        }
    
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
    
    def __init__(self):
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
        self._tflite_lock = threading.Lock()  # TFLite interpreters aren't thread-safe
        self._throttled = False  # Backend throttle flag set by frontend
        
        logger.info("🧠 Cargando modelo local Magenta RealTime 2 (mrt2_small)...")
        from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
        self.mrt = MagentaRT2SystemMlxfn(size='mrt2_small')
        self.state = None
        logger.info("✅ Modelo local Magenta RealTime 2 listo.")
        
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
        """Build a continuous vibe prompt from EEG bands — no discrete thresholds.
        Each band contributes proportionally to the musical texture."""
        delta = params.get('delta', 0.2)
        theta = params.get('theta', 0.2)
        alpha = params.get('alpha', 0.2)
        beta = params.get('beta', 0.2)
        gamma = params.get('gamma', 0.2)
        arousal = params.get('arousal_adjusted', params.get('arousal', 0.5))
        valence = params.get('valence_adjusted', params.get('valence', 0.5))
        relaxation = params.get('relaxation_adjusted', params.get('relaxation', 0.5))
        focus = params.get('focus_adjusted', params.get('focus', 0.5))
        
        # --- Energy layer (continuous) ---
        energy_words = []
        if arousal > 0.05:
            energy_words.append("sustained tones")
        if arousal > 0.25:
            energy_words.append("gentle pulse")
        if arousal > 0.45:
            energy_words.append("flowing rhythm")
        if arousal > 0.65:
            energy_words.append("driving groove")
        if arousal > 0.8:
            energy_words.append("intense energy")
        energy_str = ", ".join(energy_words[-2:]) if energy_words else "calm"
        
        # --- Mood layer (valence: dark <-> bright, continuous blend) ---
        if valence < 0.5:
            mood_str = "contemplative, introspective"
            if valence < 0.3:
                mood_str += ", melancholic"
        else:
            mood_str = "warm, open"
            if valence > 0.7:
                mood_str += ", uplifting"
        
        # --- Texture layer (from individual bands, weighted blend) ---
        texture_parts = []
        
        # Delta: depth, gravity, bass presence
        if delta > 0.15:
            weight = min(delta * 3, 1.0)
            texture_parts.append(f"deep bass drones ({weight:.0%})")
        
        # Theta: dreamy, surreal, creative
        if theta > 0.15:
            weight = min(theta * 3, 1.0)
            texture_parts.append(f"dreamy drifting atmosphere ({weight:.0%})")
        
        # Alpha: smooth, flowing, melodic
        if alpha > 0.15:
            weight = min(alpha * 3, 1.0)
            texture_parts.append(f"smooth flowing melodies ({weight:.0%})")
        
        # Beta: structured, precise, rhythmic
        if beta > 0.15:
            weight = min(beta * 3, 1.0)
            texture_parts.append(f"precise rhythmic patterns ({weight:.0%})")
        
        # Gamma: sparkling, brilliant, complex
        if gamma > 0.1:
            weight = min(gamma * 4, 1.0)
            texture_parts.append(f"sparkling shimmering details ({weight:.0%})")
        
        texture_str = ", ".join(texture_parts) if texture_parts else "balanced texture"
        
        # --- Space layer (relaxation vs focus) ---
        if relaxation > 0.6:
            space_str = "spacious, reverb-drenched, ethereal"
        elif focus > 0.6:
            space_str = "tight, focused, intimate"
        else:
            space_str = "natural room ambience"
        
        # --- Instrument layer (dominant band selects timbre) ---
        bands = {'delta': delta, 'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
        dominant = max(bands, key=bands.get)
        instrument_map = {
            'delta': "contrabass, cello, sub-bass",
            'theta': "hang drum, kalimba, celestial pads",
            'alpha': "grand piano, nylon guitar, warm rhodes",
            'beta': "marimba, vibraphone, electric guitar",
            'gamma': "bells, chimes, harp, digital synths",
        }
        instruments = instrument_map.get(dominant, "piano")
        
        # Combine all layers
        prompt = f"{energy_str}, {mood_str}, {texture_str}, {space_str}, {instruments}, instrumental, no vocals"
        return prompt
    
    async def update(self, params: dict, prompts: list):
        """Actualizar estilo de Magenta con mapping continuo y transiciones suaves"""
        if not self.is_generating:
            return
            
        try:
            self.current_params = params
            
            # --- Continuous parameter mapping (no thresholds) ---
            delta = params.get('delta', 0.2)
            theta = params.get('theta', 0.2)
            alpha = params.get('alpha', 0.2)
            beta = params.get('beta', 0.2)
            gamma = params.get('gamma', 0.2)
            arousal = params.get('arousal_adjusted', params.get('arousal', 0.5))
            relaxation = params.get('relaxation_adjusted', params.get('relaxation', 0.5))
            focus = params.get('focus_adjusted', params.get('focus', 0.5))
            
            # Amplify band differences: normalize each band relative to its running mean
            # This makes small EEG fluctuations produce large musical changes
            
            # Temperature: wide range, driven by theta/alpha contrast
            # theta high = creative/unpredictable, alpha high = stable/consonant
            # arousal adds energy, delta adds gravity
            # Cap at 2.0 — higher values cause noise/silence from Magenta
            theta_alpha_ratio = theta / (alpha + 0.01)
            target_temp = np.clip(
                0.5 + theta_alpha_ratio * 1.2 + arousal * 0.8 - delta * 0.4,
                0.3, 2.0
            )
            
            # Top-k: gamma/beta contrast drives variety vs precision
            # Cap at 50 — higher values with high temp cause noise
            gamma_beta_ratio = gamma / (beta + 0.01)
            target_top_k = int(np.clip(
                10 + gamma_beta_ratio * 30 + arousal * 20 - focus * 15,
                5, 50
            ))
            
            # Faster interpolation — respond to brain changes quickly
            temp_blend = 0.5
            self._gen_temperature = self._gen_temperature * (1 - temp_blend) + target_temp * temp_blend
            self._gen_top_k = int(self._gen_top_k * (1 - temp_blend) + target_top_k * temp_blend)
            
            # CFG guidance: how strongly generation follows the style prompt.
            # Default (3.0) is weak -> generic/flat sound. Focus+arousal push it
            # up to 6.5 for punchier, more expressive adherence to the vibe.
            target_cfg = np.clip(2.5 + focus * 2.5 + arousal * 2.0, 2.5, 6.5)
            self._gen_cfg_musiccoca = self._gen_cfg_musiccoca * (1 - temp_blend) + target_cfg * temp_blend
            
            # Drums: always on to maintain rhythmic presence.
            # None can cause Magenta to generate silence/ambient noise.
            self._gen_drums = [1]
            
            # --- Build vibe prompt and compute target style embedding ---
            vibe_prompt = self._build_vibe_prompt(params)
            self._vibe_history.append(vibe_prompt)
            
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
                
                with self._tflite_lock:
                    waveform, self.state = self.mrt.generate(
                        style=self.current_style_embedding,
                        drums=self._gen_drums,
                        frames=frames_per_step,
                        state=self.state,
                        temperature=self._gen_temperature,
                        top_k=self._gen_top_k,
                        cfg_musiccoca=self._gen_cfg_musiccoca
                    )
                
                generation_time = time.time() - t_start
                realtime_ratio = step_duration / generation_time if generation_time > 0 else 0
                
                samples_int16 = (waveform.samples * 32767.0).astype(np.int16)
                chunk_data = samples_int16.tobytes()
                
                chunk_count += 1
                self.audio_chunks.append(chunk_data)
                
                if chunk_count % 50 == 0:
                    logger.info(f"🎵 Chunk {chunk_count}: {generation_time:.2f}s gen / {step_duration:.1f}s audio (ratio {realtime_ratio:.2f}x) | temp={self._gen_temperature:.2f} top_k={self._gen_top_k} cfg={self._gen_cfg_musiccoca:.2f} drums={self._gen_drums}")
                    
                if self.audio_callback:
                    audio_b64 = base64.b64encode(chunk_data).decode('utf-8')
                    await self.audio_callback({
                        'type': 'audio',
                        'data': audio_b64,
                        'sample_rate': SAMPLE_RATE_AUDIO,
                        'channels': CHANNELS,
                        'chunk_id': chunk_count,
                    })
                    
                total_bytes = sum(len(c) for c in self.audio_chunks)
                target_bytes = 20 * SAMPLE_RATE_AUDIO * CHANNELS * 2
                self.buffer_progress = min(100, (total_bytes / target_bytes) * 100)
                if self.buffer_progress >= 100:
                    self.buffer_ready = True
                
                # If frontend says we have enough buffered, wait before generating more
                if self._throttled:
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0)
                
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
                
                <div class="music-controls">
                    <button class="btn-start" id="btn-start" onclick="startMusic()" disabled>▶️ Start Music</button>
                    <button class="btn-stop" id="btn-stop" onclick="stopMusic()" disabled>⏹️ Stop</button>
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
            
            ws.onmessage = (event) => {
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
            addLog('Engine: ' + (engine === 'local' ? 'Local (Magenta)' : 'Online (Lyria)'));
        }
        
        function startMusic() {
            ws.send(JSON.stringify({ action: 'start_music', engine: selectedEngine }));
            addLog('Starting music generation...');
        }
        
        function stopMusic() {
            ws.send(JSON.stringify({ action: 'stop_music' }));
            addLog('Stopping music...');
        }
        
        function resetBaseline() {
            ws.send(JSON.stringify({ action: 'reset_baseline' }));
            addLog('🔄 Resetting baseline...', 'change');
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
        const BUFFER_SECONDS = 6; // Buffer before starting playback
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
        
        function handleAudioChunk(data) {
            if (!audioContext) initAudio();
            
            const now = performance.now();
            chunkCount++;
            const interChunkMs = lastChunkTime > 0 ? (now - lastChunkTime).toFixed(0) : '-';
            lastChunkTime = now;
            
            // Fast base64 decode: use atob + Uint8Array from char codes in chunks
            const binaryString = atob(data.data);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i += 8192) {
                const end = Math.min(i + 8192, len);
                for (let j = i; j < end; j++) {
                    bytes[j] = binaryString.charCodeAt(j);
                }
            }
            
            // Convert Int16 PCM to Float32 stereo using DataView (fast, no intermediate array)
            const int16View = new Int16Array(bytes.buffer);
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
                addLog('🔊 Playback started!', 'change');
                if (AUDIO_DEBUG) console.log(`[AUDIO] === PLAYBACK STARTED === queue=${audioQueue.length} buffered=${totalBufferedSeconds.toFixed(1)}s`);
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
                nextPlayTime = audioContext.currentTime + 0.05;
            }
            // Schedule up to 3s ahead for low latency
            const LOOKAHEAD = 3.0;
            const CROSSFADE_MS = 10;
            const crossfadeTime = CROSSFADE_MS / 1000;
            let scheduledThisRun = 0;
            while (audioQueue.length > 0 && nextPlayTime < audioContext.currentTime + LOOKAHEAD) {
                const buffer = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                
                // Crossfade: fade-in at start, fade-out at end
                const gain = audioContext.createGain();
                const startTime = nextPlayTime;
                const endTime = startTime + buffer.duration;
                
                gain.gain.setValueAtTime(0, startTime);
                gain.gain.linearRampToValueAtTime(1, startTime + crossfadeTime);
                gain.gain.setValueAtTime(1, endTime - crossfadeTime);
                gain.gain.linearRampToValueAtTime(0, endTime);
                
                source.connect(gain);
                gain.connect(audioContext.destination);
                source.start(startTime);
                source.onended = function() {
                    gain.disconnect();
                    source.disconnect();
                };
                nextPlayTime = endTime - crossfadeTime;
                scheduledThisRun++;
            }
            if (AUDIO_DEBUG && scheduledThisRun > 0 && chunkCount % 10 === 0) {
                const ahead = (nextPlayTime - audioContext.currentTime).toFixed(2);
                console.log(`[SCHED] scheduled ${scheduledThisRun} | queue: ${audioQueue.length} | ahead: ${ahead}s | gaps: ${gapCount}`);
            }
            // Throttle backend: tell it to pause if queue is large, resume if small
            if (ws && ws.readyState === 1) {
                if (audioQueue.length > 10 && !backendThrottled) {
                    backendThrottled = true;
                    ws.send(JSON.stringify({ action: 'throttle', throttled: true }));
                if (AUDIO_DEBUG) console.log('[THROTTLE] telling backend to PAUSE generation');
                } else if (audioQueue.length < 5 && backendThrottled) {
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
            }, 50);
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
        'music_gen_local': None,
        'music_gen_online': None,
        'current_engine': None,
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
                        eeg_state['inlet'] = StreamInlet(streams[0])
                        logger.info(f"✅ MUSE conectado: {streams[0].name()}")
                        await broadcast({'type': 'log', 'message': 'MUSE EEG conectado', 'level': 'change'})
                        await send_muse_status(True)
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
                            if mg and mg.is_generating and now - last_update >= 3.0:
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
                            if state['music_gen_local'] is None:
                                state['music_gen_local'] = MusicGenerator()
                            state['music_gen'] = state['music_gen_local']
                        
                        state['current_engine'] = engine
                        mg = state['music_gen']
                        mg.audio_callback = broadcast_fire
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
                                    eeg_state['inlet'] = StreamInlet(streams[0])
                                    lsl_found = True
                                    logger.info(f"✅ MUSE LSL stream detected after muselsl start")
                                    await broadcast({'type': 'log', 'message': '✅ Muse connected via LSL', 'level': 'change'})
                                    await send_muse_status(True)
                                    break
                            
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
