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
        
        if arousal > 0.75:
            prompts.append(("driving drums, energetic pulse, tight groove", 1.0))
        elif arousal > 0.55:
            prompts.append(("steady rhythm, flowing movement, moderate energy", 1.0))
        elif arousal > 0.35:
            prompts.append(("gentle pulse, soft dynamics, relaxed tempo", 1.0))
        else:
            prompts.append(("slow sustained drones, minimal rhythm, spacious", 1.0))
        
        dominant = max(['delta', 'theta', 'alpha', 'beta', 'gamma'], key=lambda b: params.get(b, 0))
        instruments = {
            'delta': "deep bass, cello",
            'theta': "synth pads, hang drum",
            'alpha': "piano, acoustic guitar",
            'beta': "electric guitar, marimba",
            'gamma': "bright bells, glockenspiel"
        }
        prompts.append((instruments.get(dominant, "piano"), 0.8))
        
        if params.get('significant_change'):
            self.current_style_idx = (self.current_style_idx + 1) % 6
        
        styles = ["ambient electronic", "neo classical", "lo-fi chill", "cinematic orchestral", "jazz fusion", "minimal techno"]
        prompts.append((styles[self.current_style_idx], 0.4))
        
        return prompts
    
    def reset_baseline(self):
        """Reset baseline para recalibrar"""
        self.baseline_metrics = None
        self.baseline_samples = []
        self.prev_metrics = None


class MusicGenerator:
    """Generador de música con Lyria"""
    
    def __init__(self):
        self.is_generating = False
        self.session = None
        self.client = None
        self.audio_chunks = []
        self.current_params = None
        self.start_time = None
        self.buffer_ready = False
        self.buffer_progress = 0
        self.audio_callback = None  # Callback para enviar audio al navegador
    
    async def start(self):
        """Iniciar sesión de Lyria"""
        if self.is_generating:
            return False
        
        try:
            from google import genai
            from google.genai import types
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found")
                return False
            
            self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
            self._connection = self.client.aio.live.music.connect(model='models/lyria-realtime-exp')
            self.session = await self._connection.__aenter__()
            
            await self.session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text="gentle flowing music", weight=1.0)]
            )
            await self.session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=80, temperature=1.0)
            )
            
            await self.session.play()
            self.is_generating = True
            self.start_time = time.time()
            self.audio_chunks = []
            self.buffer_ready = False
            self.buffer_progress = 0
            
            logger.info("🎵 Lyria iniciado")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando Lyria: {e}")
            return False
    
    async def stop(self):
        """Detener sesión de Lyria"""
        if not self.is_generating:
            return None
        
        try:
            if self.session:
                await self.session.pause()
            if hasattr(self, '_connection') and self._connection:
                await self._connection.__aexit__(None, None, None)
            
            self.is_generating = False
            self.session = None
            
            # Guardar audio
            if self.audio_chunks:
                return self._save_audio()
            
        except Exception as e:
            logger.error(f"Error deteniendo Lyria: {e}")
        
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
    
    async def update(self, params: dict, prompts: list):
        """Actualizar parámetros de Lyria"""
        if not self.is_generating or not self.session:
            return
        
        try:
            from google.genai import types
            
            self.current_params = params
            
            lyria_prompts = [types.WeightedPrompt(text=t, weight=w) for t, w in prompts]
            await self.session.set_weighted_prompts(prompts=lyria_prompts)
            
            scale_enum = getattr(types.Scale, params['scale'], types.Scale.SCALE_UNSPECIFIED)
            config = types.LiveMusicGenerationConfig(
                bpm=params['bpm'],
                density=params['density'],
                brightness=params['brightness'],
                guidance=params['guidance'],
                temperature=params['temperature'],
                scale=scale_enum,
                mute_drums=params['mute_drums'],
                mute_bass=params['mute_bass'],
            )
            await self.session.set_music_generation_config(config=config)
            
        except Exception as e:
            logger.error(f"Error actualizando Lyria: {e}")
    
    async def receive_audio(self):
        """Recibir audio de Lyria y enviarlo al navegador"""
        if not self.session:
            logger.error("No session for receive_audio")
            return
        
        import base64
        chunk_count = 0
        
        try:
            logger.info("🎧 Starting audio receive loop...")
            async for message in self.session.receive():
                if not self.is_generating:
                    logger.info("Stopping - not generating")
                    break
                    
                # Log message type for debugging
                if chunk_count == 0:
                    logger.info(f"First message type: {type(message)}")
                    logger.info(f"Message attrs: {dir(message)}")
                    if hasattr(message, 'server_content'):
                        logger.info(f"server_content: {message.server_content}")
                
                if hasattr(message, 'server_content') and message.server_content:
                    if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                        for chunk in message.server_content.audio_chunks:
                            chunk_count += 1
                            self.audio_chunks.append(chunk.data)
                            
                            # Log every 10 chunks
                            if chunk_count % 10 == 0:
                                logger.info(f"🎵 Received {chunk_count} audio chunks ({len(chunk.data)} bytes each)")
                            
                            # Enviar audio al navegador si hay callback
                            if self.audio_callback:
                                audio_b64 = base64.b64encode(chunk.data).decode('utf-8')
                                await self.audio_callback({
                                    'type': 'audio',
                                    'data': audio_b64,
                                    'sample_rate': SAMPLE_RATE_AUDIO,
                                    'channels': CHANNELS
                                })
                            
                            # Calcular progreso del buffer (20s)
                            total_bytes = sum(len(c) for c in self.audio_chunks)
                            target_bytes = 20 * SAMPLE_RATE_AUDIO * CHANNELS * 2
                            self.buffer_progress = min(100, (total_bytes / target_bytes) * 100)
                            if self.buffer_progress >= 100:
                                self.buffer_ready = True
            
            logger.info(f"Audio receive loop ended. Total chunks: {chunk_count}")
        except Exception as e:
            logger.error(f"Error recibiendo audio: {e}")
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
                
                <div class="music-controls">
                    <button class="btn-start" id="btn-start" onclick="startMusic()">▶️ Start Music</button>
                    <button class="btn-stop" id="btn-stop" onclick="stopMusic()" disabled>⏹️ Stop</button>
                </div>
                
                <button class="btn-reset" onclick="resetBaseline()" style="width:100%; margin-bottom:15px;">🔄 Reset Baseline</button>
                
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
        
        function updateEEG(data) {
            const bands = data.bands;
            const metrics = data.metrics;
            const music = data.music_params;
            
            // Bands
            for (const band of ['delta', 'theta', 'alpha', 'beta', 'gamma']) {
                const val = bands[band] || 0;
                document.getElementById(`${band}-bar`).style.width = `${val * 100}%`;
                document.getElementById(`${band}-val`).textContent = val.toFixed(2);
                bandHistory[band].push(val);
                if (bandHistory[band].length > maxHistory) bandHistory[band].shift();
            }
            
            // Chart
            bandChart.data.datasets[0].data = bandHistory.delta;
            bandChart.data.datasets[1].data = bandHistory.theta;
            bandChart.data.datasets[2].data = bandHistory.alpha;
            bandChart.data.datasets[3].data = bandHistory.beta;
            bandChart.data.datasets[4].data = bandHistory.gamma;
            bandChart.update('none');
            
            // Metrics
            const adj_a = metrics.arousal_adjusted || metrics.arousal;
            const adj_v = metrics.valence_adjusted || metrics.valence;
            document.getElementById('arousal-val').textContent = adj_a.toFixed(2);
            document.getElementById('valence-val').textContent = adj_v.toFixed(2);
            document.getElementById('focus-val').textContent = (metrics.focus_adjusted || metrics.focus).toFixed(2);
            document.getElementById('relax-val').textContent = (metrics.relaxation_adjusted || metrics.relaxation).toFixed(2);
            
            // Dominant band
            const bandMap = {delta: 'δ', theta: 'θ', alpha: 'α', beta: 'β', gamma: 'γ'};
            const dominant = Object.keys(bands).reduce((a, b) => bands[a] > bands[b] ? a : b);
            document.getElementById('dominant-val').textContent = bandMap[dominant];
            
            // Change indicator
            document.getElementById('change-val').textContent = metrics.significant_change ? '🔄' : '-';
            
            // State
            updateState(metrics);
            
            // Music params
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
        
        function startMusic() {
            ws.send(JSON.stringify({ action: 'start_music' }));
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
        
        // ============ SENSITIVITY CONTROLS ============
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
            
            // Update display values
            document.getElementById('sens-delta-val').textContent = settings.band_sensitivity.delta.toFixed(1) + 'x';
            document.getElementById('sens-theta-val').textContent = settings.band_sensitivity.theta.toFixed(1) + 'x';
            document.getElementById('sens-alpha-val').textContent = settings.band_sensitivity.alpha.toFixed(1) + 'x';
            document.getElementById('sens-beta-val').textContent = settings.band_sensitivity.beta.toFixed(1) + 'x';
            document.getElementById('sens-gamma-val').textContent = settings.band_sensitivity.gamma.toFixed(1) + 'x';
            document.getElementById('smoothing-val').textContent = settings.smoothing;
            document.getElementById('global-sens-val').textContent = settings.global_sensitivity.toFixed(1) + 'x';
            
            // Send to server
            ws.send(JSON.stringify({ action: 'update_settings', settings: settings }));
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
        const BUFFER_SECONDS = 5; // Buffer before starting playback
        let totalBufferedSeconds = 0;
        let playbackStarted = false;
        
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
            
            // Decode base64 to ArrayBuffer
            const binaryString = atob(data.data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Convert Int16 PCM to Float32
            const int16Array = new Int16Array(bytes.buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }
            
            // Create audio buffer (stereo)
            const numSamples = float32Array.length / 2;
            const audioBuffer = audioContext.createBuffer(2, numSamples, 48000);
            
            // Deinterleave stereo channels
            const leftChannel = audioBuffer.getChannelData(0);
            const rightChannel = audioBuffer.getChannelData(1);
            for (let i = 0; i < numSamples; i++) {
                leftChannel[i] = float32Array[i * 2];
                rightChannel[i] = float32Array[i * 2 + 1];
            }
            
            // Add to queue
            audioQueue.push(audioBuffer);
            totalBufferedSeconds += audioBuffer.duration;
            
            // Start playback after buffering enough
            if (!playbackStarted && totalBufferedSeconds >= BUFFER_SECONDS) {
                playbackStarted = true;
                nextPlayTime = audioContext.currentTime + 0.1;
                scheduleBuffers();
                addLog('🔊 Playback started!', 'change');
            }
        }
        
        function scheduleBuffers() {
            while (audioQueue.length > 0 && nextPlayTime < audioContext.currentTime + 2) {
                const buffer = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start(nextPlayTime);
                nextPlayTime += buffer.duration;
            }
            
            if (isGenerating || audioQueue.length > 0) {
                setTimeout(scheduleBuffers, 100);
            }
        }
        
        function stopAudio() {
            audioQueue = [];
            playbackStarted = false;
            totalBufferedSeconds = 0;
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
    music_gen = MusicGenerator()
    
    # Conectar MUSE
    inlet = None
    try:
        from pylsl import StreamInlet, resolve_byprop
        logger.info("🔍 Buscando MUSE...")
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if streams:
            inlet = StreamInlet(streams[0])
            logger.info(f"✅ MUSE conectado: {streams[0].name()}")
        else:
            logger.warning("⚠️ MUSE no encontrado")
    except Exception as e:
        logger.warning(f"⚠️ Error MUSE: {e}")
    
    connected_clients = set()
    eeg_buffer = []
    
    async def broadcast(message):
        if connected_clients:
            await asyncio.gather(*[client.send(json.dumps(message)) for client in connected_clients])
    
    async def handler(websocket):
        connected_clients.add(websocket)
        logger.info(f"📱 Cliente conectado ({len(connected_clients)})")
        
        audio_task = None
        
        try:
            # Tarea para procesar EEG
            async def process_eeg():
                nonlocal eeg_buffer
                last_update = 0
                
                while True:
                    try:
                        if inlet:
                            sample, _ = inlet.pull_sample(timeout=0.1)
                            if sample:
                                eeg_buffer.append(sample[:4])
                        else:
                            # Datos simulados
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
                            
                            # Enviar a clientes
                            await broadcast({
                                'type': 'eeg',
                                'bands': to_json_safe(bands),
                                'metrics': to_json_safe(metrics),
                                'music_params': to_json_safe(music_params),
                                'prompt': prompts[0][0] if prompts else ''
                            })
                            
                            # Actualizar Lyria si está generando
                            now = time.time()
                            if music_gen.is_generating and now - last_update >= 1.5:
                                last_update = now
                                await music_gen.update(music_params, prompts)
                                
                                # Enviar estado del buffer
                                await broadcast({
                                    'type': 'music_status',
                                    'buffer_progress': music_gen.buffer_progress,
                                    'buffer_ready': music_gen.buffer_ready
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
                        # Configurar callback para enviar audio al navegador
                        music_gen.audio_callback = broadcast
                        success = await music_gen.start()
                        if success:
                            audio_task = asyncio.create_task(music_gen.receive_audio())
                            await broadcast({'type': 'music_started'})
                    
                    elif action == 'stop_music':
                        filepath = await music_gen.stop()
                        if audio_task:
                            audio_task.cancel()
                        await broadcast({'type': 'music_stopped', 'file': filepath})
                    
                    elif action == 'reset_baseline':
                        processor.reset_baseline()
                        await broadcast({'type': 'log', 'message': 'Baseline reset', 'level': 'change'})
                    
                    elif action == 'update_settings':
                        settings = cmd.get('settings', {})
                        processor.update_settings(settings)
                        logger.info(f"🎚️ Settings updated: smoothing={processor.smoothing}, sensitivity={processor.sensitivity:.1f}")
                        
                except Exception as e:
                    logger.error(f"Error comando: {e}")
            
        finally:
            connected_clients.remove(websocket)
            eeg_task.cancel()
            if audio_task:
                audio_task.cancel()
            logger.info(f"📱 Cliente desconectado ({len(connected_clients)})")
    
    server = await websockets.serve(handler, "localhost", 8767)
    logger.info("🌐 Dashboard server en ws://localhost:8767")
    await server.wait_closed()


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
