#!/usr/bin/env python3
"""
EEG → Lyria LIVE Music Generator
Genera y reproduce música en TIEMPO REAL controlada por tu cerebro
"""

import os
import asyncio
import wave
import time
import numpy as np
import sounddevice as sd
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
import logging
import threading
import queue

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuración Audio
SAMPLE_RATE_AUDIO = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
BLOCKSIZE = 4096

# Configuración EEG
SAMPLE_RATE_EEG = 256
WINDOW_SIZE = 256
OUTPUT_DIR = "output"

# Bandas EEG
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


class AudioPlayer:
    """
    Reproductor de audio en tiempo real con buffer circular.
    Usa un buffer continuo de bytes para evitar cortes y aceleración.
    """
    
    def __init__(self, sample_rate=48000, channels=2, buffer_seconds=20):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.bytes_per_frame = channels * 2  # 16-bit stereo = 4 bytes/frame
        self.bytes_per_second = sample_rate * self.bytes_per_frame
        
        # Buffer circular grande (buffer_seconds + 10s extra)
        self.buffer_size = (buffer_seconds + 30) * self.bytes_per_second
        self.audio_buffer = bytearray(self.buffer_size)
        self.write_pos = 0  # Posición de escritura
        self.read_pos = 0   # Posición de lectura
        self.available_bytes = 0  # Bytes disponibles para leer
        self.lock = threading.Lock()
        
        self.is_playing = False
        self.is_buffering = True
        self.stream = None
        self.all_audio = []  # Para guardar al final
        
        self.target_buffer_bytes = buffer_seconds * self.bytes_per_second
        
    def start(self):
        """Iniciar en modo buffering"""
        self.is_playing = True
        self.is_buffering = True
        self.write_pos = 0
        self.read_pos = 0
        self.available_bytes = 0
        logger.info(f"⏳ Buffering {self.buffer_seconds}s de audio antes de reproducir...")
    
    def _start_playback(self):
        """Iniciar reproducción después del buffering"""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            blocksize=2048,  # Bloques más pequeños para menor latencia
            callback=self._audio_callback,
            latency='low'
        )
        self.stream.start()
        logger.info("🔊 ¡Reproducción iniciada!")
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback para sounddevice - lee del buffer circular"""
        bytes_needed = frames * self.bytes_per_frame
        
        with self.lock:
            if self.available_bytes >= bytes_needed:
                # Leer del buffer circular
                data = bytearray(bytes_needed)
                
                # Manejar wrap-around del buffer circular
                first_chunk = min(bytes_needed, self.buffer_size - self.read_pos)
                data[:first_chunk] = self.audio_buffer[self.read_pos:self.read_pos + first_chunk]
                
                if first_chunk < bytes_needed:
                    # Wrap around
                    remaining = bytes_needed - first_chunk
                    data[first_chunk:] = self.audio_buffer[:remaining]
                    self.read_pos = remaining
                else:
                    self.read_pos = (self.read_pos + bytes_needed) % self.buffer_size
                
                self.available_bytes -= bytes_needed
                
                # Convertir a numpy array
                audio_array = np.frombuffer(bytes(data), dtype=np.int16).reshape(-1, self.channels)
                outdata[:] = audio_array
            else:
                # Buffer underrun - silencio
                outdata[:] = 0
    
    def add_audio(self, audio_bytes):
        """Añadir audio al buffer circular"""
        if not self.is_playing:
            return
        
        self.all_audio.append(audio_bytes)
        data_len = len(audio_bytes)
        
        with self.lock:
            # Escribir al buffer circular
            first_chunk = min(data_len, self.buffer_size - self.write_pos)
            self.audio_buffer[self.write_pos:self.write_pos + first_chunk] = audio_bytes[:first_chunk]
            
            if first_chunk < data_len:
                # Wrap around
                remaining = data_len - first_chunk
                self.audio_buffer[:remaining] = audio_bytes[first_chunk:]
                self.write_pos = remaining
            else:
                self.write_pos = (self.write_pos + data_len) % self.buffer_size
            
            self.available_bytes += data_len
        
        if self.is_buffering:
            # Mostrar progreso
            progress = min(100, (self.available_bytes / self.target_buffer_bytes) * 100)
            buffered_seconds = self.available_bytes // self.bytes_per_second
            print(f"\r⏳ Buffering: {progress:.0f}% ({buffered_seconds}s / {self.buffer_seconds}s)", end='', flush=True)
            
            # Iniciar reproducción cuando tengamos suficiente buffer
            if self.available_bytes >= self.target_buffer_bytes:
                print()
                self.is_buffering = False
                self._start_playback()
    
    def stop(self):
        """Detener reproducción"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("🔇 Audio playback detenido")
    
    def save(self, filename):
        """Guardar todo el audio a archivo"""
        if not self.all_audio:
            return None
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        audio_data = b''.join(self.all_audio)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        duration = len(audio_data) / (self.sample_rate * 2 * self.channels)
        logger.info(f"💾 Audio guardado: {filepath} ({duration:.1f}s)")
        return filepath


class EEGProcessor:
    """
    Procesador de señales EEG con detección de cambios.
    Más sensible a variaciones en el estado cerebral.
    """
    
    def __init__(self, sensitivity=2.0):
        self.band_history = {band: deque(maxlen=3) for band in EEG_BANDS}  # Menos suavizado
        self.sensitivity = sensitivity  # Multiplicador de cambios
        
        # Historial para detectar cambios
        self.prev_metrics = None
        self.baseline_metrics = None
        self.samples_for_baseline = 5
        self.baseline_samples = []
        
        # Estilo actual (para variedad)
        self.current_style_idx = 0
    
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
        
        for band, power in band_powers.items():
            self.band_history[band].append(power)
            band_powers[band] = np.mean(list(self.band_history[band]))
        
        return band_powers
    
    def calculate_metrics(self, bands: dict) -> dict:
        d, t, a, b, g = [bands.get(x, 0.2) for x in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
        
        arousal = np.clip(0.1*d + 0.2*t + 0.3*a + 0.5*b + 0.7*g, 0, 1)
        valence = np.clip((a - 0.3*t - 0.2*abs(b-t) + 0.5), 0, 1)
        focus = np.clip(b / (t + 0.01) / 3, 0, 1)
        relaxation = np.clip(a / (b + 0.01) / 2, 0, 1)
        
        metrics = {'arousal': arousal, 'valence': valence, 'focus': focus, 'relaxation': relaxation, **bands}
        
        # Establecer baseline con las primeras muestras
        if self.baseline_metrics is None:
            self.baseline_samples.append(metrics.copy())
            if len(self.baseline_samples) >= self.samples_for_baseline:
                self.baseline_metrics = {
                    k: np.mean([s[k] for s in self.baseline_samples])
                    for k in ['arousal', 'valence', 'focus', 'relaxation']
                }
                logger.info(f"📊 Baseline establecido: A={self.baseline_metrics['arousal']:.2f}")
        
        # Calcular cambios relativos al baseline (más sensible)
        if self.baseline_metrics:
            for key in ['arousal', 'valence', 'focus', 'relaxation']:
                delta = metrics[key] - self.baseline_metrics[key]
                # Amplificar cambios con sensibilidad
                metrics[f'{key}_delta'] = delta * self.sensitivity
                # Valor ajustado = baseline + delta amplificado
                metrics[f'{key}_adjusted'] = np.clip(
                    self.baseline_metrics[key] + delta * self.sensitivity, 0, 1
                )
        
        # Detectar cambios significativos
        metrics['significant_change'] = False
        if self.prev_metrics:
            for key in ['arousal', 'valence']:
                if abs(metrics[key] - self.prev_metrics[key]) > 0.05:
                    metrics['significant_change'] = True
                    break
        
        self.prev_metrics = metrics.copy()
        return metrics
    
    def map_to_lyria(self, bands: dict, metrics: dict) -> dict:
        # Usar valores ajustados (más sensibles) si están disponibles
        arousal = metrics.get('arousal_adjusted', metrics['arousal'])
        valence = metrics.get('valence_adjusted', metrics['valence'])
        focus = metrics.get('focus_adjusted', metrics['focus'])
        relaxation = metrics.get('relaxation_adjusted', metrics['relaxation'])
        
        delta = bands.get('delta', 0.2)
        theta = bands.get('theta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        # Mapeos con rangos más amplios
        bpm = int(60 + arousal * 140)
        density = np.clip(0.2 + arousal * 0.6 - relaxation * 0.3, 0, 1)
        brightness = np.clip(gamma * 2.0 + valence * 0.4, 0, 1)
        guidance = 1.5 + focus * 4.0
        temperature = 0.6 + theta * 2.0
        scale = self._select_scale(valence, arousal)
        
        # Controles de textura más dinámicos
        mute_drums = relaxation > 0.65
        mute_bass = delta > 0.35
        only_bass_drums = focus > 0.7 and arousal > 0.5
        
        return {
            'bpm': max(60, min(200, bpm)),
            'density': max(0, min(1, density)),
            'brightness': max(0, min(1, brightness)),
            'guidance': max(0, min(6, guidance)),
            'temperature': max(0, min(3, temperature)),
            'scale': scale,
            'mute_drums': mute_drums,
            'mute_bass': mute_bass,
            'only_bass_and_drums': only_bass_drums,
            **metrics
        }
    
    def _select_scale(self, valence: float, arousal: float) -> str:
        if valence > 0.6:
            return 'D_MAJOR_B_MINOR' if arousal > 0.6 else ('G_MAJOR_E_MINOR' if arousal > 0.4 else 'C_MAJOR_A_MINOR')
        elif valence > 0.4:
            return 'A_MAJOR_G_FLAT_MINOR' if arousal > 0.5 else 'F_MAJOR_D_MINOR'
        else:
            return 'E_FLAT_MAJOR_C_MINOR' if arousal > 0.6 else ('B_FLAT_MAJOR_G_MINOR' if arousal > 0.4 else 'A_FLAT_MAJOR_F_MINOR')
    
    def generate_prompts(self, params: dict) -> list:
        """
        Genera prompts variados basados en el estado cerebral.
        Usa instrumentos y estilos específicos de Lyria.
        """
        prompts = []
        
        # Usar valores ajustados si disponibles
        arousal = params.get('arousal_adjusted', params['arousal'])
        valence = params.get('valence_adjusted', params['valence'])
        relaxation = params.get('relaxation_adjusted', params['relaxation'])
        focus = params.get('focus_adjusted', params.get('focus', 0.5))
        
        theta = params.get('theta', 0.2)
        alpha = params.get('alpha', 0.2)
        beta = params.get('beta', 0.2)
        gamma = params.get('gamma', 0.2)
        
        # === ENERGÍA / RITMO ===
        if arousal > 0.75:
            prompts.append(("driving drums, energetic pulse, tight groove", 1.0))
        elif arousal > 0.55:
            prompts.append(("steady rhythm, flowing movement, moderate energy", 1.0))
        elif arousal > 0.35:
            prompts.append(("gentle pulse, soft dynamics, relaxed tempo", 1.0))
        else:
            prompts.append(("slow sustained drones, minimal rhythm, spacious", 1.0))
        
        # === INSTRUMENTOS por banda dominante ===
        dominant_band = max(['delta', 'theta', 'alpha', 'beta', 'gamma'], 
                           key=lambda b: params.get(b, 0))
        
        instrument_map = {
            'delta': ["deep bass, sub frequencies", "cello, contrabass"],
            'theta': ["synth pads, ethereal textures", "hang drum, kalimba"],
            'alpha': ["piano, acoustic guitar", "rhodes piano, soft strings"],
            'beta': ["electric guitar, synth bass", "marimba, vibraphone"],
            'gamma': ["bright bells, glockenspiel", "harpsichord, sparkling synths"]
        }
        instruments = instrument_map.get(dominant_band, ["piano"])
        prompts.append((instruments[int(arousal > 0.5)], 0.8))
        
        # === MOOD / ATMÓSFERA ===
        if valence > 0.65:
            prompts.append(("uplifting, bright tones, positive energy", valence))
        elif valence > 0.45:
            prompts.append(("peaceful, balanced, neutral mood", 0.5))
        else:
            prompts.append(("contemplative, introspective, deep", 0.7 - valence))
        
        # === TEXTURA por estado ===
        if relaxation > 0.6:
            prompts.append(("ambient pads, reverb, spacious atmosphere", relaxation))
        
        if focus > 0.6:
            prompts.append(("precise articulation, clear melodies, structured", focus))
        
        if theta > 0.3:
            prompts.append(("dreamy, floating, ethereal atmosphere", theta * 1.5))
        
        if gamma > 0.25:
            prompts.append(("shimmering highs, crisp details, bright overtones", gamma * 2))
        
        # === ESTILO MUSICAL (cambia con cambios significativos) ===
        if params.get('significant_change', False):
            self.current_style_idx = (self.current_style_idx + 1) % 6
        
        styles = [
            "ambient electronic",
            "neo classical",
            "lo-fi chill",
            "cinematic orchestral", 
            "jazz fusion",
            "minimal techno"
        ]
        prompts.append((styles[self.current_style_idx], 0.4))
        
        return prompts


async def run_live_session(duration: int = 60):
    """Sesión LIVE: genera y reproduce música en tiempo real"""
    
    try:
        from google import genai
        from google.genai import types
        from pylsl import StreamInlet, resolve_byprop
    except ImportError as e:
        logger.error(f"❌ Dependencia faltante: {e}")
        return
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY no encontrada")
        return
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      🧠🎵 EEG → LYRIA LIVE MUSIC 🎵🧠                   ║
    ║      Escucha tu cerebro en TIEMPO REAL                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Conectar MUSE
    print("🔍 Buscando MUSE...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        logger.error("❌ No se encontró MUSE. Ejecuta: muselsl stream")
        return
    
    inlet = StreamInlet(streams[0])
    print(f"✅ MUSE conectado: {streams[0].name()}")
    
    # Inicializar con mayor sensibilidad
    processor = EEGProcessor(sensitivity=2.5)  # Amplifica cambios x2.5
    player = AudioPlayer(sample_rate=SAMPLE_RATE_AUDIO, channels=CHANNELS)
    
    # Conectar Lyria
    print("🎵 Conectando a Lyria...")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    
    eeg_buffer = []
    
    try:
        async with client.aio.live.music.connect(model='models/lyria-realtime-exp') as session:
            print("✅ Lyria conectado!")
            
            # Config inicial
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text="gentle flowing music", weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=80, temperature=1.0)
            )
            
            # Iniciar playback
            await session.play()
            player.start()
            
            print("\n" + "=" * 60)
            print("🎧 ESCUCHANDO EN VIVO - Tu cerebro controla la música")
            print("=" * 60)
            print("   Ctrl+C para parar\n")
            
            start_time = time.time()
            last_update = 0
            update_interval = 1.5  # Actualizar más frecuentemente
            
            # Tarea para recibir y reproducir audio
            async def receive_and_play():
                async for message in session.receive():
                    if time.time() - start_time >= duration:
                        break
                    if hasattr(message, 'server_content') and message.server_content:
                        if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                            for chunk in message.server_content.audio_chunks:
                                player.add_audio(chunk.data)
            
            audio_task = asyncio.create_task(receive_and_play())
            
            # Loop principal: EEG → Lyria
            while time.time() - start_time < duration:
                sample, _ = inlet.pull_sample(timeout=0.1)
                if sample:
                    eeg_buffer.append(sample[:4])
                
                if len(eeg_buffer) >= WINDOW_SIZE:
                    eeg_data = np.array(eeg_buffer).T
                    bands = processor.extract_bands(eeg_data)
                    metrics = processor.calculate_metrics(bands)
                    params = processor.map_to_lyria(bands, metrics)
                    
                    elapsed = time.time() - start_time
                    if elapsed - last_update >= update_interval:
                        last_update = elapsed
                        
                        # Actualizar Lyria
                        prompts = processor.generate_prompts(params)
                        lyria_prompts = [types.WeightedPrompt(text=t, weight=w) for t, w in prompts]
                        await session.set_weighted_prompts(prompts=lyria_prompts)
                        
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
                        await session.set_music_generation_config(config=config)
                        
                        # Mostrar estado con cambios
                        change_indicator = "🔄" if params.get('significant_change') else "  "
                        adj_a = params.get('arousal_adjusted', params['arousal'])
                        adj_v = params.get('valence_adjusted', params['valence'])
                        dominant = max(['δ', 'θ', 'α', 'β', 'γ'], 
                                      key=lambda b: params.get({'δ':'delta','θ':'theta','α':'alpha','β':'beta','γ':'gamma'}[b], 0))
                        print(f"{change_indicator} {elapsed:.0f}s | 🧠 A:{adj_a:.2f} V:{adj_v:.2f} {dominant} | 🎵 {params['bpm']}BPM {params['scale'][:7]} | 🎨 {prompts[0][0][:25]}...")
                    
                    eeg_buffer = eeg_buffer[WINDOW_SIZE // 2:]
                
                await asyncio.sleep(0.01)
            
            await session.pause()
            audio_task.cancel()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Detenido por usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        player.stop()
    
    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    player.save(f"eeg_live_{timestamp}.wav")
    
    print("\n✅ Sesión completada!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EEG → Lyria LIVE")
    parser.add_argument('--duration', type=int, default=60, help='Duración (segundos)')
    args = parser.parse_args()
    
    asyncio.run(run_live_session(duration=args.duration))


if __name__ == "__main__":
    main()
