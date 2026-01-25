#!/usr/bin/env python3
"""
EEG → Lyria RealTime Music Generator
Genera música en tiempo real controlada por tu cerebro usando MUSE + Lyria

Mappings abstractos (sin géneros fijos):
- BPM: controlado por arousal (60-200)
- Scale: controlado por valence + arousal (tonalidades)
- Density: controlado por relaxation (inverso)
- Brightness: controlado por gamma + valence
- Guidance: controlado por focus
- Temperature: controlado por theta
- Texture: mute_drums, mute_bass según estado
"""

import os
import asyncio
import wave
import time
import numpy as np
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuración
SAMPLE_RATE_AUDIO = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
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


class EEGProcessor:
    """Procesador de señales EEG simplificado"""
    
    def __init__(self):
        self.band_history = {band: deque(maxlen=5) for band in EEG_BANDS}
    
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
        
        return {
            'arousal': arousal,
            'valence': valence,
            'focus': focus,
            'relaxation': relaxation,
            **bands
        }
    
    def map_to_lyria_params(self, bands: dict, metrics: dict) -> dict:
        """
        Mapear EEG a parámetros de Lyria (abstractos, sin géneros)
        """
        arousal = metrics['arousal']
        valence = metrics['valence']
        focus = metrics['focus']
        relaxation = metrics['relaxation']
        
        delta = bands.get('delta', 0.2)
        theta = bands.get('theta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        # === PARÁMETROS LYRIA ===
        
        # BPM: 60-200
        bpm = int(60 + arousal * 140)
        
        # Density: 0-1 (relajado = sparse)
        density = 1.0 - relaxation * 0.7
        
        # Brightness: 0-1 (gamma + valence)
        brightness = np.clip(gamma * 1.5 + valence * 0.3, 0, 1)
        
        # Guidance: 0-6 (focus = más estructura)
        guidance = 2.0 + focus * 3.0
        
        # Temperature: 0-3 (theta = más experimental)
        temperature = 0.8 + theta * 1.5
        
        # Scale (tonalidad)
        scale = self._select_scale(valence, arousal)
        
        # Texture controls
        mute_drums = relaxation > 0.7
        mute_bass = delta > 0.4
        
        return {
            'bpm': max(60, min(200, bpm)),
            'density': max(0, min(1, density)),
            'brightness': max(0, min(1, brightness)),
            'guidance': max(0, min(6, guidance)),
            'temperature': max(0, min(3, temperature)),
            'scale': scale,
            'mute_drums': mute_drums,
            'mute_bass': mute_bass,
            **metrics
        }
    
    def _select_scale(self, valence: float, arousal: float) -> str:
        if valence > 0.6:
            if arousal > 0.6:
                return 'D_MAJOR_B_MINOR'
            elif arousal > 0.4:
                return 'G_MAJOR_E_MINOR'
            else:
                return 'C_MAJOR_A_MINOR'
        elif valence > 0.4:
            if arousal > 0.5:
                return 'A_MAJOR_G_FLAT_MINOR'
            else:
                return 'F_MAJOR_D_MINOR'
        else:
            if arousal > 0.6:
                return 'E_FLAT_MAJOR_C_MINOR'
            elif arousal > 0.4:
                return 'B_FLAT_MAJOR_G_MINOR'
            else:
                return 'A_FLAT_MAJOR_F_MINOR'
    
    def generate_prompts(self, params: dict) -> list:
        """Generar weighted prompts abstractos (sin géneros)"""
        prompts = []
        
        arousal = params['arousal']
        valence = params['valence']
        relaxation = params['relaxation']
        theta = params.get('theta', 0.2)
        alpha = params.get('alpha', 0.2)
        gamma = params.get('gamma', 0.2)
        
        # Prompt principal basado en energía
        if arousal > 0.7:
            prompts.append(("driving rhythmic pulse, full ensemble", 1.0))
        elif arousal > 0.5:
            prompts.append(("steady flowing movement, balanced textures", 1.0))
        elif arousal > 0.3:
            prompts.append(("gentle undulating waves, soft dynamics", 1.0))
        else:
            prompts.append(("slow sustained tones, sparse textures", 1.0))
        
        # Textura tonal
        if valence > 0.6 and gamma > 0.2:
            prompts.append(("bright shimmering harmonics", valence))
        elif valence < 0.4:
            prompts.append(("deep resonant frequencies", 0.6 - valence))
        
        # Carácter por banda
        if alpha > 0.3:
            prompts.append(("smooth flowing melodies", alpha))
        
        if theta > 0.25:
            prompts.append(("dreamy drifting atmosphere", theta * 1.5))
        
        if relaxation > 0.6:
            prompts.append(("spacious ambient pads", relaxation))
        
        return prompts


async def run_eeg_lyria_session(duration: int = 60):
    """
    Sesión de generación de música controlada por EEG
    
    Args:
        duration: Duración en segundos
    """
    try:
        from google import genai
        from google.genai import types
        from pylsl import StreamInlet, resolve_byprop
    except ImportError as e:
        logger.error(f"❌ Dependencia faltante: {e}")
        return
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY no encontrada en .env")
        return
    
    print("🧠🎵 EEG → Lyria RealTime Music Generator")
    print("=" * 60)
    
    # Conectar a MUSE
    print("\n🔍 Buscando stream MUSE...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        logger.error("❌ No se encontró stream MUSE. Ejecuta: muselsl stream")
        return
    
    inlet = StreamInlet(streams[0])
    print(f"✅ Conectado a MUSE: {streams[0].name()}")
    
    # Inicializar procesador EEG
    processor = EEGProcessor()
    
    # Inicializar cliente Lyria
    print("\n🎵 Conectando a Lyria RealTime...")
    client = genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1alpha'}
    )
    
    audio_chunks = []
    eeg_buffer = []
    current_params = None
    
    try:
        async with client.aio.live.music.connect(
            model='models/lyria-realtime-exp'
        ) as session:
            print("✅ Conectado a Lyria!")
            
            # Configuración inicial
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text="gentle flowing music", weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=80, temperature=1.0)
            )
            
            # Iniciar playback
            await session.play()
            print("▶️ Generando música...")
            print("\n" + "=" * 60)
            print("🎛️ CONTROLES EN TIEMPO REAL (desde tu cerebro)")
            print("=" * 60)
            
            start_time = time.time()
            last_update = 0
            update_interval = 2.0  # Actualizar cada 2 segundos
            
            # Tarea para recibir audio
            async def receive_audio():
                async for message in session.receive():
                    if not (time.time() - start_time < duration):
                        break
                    if hasattr(message, 'server_content') and message.server_content:
                        if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                            for chunk in message.server_content.audio_chunks:
                                audio_chunks.append(chunk.data)
            
            # Iniciar recepción de audio en background
            audio_task = asyncio.create_task(receive_audio())
            
            # Loop principal: leer EEG y actualizar Lyria
            while time.time() - start_time < duration:
                # Leer muestras EEG
                sample, _ = inlet.pull_sample(timeout=0.1)
                if sample:
                    eeg_buffer.append(sample[:4])
                
                # Procesar cuando tengamos suficientes muestras
                if len(eeg_buffer) >= WINDOW_SIZE:
                    eeg_data = np.array(eeg_buffer).T
                    
                    # Extraer bandas y métricas
                    bands = processor.extract_bands(eeg_data)
                    metrics = processor.calculate_metrics(bands)
                    params = processor.map_to_lyria_params(bands, metrics)
                    current_params = params
                    
                    # Actualizar Lyria periódicamente
                    elapsed = time.time() - start_time
                    if elapsed - last_update >= update_interval:
                        last_update = elapsed
                        
                        # Generar prompts
                        weighted_prompts = processor.generate_prompts(params)
                        
                        # Actualizar prompts en Lyria
                        lyria_prompts = [
                            types.WeightedPrompt(text=text, weight=weight)
                            for text, weight in weighted_prompts
                        ]
                        await session.set_weighted_prompts(prompts=lyria_prompts)
                        
                        # Actualizar configuración
                        # Mapear scale string a enum
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
                        
                        # Mostrar estado
                        print(f"\n⏱️  {elapsed:.0f}s / {duration}s")
                        print(f"🧠 Arousal: {params['arousal']:.2f} | Valence: {params['valence']:.2f}")
                        print(f"🎵 BPM: {params['bpm']} | Scale: {params['scale']}")
                        print(f"🎛️ Density: {params['density']:.2f} | Brightness: {params['brightness']:.2f}")
                        print(f"🔧 Guidance: {params['guidance']:.1f} | Temp: {params['temperature']:.2f}")
                        print(f"🥁 Mute drums: {params['mute_drums']} | Mute bass: {params['mute_bass']}")
                        print(f"📝 Prompt: {weighted_prompts[0][0][:50]}...")
                    
                    # Mantener overlap
                    eeg_buffer = eeg_buffer[WINDOW_SIZE // 2:]
                
                await asyncio.sleep(0.01)
            
            # Parar
            await session.pause()
            audio_task.cancel()
            print("\n\n⏸️ Sesión terminada")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Guardar audio
    if audio_chunks:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"eeg_music_{timestamp}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        audio_data = b''.join(audio_chunks)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE_AUDIO)
            wav_file.writeframes(audio_data)
        
        audio_duration = len(audio_data) / (SAMPLE_RATE_AUDIO * SAMPLE_WIDTH * CHANNELS)
        print(f"\n💾 Audio guardado: {filepath}")
        print(f"   Duración: {audio_duration:.1f}s")
        print(f"\n🎧 Reproducir con: afplay {filepath}")
        
        return filepath
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EEG → Lyria Music Generator")
    parser.add_argument('--duration', type=int, default=60,
                       help='Duración en segundos (default: 60)')
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🧠🎵 EEG → LYRIA REALTIME MUSIC GENERATOR 🎵🧠      ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Tu cerebro controla la música en tiempo real:           ║
    ║                                                          ║
    ║  • Arousal → Tempo (BPM)                                 ║
    ║  • Valence + Arousal → Tonalidad (Scale)                 ║
    ║  • Relaxation → Densidad (inverso)                       ║
    ║  • Gamma → Brillo                                        ║
    ║  • Focus → Guidance (estructura)                         ║
    ║  • Theta → Temperature (experimentación)                 ║
    ║  • Estados → Mute drums/bass                             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(run_eeg_lyria_session(duration=args.duration))


if __name__ == "__main__":
    main()
