#!/usr/bin/env python3
"""
Test MUSE + Lyria integration with synthetic EEG data
Generates music that changes based on simulated brain states
"""

import os
import asyncio
import wave
import numpy as np
from dotenv import load_dotenv

from muse_adapter import MuseEEGAdapter

load_dotenv()

# Output configuration
OUTPUT_FILE = "test_muse_lyria_output.wav"
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2


def generate_synthetic_eeg(state: str, duration_samples: int = 256) -> np.ndarray:
    """
    Generate synthetic EEG data for testing.
    
    Args:
        state: 'relaxed', 'focused', 'meditative', 'alert'
        duration_samples: Number of samples
    
    Returns:
        numpy array of shape (4, duration_samples)
    """
    t = np.linspace(0, 1, duration_samples)
    signal = np.zeros((4, duration_samples))
    
    # State-specific frequency boosts
    boosts = {
        'relaxed': {'alpha': 3.0, 'beta': 0.3, 'theta': 0.5},
        'focused': {'alpha': 0.5, 'beta': 2.5, 'theta': 0.3},
        'meditative': {'alpha': 1.5, 'beta': 0.2, 'theta': 3.0},
        'alert': {'alpha': 0.3, 'beta': 2.0, 'gamma': 2.0},
    }
    
    boost = boosts.get(state, boosts['relaxed'])
    
    for ch in range(4):
        # Base frequencies for each band
        signal[ch] += 0.3 * np.sin(2 * np.pi * 2 * t)  # Delta (2 Hz)
        signal[ch] += 0.5 * np.sin(2 * np.pi * 6 * t) * boost.get('theta', 1.0)  # Theta
        signal[ch] += 1.0 * np.sin(2 * np.pi * 10 * t) * boost.get('alpha', 1.0)  # Alpha
        signal[ch] += 0.5 * np.sin(2 * np.pi * 20 * t) * boost.get('beta', 1.0)  # Beta
        signal[ch] += 0.2 * np.sin(2 * np.pi * 40 * t) * boost.get('gamma', 1.0)  # Gamma
        signal[ch] += np.random.randn(duration_samples) * 0.05  # Noise
    
    return signal


async def test_muse_lyria_integration():
    """Test the full MUSE → Lyria pipeline with synthetic data"""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("❌ google-genai not installed")
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    print("🧠🎵 MUSE + Lyria Integration Test")
    print("=" * 50)
    
    # Initialize
    adapter = MuseEEGAdapter()
    client = genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1alpha'}
    )
    
    audio_chunks = []
    
    # Brain states to simulate (each for ~5 seconds)
    states = [
        ('relaxed', 5),
        ('focused', 5),
        ('meditative', 5),
        ('alert', 5),
    ]
    
    try:
        print("\n🎵 Connecting to Lyria RealTime...")
        
        async with client.aio.live.music.connect(
            model='models/lyria-realtime-exp'
        ) as session:
            print("✅ Connected!")
            
            # Initial config
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text="calm ambient music", weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=80, temperature=1.0)
            )
            
            # Start playback
            await session.play()
            print("▶️ Playback started\n")
            
            # Process each brain state
            for state_name, duration in states:
                print(f"🧠 Simulating state: {state_name.upper()} ({duration}s)")
                
                # Generate synthetic EEG
                eeg_data = generate_synthetic_eeg(state_name)
                
                # Extract band powers and map to music
                band_powers = adapter.extract_band_powers(eeg_data)
                music_params = adapter.map_to_music_params(band_powers)
                prompt = adapter.generate_prompt_from_state(music_params)
                
                print(f"   Arousal={music_params['arousal']:.2f}, Valence={music_params['valence']:.2f}")
                print(f"   BPM={music_params['bpm']}, Density={music_params['density']:.2f}")
                print(f"   🎵 {prompt[:60]}...")
                
                # Update Lyria
                await session.set_weighted_prompts(
                    prompts=[
                        types.WeightedPrompt(text=prompt, weight=1.0)
                    ]
                )
                
                await session.set_music_generation_config(
                    config=types.LiveMusicGenerationConfig(
                        bpm=music_params['bpm'],
                        density=music_params['density'],
                        brightness=music_params['brightness'],
                        temperature=1.0
                    )
                )
                
                # Collect audio for this state
                start_time = asyncio.get_event_loop().time()
                
                async for message in session.receive():
                    elapsed = asyncio.get_event_loop().time() - start_time
                    
                    if hasattr(message, 'server_content') and message.server_content:
                        if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                            for chunk in message.server_content.audio_chunks:
                                audio_chunks.append(chunk.data)
                    
                    if elapsed >= duration:
                        break
                
                print(f"   ✅ Collected {len(audio_chunks)} chunks\n")
            
            # Stop
            await session.pause()
            print("⏸️ Playback stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Save audio
    if audio_chunks:
        print(f"\n💾 Saving audio to {OUTPUT_FILE}...")
        
        audio_data = b''.join(audio_chunks)
        
        with wave.open(OUTPUT_FILE, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        
        duration = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        print(f"✅ Saved {duration:.1f}s of audio")
        print(f"\n🎧 Play with: afplay {OUTPUT_FILE}")
        return True
    
    return False


if __name__ == "__main__":
    asyncio.run(test_muse_lyria_integration())
