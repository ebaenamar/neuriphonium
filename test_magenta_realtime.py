#!/usr/bin/env python3
"""
Test local Magenta RealTime 2 model (mrt2_small)
Generates music and saves it to a WAV file
"""

import os
import sys
import wave
import numpy as np

try:
    from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
    from magenta_rt.audio import Waveform
except ImportError as e:
    print(f"❌ Error importing magenta-rt: {e}")
    print("Run: venv/bin/pip install 'magenta-rt[mlx]'")
    sys.exit(1)

OUTPUT_FILE = "test_magenta_output.wav"
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2

def main():
    print("🎵 Initializing local Magenta RealTime 2 (mrt2_small)...")
    try:
        # Load small model using the compiled .mlxfn path
        mrt = MagentaRT2SystemMlxfn(size='mrt2_small')
        print("✅ Local Magenta RealTime 2 loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Magenta RT: {e}")
        import traceback
        traceback.print_exc()
        return

    # Text prompt for the style
    prompt = "smooth classical piano with warm ambient strings, peaceful, lo-fi"
    print(f"🎨 Encoding style prompt: '{prompt}'...")
    try:
        style_embedding = mrt.embed_style(prompt)
        print("✅ Style prompt encoded successfully!")
    except Exception as e:
        print(f"❌ Error embedding style: {e}")
        return

    print("🎹 Generating 5 seconds of audio locally...")
    # 25 frames ≈ 1 second at 48kHz
    total_frames = 25 * 5
    
    try:
        waveform, state = mrt.generate(
            style=style_embedding,
            frames=total_frames,
            temperature=1.2,
            top_k=40
        )
        print("✅ Audio generated successfully!")
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return

    # Convert generated float32 samples to int16 PCM bytes
    # waveform.samples has shape [total_samples, 2]
    print(f"🔊 Saving audio to {OUTPUT_FILE}...")
    try:
        # Multiply by 32767.0 to scale to int16
        samples_int16 = (waveform.samples * 32767.0).astype(np.int16)
        pcm_bytes = samples_int16.tobytes()
        
        with wave.open(OUTPUT_FILE, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)
            
        duration = len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        print(f"🎉 Completed! Saved {duration:.1f} seconds of audio.")
        print(f"🎧 You can play it with: afplay {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving WAV: {e}")

if __name__ == "__main__":
    main()
