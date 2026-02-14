#!/usr/bin/env python3
"""
Microphone Diagnostic Tool
Tests audio input availability and captures a short recording
"""

import sounddevice as sd
import numpy as np
import time

print("=" * 60)
print("🎤 MICROPHONE DIAGNOSTIC TOOL")
print("=" * 60)

# 1. List all audio devices
print("\n1️⃣ Available Audio Devices:")
print("-" * 60)
try:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_marker = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  {i}: {device['name']}{default_marker}")
            print(f"     Channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            print(f"     Sample rate: {device['default_samplerate']} Hz")
except Exception as e:
    print(f"  ❌ Error listing devices: {e}")
    exit(1)

# 2. Get default input device
print("\n2️⃣ Default Input Device:")
print("-" * 60)
try:
    default_device = sd.default.device[0]
    default_info = sd.query_devices(default_device, 'input')
    print(f"  Device: {default_info['name']}")
    print(f"  Channels: {default_info['max_input_channels']}")
    print(f"  Sample rate: {default_info['default_samplerate']} Hz")
except Exception as e:
    print(f"  ❌ Error getting default device: {e}")
    exit(1)

# 3. Test microphone capture
print("\n3️⃣ Testing Microphone Capture (3 seconds)...")
print("-" * 60)
print("  🎤 Recording... (make some noise!)")

try:
    duration = 3  # seconds
    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    
    # Analyze recording
    rms = np.sqrt(np.mean(recording**2))
    peak = np.max(np.abs(recording))
    
    print(f"  ✅ Recording successful!")
    print(f"  RMS level: {rms:.6f}")
    print(f"  Peak level: {peak:.6f}")
    
    if peak < 0.001:
        print(f"  ⚠️  WARNING: Very low signal detected!")
        print(f"     - Check if microphone is muted")
        print(f"     - Check microphone input level in System Preferences")
        print(f"     - Try speaking/playing louder")
    elif peak < 0.01:
        print(f"  ⚠️  Signal is weak but present")
        print(f"     - Consider increasing input volume")
    else:
        print(f"  ✅ Good signal level detected!")
    
except Exception as e:
    print(f"  ❌ Error recording: {e}")
    print(f"\n  Possible issues:")
    print(f"  - Microphone permission not granted")
    print(f"  - No microphone connected")
    print(f"  - Microphone in use by another application")
    exit(1)

# 4. Test pitch detection with aubio
print("\n4️⃣ Testing Pitch Detection...")
print("-" * 60)
try:
    import aubio
    
    # Create pitch detector
    hop_size = 512
    buffer_size = 2048
    pitch_detector = aubio.pitch("yinfft", buffer_size, hop_size, sample_rate)
    pitch_detector.set_unit("midi")
    pitch_detector.set_silence(-40)
    
    print("  ✅ Aubio pitch detector initialized")
    
    # Test on recording
    pitches_detected = []
    for i in range(0, len(recording) - hop_size, hop_size):
        frame = recording[i:i+hop_size].flatten()
        pitch_midi = pitch_detector(frame)[0]
        confidence = pitch_detector.get_confidence()
        
        if confidence > 0.7 and pitch_midi > 0:
            pitches_detected.append(pitch_midi)
    
    if len(pitches_detected) > 0:
        print(f"  ✅ Detected {len(pitches_detected)} pitch samples")
        
        # Convert to note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_classes = [int(round(p)) % 12 for p in pitches_detected]
        from collections import Counter
        note_counts = Counter(pitch_classes)
        
        print(f"  Top notes detected:")
        for note_idx, count in note_counts.most_common(5):
            pct = count / len(pitches_detected) * 100
            print(f"    {note_names[note_idx]}: {pct:.1f}%")
    else:
        print(f"  ⚠️  No clear pitches detected")
        print(f"     - Try playing/singing louder")
        print(f"     - Reduce background noise")
        print(f"     - Get closer to the microphone")
    
except ImportError:
    print(f"  ⚠️  Aubio not installed - skipping pitch detection test")
except Exception as e:
    print(f"  ❌ Error in pitch detection: {e}")

# 5. Summary
print("\n" + "=" * 60)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 60)

if peak > 0.01:
    print("✅ Microphone is working and capturing audio")
    if len(pitches_detected) > 0:
        print("✅ Pitch detection is working")
        print("\n🎻 Your system is ready for violin input!")
    else:
        print("⚠️  Pitch detection needs improvement")
        print("   Try playing sustained notes louder and closer to mic")
else:
    print("❌ Microphone signal is too weak")
    print("\n🔧 Troubleshooting steps:")
    print("   1. Open System Preferences → Sound → Input")
    print("   2. Select your microphone")
    print("   3. Increase input volume")
    print("   4. Test by speaking - you should see the input level meter move")
    print("   5. Grant microphone permission to Terminal/Python if prompted")

print("\n" + "=" * 60)
