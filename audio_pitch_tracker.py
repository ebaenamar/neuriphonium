#!/usr/bin/env python3
"""
Audio Pitch Tracker with Key Detection
Analyzes violin/instrument melody to detect musical key
"""

import aubio
import sounddevice as sd
import numpy as np
from collections import deque, Counter
import time

# Note names for display
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Key profiles (Krumhansl-Schmuckler weights for major/minor)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Map detected key to Lyria scale enum
KEY_TO_LYRIA = {
    ('C', 'major'): 'C_MAJOR_A_MINOR',
    ('C#', 'major'): 'D_FLAT_MAJOR_B_FLAT_MINOR',
    ('D', 'major'): 'D_MAJOR_B_MINOR',
    ('D#', 'major'): 'E_FLAT_MAJOR_C_MINOR',
    ('E', 'major'): 'E_MAJOR_D_FLAT_MINOR',
    ('F', 'major'): 'F_MAJOR_D_MINOR',
    ('F#', 'major'): 'G_FLAT_MAJOR_E_FLAT_MINOR',
    ('G', 'major'): 'G_MAJOR_E_MINOR',
    ('G#', 'major'): 'A_FLAT_MAJOR_F_MINOR',
    ('A', 'major'): 'A_MAJOR_G_FLAT_MINOR',
    ('A#', 'major'): 'B_FLAT_MAJOR_G_MINOR',
    ('B', 'major'): 'B_MAJOR_A_FLAT_MINOR',
    ('C', 'minor'): 'C_MAJOR_A_MINOR',
    ('C#', 'minor'): 'D_FLAT_MAJOR_B_FLAT_MINOR',
    ('D', 'minor'): 'F_MAJOR_D_MINOR',
    ('D#', 'minor'): 'E_FLAT_MAJOR_C_MINOR',
    ('E', 'minor'): 'G_MAJOR_E_MINOR',
    ('F', 'minor'): 'A_FLAT_MAJOR_F_MINOR',
    ('F#', 'minor'): 'G_FLAT_MAJOR_E_FLAT_MINOR',
    ('G', 'minor'): 'B_FLAT_MAJOR_G_MINOR',
    ('G#', 'minor'): 'B_MAJOR_A_FLAT_MINOR',
    ('A', 'minor'): 'C_MAJOR_A_MINOR',
    ('A#', 'minor'): 'D_FLAT_MAJOR_B_FLAT_MINOR',
    ('B', 'minor'): 'D_MAJOR_B_MINOR',
}


class AudioPitchTracker:
    """Real-time pitch tracking and key detection from audio input"""
    
    def __init__(self, 
                 sample_rate=44100, 
                 hop_size=512, 
                 buffer_size=2048,
                 pitch_history_seconds=8.0,
                 key_update_interval=4.0,
                 min_confidence=0.3,
                 min_notes_required=20):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            hop_size: Hop size for pitch detection
            buffer_size: Buffer size for pitch detection
            pitch_history_seconds: Window size for key detection (seconds) - LARGER = more stable, less responsive
            key_update_interval: How often to update key detection (seconds) - SMALLER = more responsive
            min_confidence: Minimum confidence threshold (0-1) - HIGHER = more conservative
            min_notes_required: Minimum number of pitches needed - HIGHER = more data required
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        
        # Sensitivity parameters
        self.pitch_history_seconds = pitch_history_seconds
        self.key_update_interval = key_update_interval
        self.min_confidence = min_confidence
        self.min_notes_required = min_notes_required
        
        # Aubio pitch detector
        self.pitch_detector = aubio.pitch("yinfft", buffer_size, hop_size, sample_rate)
        self.pitch_detector.set_unit("midi")
        self.pitch_detector.set_silence(-40)  # dB threshold
        
        # Pitch history (dynamic size based on pitch_history_seconds)
        max_history = int(pitch_history_seconds * sample_rate / hop_size)
        self.pitch_history = deque(maxlen=max_history)
        
        # Key detection state
        self.current_key = None
        self.current_mode = None
        self.key_confidence = 0.0
        self.last_key_update = 0
        
        # Audio stream
        self.stream = None
        self.is_running = False
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Detect pitch
        pitch_midi = self.pitch_detector(indata[:, 0])[0]
        confidence = self.pitch_detector.get_confidence()
        
        # Store if confident and valid
        if confidence > 0.7 and pitch_midi > 0:
            self.pitch_history.append(pitch_midi)
    
    def detect_key(self):
        """
        Detect musical key from pitch history using Krumhansl-Schmuckler algorithm
        Returns: (key_name, mode, confidence, lyria_scale)
        """
        if len(self.pitch_history) < self.min_notes_required:
            return None, None, 0.0, None
        
        # Convert MIDI pitches to pitch classes (0-11)
        pitch_classes = [int(round(p)) % 12 for p in self.pitch_history]
        
        # Count occurrences of each pitch class
        pitch_histogram = [0] * 12
        for pc in pitch_classes:
            pitch_histogram[pc] += 1
        
        # Normalize histogram
        total = sum(pitch_histogram)
        if total == 0:
            return None, None, 0.0, None
        pitch_histogram = [count / total for count in pitch_histogram]
        
        # Correlate with major and minor profiles for all 12 keys
        best_correlation = -1
        best_key = None
        best_mode = None
        
        for tonic in range(12):
            # Test major
            major_corr = self._correlate(pitch_histogram, MAJOR_PROFILE, tonic)
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = tonic
                best_mode = 'major'
            
            # Test minor
            minor_corr = self._correlate(pitch_histogram, MINOR_PROFILE, tonic)
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = tonic
                best_mode = 'minor'
        
        # Convert to note name
        key_name = NOTE_NAMES[best_key]
        confidence = best_correlation
        
        # Get Lyria scale
        lyria_scale = KEY_TO_LYRIA.get((key_name, best_mode), 'C_MAJOR_A_MINOR')
        
        return key_name, best_mode, confidence, lyria_scale
    
    def _correlate(self, histogram, profile, tonic):
        """Compute correlation between pitch histogram and key profile"""
        # Rotate profile to match tonic
        rotated_profile = profile[tonic:] + profile[:tonic]
        
        # Pearson correlation
        hist_mean = np.mean(histogram)
        prof_mean = np.mean(rotated_profile)
        
        numerator = sum((h - hist_mean) * (p - prof_mean) 
                       for h, p in zip(histogram, rotated_profile))
        
        hist_std = np.sqrt(sum((h - hist_mean)**2 for h in histogram))
        prof_std = np.sqrt(sum((p - prof_mean)**2 for p in rotated_profile))
        
        if hist_std == 0 or prof_std == 0:
            return 0
        
        return numerator / (hist_std * prof_std)
    
    def update_key(self):
        """Update detected key if enough time has passed"""
        now = time.time()
        if now - self.last_key_update < self.key_update_interval:
            return False
        
        key, mode, confidence, lyria_scale = self.detect_key()
        
        if key and confidence > self.min_confidence:
            self.current_key = key
            self.current_mode = mode
            self.key_confidence = confidence
            self.last_key_update = now
            return True
        
        return False
    
    def get_current_key(self):
        """Get current detected key"""
        return {
            'key': self.current_key,
            'mode': self.current_mode,
            'confidence': self.key_confidence,
            'lyria_scale': KEY_TO_LYRIA.get((self.current_key, self.current_mode)) if self.current_key else None
        }
    
    def start(self, device=None):
        """Start audio input stream"""
        if self.is_running:
            return
        
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_size,
            callback=self.audio_callback
        )
        self.stream.start()
        self.is_running = True
        print(f"🎤 Audio input started (device: {device or 'default'})")
    
    def stop(self):
        """Stop audio input stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("🎤 Audio input stopped")
    
    def update_sensitivity(self, pitch_history_seconds=None, key_update_interval=None, 
                          min_confidence=None, min_notes_required=None):
        """Update sensitivity parameters on the fly"""
        if pitch_history_seconds is not None:
            self.pitch_history_seconds = pitch_history_seconds
            max_history = int(pitch_history_seconds * self.sample_rate / self.hop_size)
            # Create new deque with updated size, preserving recent data
            old_data = list(self.pitch_history)
            self.pitch_history = deque(old_data[-max_history:], maxlen=max_history)
        
        if key_update_interval is not None:
            self.key_update_interval = key_update_interval
        
        if min_confidence is not None:
            self.min_confidence = min_confidence
        
        if min_notes_required is not None:
            self.min_notes_required = min_notes_required
    
    def get_pitch_histogram(self):
        """Get current pitch class histogram for visualization"""
        if len(self.pitch_history) < 5:
            return [0] * 12
        
        pitch_classes = [int(round(p)) % 12 for p in self.pitch_history]
        counts = Counter(pitch_classes)
        histogram = [counts.get(i, 0) for i in range(12)]
        
        # Normalize
        total = sum(histogram)
        if total > 0:
            histogram = [count / total for count in histogram]
        
        return histogram


def test_pitch_tracker():
    """Test the pitch tracker standalone"""
    print("🎻 Audio Pitch Tracker Test")
    print("=" * 50)
    
    # Test with different sensitivity presets
    presets = {
        '1': {'name': 'Fast/Responsive', 'window': 4.0, 'interval': 2.0, 'conf': 0.25, 'notes': 15},
        '2': {'name': 'Balanced (default)', 'window': 8.0, 'interval': 4.0, 'conf': 0.3, 'notes': 20},
        '3': {'name': 'Stable/Conservative', 'window': 12.0, 'interval': 6.0, 'conf': 0.4, 'notes': 30},
    }
    
    print("\nSensitivity Presets:")
    for key, preset in presets.items():
        print(f"  {key}. {preset['name']}")
        print(f"     Window: {preset['window']}s, Update: {preset['interval']}s, "
              f"Confidence: {preset['conf']}, Min notes: {preset['notes']}")
    
    choice = input("\nSelect preset (1-3) or press Enter for default: ").strip() or '2'
    preset = presets.get(choice, presets['2'])
    
    print(f"\n✓ Using: {preset['name']}")
    print("Play your violin/instrument...")
    print("Press Ctrl+C to stop\n")
    
    tracker = AudioPitchTracker(
        pitch_history_seconds=preset['window'],
        key_update_interval=preset['interval'],
        min_confidence=preset['conf'],
        min_notes_required=preset['notes']
    )
    tracker.start()
    
    try:
        while True:
            time.sleep(1)
            
            # Update key detection
            if tracker.update_key():
                key_info = tracker.get_current_key()
                print(f"🎵 Detected: {key_info['key']} {key_info['mode']} "
                      f"(confidence: {key_info['confidence']:.2f})")
                print(f"   → Lyria scale: {key_info['lyria_scale']}")
            
            # Show pitch histogram and buffer status
            histogram = tracker.get_pitch_histogram()
            if sum(histogram) > 0:
                buffer_pct = len(tracker.pitch_history) / tracker.pitch_history.maxlen * 100
                print(f"   Buffer: {len(tracker.pitch_history)}/{tracker.pitch_history.maxlen} ({buffer_pct:.0f}%)", end=" | ")
                print("Pitches:", end=" ")
                for i, count in enumerate(histogram):
                    if count > 0.1:
                        print(f"{NOTE_NAMES[i]}:{count:.2f}", end=" ")
                print()
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
        tracker.stop()


if __name__ == "__main__":
    test_pitch_tracker()
