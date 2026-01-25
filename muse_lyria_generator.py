#!/usr/bin/env python3
"""
MUSE + Lyria RealTime Music Generator
Generates music in real-time based on MUSE EEG brainwave data
"""

import os
import asyncio
import wave
from typing import Dict, Optional
from dotenv import load_dotenv
import logging

from muse_adapter import MuseEEGAdapter, MuseLSLStream, MuseCSVStream

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output configuration
OUTPUT_DIR = "output"
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2


class MuseLyriaMusicGenerator:
    """
    Real-time music generation from MUSE EEG using Lyria RealTime.
    
    Flow:
    1. MUSE EEG data → Band powers → Cognitive metrics
    2. Cognitive metrics → Lyria prompts + config
    3. Lyria RealTime → Streaming audio
    """
    
    def __init__(self):
        self.adapter = MuseEEGAdapter()
        self.session = None
        self.client = None
        self.is_running = False
        self.audio_chunks = []
        self.current_params = None
        
        # Initialize Gemini client
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client with v1alpha API"""
        try:
            from google import genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env")
            
            self.client = genai.Client(
                api_key=api_key,
                http_options={'api_version': 'v1alpha'}
            )
            logger.info("✅ Gemini client initialized")
            
        except ImportError:
            logger.error("❌ google-genai not installed. Run: pip install google-genai")
            raise
    
    async def start_lyria_session(self):
        """Start Lyria RealTime session"""
        from google.genai import types
        
        logger.info("🎵 Connecting to Lyria RealTime...")
        
        self.session = await self.client.aio.live.music.connect(
            model='models/lyria-realtime-exp'
        ).__aenter__()
        
        # Set initial prompts
        await self.session.set_weighted_prompts(
            prompts=[
                types.WeightedPrompt(text="calm ambient meditation music", weight=1.0)
            ]
        )
        
        # Set initial config
        await self.session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=80,
                temperature=1.0
            )
        )
        
        logger.info("✅ Lyria session started")
    
    async def update_music_from_eeg(self, music_params: Dict[str, float]):
        """
        Update Lyria music generation based on EEG-derived parameters.
        
        Args:
            music_params: Dictionary from MuseEEGAdapter.map_to_music_params()
        """
        if not self.session:
            return
        
        from google.genai import types
        
        self.current_params = music_params
        
        # Generate prompt from brain state
        prompt = self.adapter.generate_prompt_from_state(music_params)
        
        # Build weighted prompts
        prompts = [
            types.WeightedPrompt(text=prompt, weight=1.0)
        ]
        
        # Add instrument based on relaxation
        if music_params['relaxation'] > 0.6:
            prompts.append(types.WeightedPrompt(text="soft pads, ambient textures", weight=0.7))
        elif music_params['arousal'] > 0.6:
            prompts.append(types.WeightedPrompt(text="driving drums, synth bass", weight=0.7))
        else:
            prompts.append(types.WeightedPrompt(text="piano, gentle strings", weight=0.5))
        
        # Update prompts
        await self.session.set_weighted_prompts(prompts=prompts)
        
        # Update config (BPM requires context reset for drastic changes)
        new_bpm = music_params['bpm']
        
        await self.session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=new_bpm,
                density=music_params['density'],
                brightness=music_params['brightness'],
                temperature=music_params['temperature']
            )
        )
        
        logger.info(f"🎵 Updated: {prompt[:50]}... | BPM={new_bpm}")
    
    async def receive_audio_loop(self):
        """Background task to receive and store audio chunks"""
        logger.info("🔊 Audio receiver started")
        
        while self.is_running:
            try:
                async for message in self.session.receive():
                    if not self.is_running:
                        break
                    
                    if hasattr(message, 'server_content') and message.server_content:
                        if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                            for chunk in message.server_content.audio_chunks:
                                self.audio_chunks.append(chunk.data)
                    
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                if self.is_running:
                    logger.error(f"❌ Audio receive error: {e}")
                break
        
        logger.info("🔊 Audio receiver stopped")
    
    def eeg_callback(self, music_params: Dict[str, float]):
        """Callback for EEG stream - schedules async update"""
        if self.is_running and self.session:
            asyncio.create_task(self.update_music_from_eeg(music_params))
    
    async def run_with_lsl(self, duration: int = 60):
        """
        Run music generation with real-time MUSE LSL stream.
        
        Args:
            duration: Duration in seconds
        """
        lsl_stream = MuseLSLStream(self.adapter)
        
        if not lsl_stream.connect():
            logger.error("❌ Could not connect to MUSE. Make sure muselsl is running.")
            return
        
        await self._run_generation(
            lambda: lsl_stream.start_streaming(self.eeg_callback),
            lambda: lsl_stream.stop_streaming(),
            duration
        )
    
    async def run_with_csv(self, csv_path: str, duration: int = 60, delay: float = 1.0):
        """
        Run music generation with CSV file playback.
        
        Args:
            csv_path: Path to MUSE CSV file
            duration: Max duration in seconds
            delay: Delay between EEG updates
        """
        csv_stream = MuseCSVStream(self.adapter)
        
        await self._run_generation(
            lambda: csv_stream.stream_from_csv(csv_path, self.eeg_callback, delay=delay),
            lambda: csv_stream.stop_streaming(),
            duration
        )
    
    async def _run_generation(self, start_eeg_fn, stop_eeg_fn, duration: int):
        """Internal method to run the generation loop"""
        import threading
        
        self.is_running = True
        self.audio_chunks = []
        
        try:
            # Start Lyria session
            await self.start_lyria_session()
            
            # Start playback
            await self.session.play()
            logger.info("▶️ Lyria playback started")
            
            # Start audio receiver
            audio_task = asyncio.create_task(self.receive_audio_loop())
            
            # Start EEG stream in background thread
            eeg_thread = threading.Thread(target=start_eeg_fn, daemon=True)
            eeg_thread.start()
            
            # Run for specified duration
            logger.info(f"🎵 Generating music for {duration} seconds...")
            await asyncio.sleep(duration)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Stopped by user")
        finally:
            self.is_running = False
            stop_eeg_fn()
            
            if self.session:
                await self.session.pause()
                await self.session.__aexit__(None, None, None)
            
            logger.info("✅ Generation complete")
    
    def save_audio(self, filename: str = "muse_music.wav"):
        """Save collected audio to WAV file"""
        if not self.audio_chunks:
            logger.warning("⚠️ No audio to save")
            return None
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        audio_data = b''.join(self.audio_chunks)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        
        duration = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        logger.info(f"💾 Saved {duration:.1f}s of audio to {filepath}")
        
        return filepath


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MUSE + Lyria Music Generator")
    parser.add_argument('--mode', choices=['lsl', 'csv'], default='lsl',
                       help='Input mode: lsl (real-time) or csv (file)')
    parser.add_argument('--csv', type=str, help='Path to MUSE CSV file')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds (default: 60)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between EEG updates for CSV mode (default: 1.0)')
    parser.add_argument('--output', type=str, default='muse_music.wav',
                       help='Output filename (default: muse_music.wav)')
    
    args = parser.parse_args()
    
    print("🧠🎵 MUSE + Lyria RealTime Music Generator")
    print("=" * 50)
    
    generator = MuseLyriaMusicGenerator()
    
    try:
        if args.mode == 'csv':
            if not args.csv:
                print("❌ CSV mode requires --csv argument")
                return
            await generator.run_with_csv(args.csv, args.duration, args.delay)
        else:
            print("📡 LSL mode: Make sure MUSE is streaming via muselsl")
            print("   Run: muselsl stream")
            await generator.run_with_lsl(args.duration)
        
        # Save audio
        generator.save_audio(args.output)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
