#!/usr/bin/env python3
"""Quick test: can we connect to Lyria RealTime and get audio?"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API key: {api_key[:10]}...{api_key[-4:]}")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    print("Client created OK")

    print("Connecting to Lyria RealTime...")
    try:
        session = await client.aio.live.music.connect(
            model="models/lyria-realtime-exp"
        ).__aenter__()
        print("Connected!")

        await session.set_weighted_prompts(
            prompts=[types.WeightedPrompt(text="calm ambient music", weight=1.0)]
        )
        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=80,
                temperature=1.0,
            )
        )
        print("Config set, calling play()...")
        await session.play()
        print("Playback started, waiting for audio...")

        chunk_count = 0
        async for message in session.receive():
            sc = getattr(message, "server_content", None)
            if sc:
                ac = getattr(sc, "audio_chunks", None)
                if ac:
                    for chunk in ac:
                        chunk_count += 1
                        print(f"  Chunk {chunk_count}: {len(chunk.data)} bytes")
            if chunk_count >= 5:
                break

        await session.pause()
        await session.__aexit__(None, None, None)
        print(f"SUCCESS: got {chunk_count} audio chunks from Lyria")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
