#!/usr/bin/env python3
"""Test if lyria-realtime-exp still works with new key"""
import os, asyncio
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

async def test():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    print("Testing lyria-realtime-exp with new key...")
    try:
        session = await client.aio.live.music.connect(model="models/lyria-realtime-exp").__aenter__()
        print("Connected!")
        await session.set_weighted_prompts(prompts=[types.WeightedPrompt(text="calm ambient", weight=1.0)])
        await session.set_music_generation_config(config=types.LiveMusicGenerationConfig(bpm=80, temperature=1.0))
        await session.play()
        print("Play sent, waiting for audio...")
        count = 0
        async for msg in session.receive():
            sc = getattr(msg, "server_content", None)
            if sc and getattr(sc, "audio_chunks", None):
                for ch in sc.audio_chunks:
                    count += 1
                    print(f"  Chunk {count}: {len(ch.data)} bytes")
            if count >= 3:
                break
        await session.__aexit__(None, None, None)
        print(f"SUCCESS: {count} chunks from realtime API")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

asyncio.run(test())
