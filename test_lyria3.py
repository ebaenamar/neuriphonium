#!/usr/bin/env python3
"""Test Lyria 3 Pro Preview - new non-realtime API"""
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Testing Lyria 3 Pro Preview...")
try:
    response = client.models.generate_content(
        model="models/lyria-3-pro-preview",
        contents="Generate 10 seconds of calm ambient meditation music with soft pads and gentle piano"
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            mime = part.inline_data.mime_type
            data = part.inline_data.data
            print(f"Got audio: {mime}, {len(data)} bytes")
            with open("test_lyria3_output.wav", "wb") as f:
                f.write(data)
            print("Saved to test_lyria3_output.wav")
        elif hasattr(part, "text") and part.text:
            print(f"Text: {part.text[:200]}")
        else:
            print(f"Part: {type(part)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
