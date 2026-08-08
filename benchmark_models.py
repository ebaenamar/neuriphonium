"""Benchmark mrt2_small vs mrt2_base generation speed."""
import time
import sys
sys.path.insert(0, 'venv/lib/python3.12/site-packages')

import mlx.core as mx
from magenta_rt.mlx.system import MagentaRT2SystemMlxfn

def benchmark(model_name, frames=25, runs=5):
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name} ({frames} frames = {frames*0.04:.1f}s audio)")
    print(f"{'='*60}")
    
    t0 = time.time()
    mrt = MagentaRT2SystemMlxfn(size=model_name)
    load_time = time.time() - t0
    print(f"Load time: {load_time:.1f}s")
    
    # Embed a test style
    t0 = time.time()
    style = mrt.embed_style("cool jazz, laid back, mellow trumpet, instrumental")
    embed_time = time.time() - t0
    print(f"embed_style time: {embed_time:.2f}s")
    
    # Warmup
    print("Warmup generation...")
    wav, state = mrt.generate(style=style, frames=frames)
    print(f"Warmup done. Audio shape: {wav.samples.shape}")
    
    # Benchmark
    times = []
    for i in range(runs):
        t0 = time.time()
        wav, state = mrt.generate(style=style, frames=frames, state=state)
        elapsed = time.time() - t0
        audio_duration = frames * 0.04
        ratio = audio_duration / elapsed
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s gen / {audio_duration:.1f}s audio = {ratio:.2f}x realtime")
    
    avg = sum(times) / len(times)
    avg_ratio = (frames * 0.04) / avg
    print(f"\n  Average: {avg:.2f}s gen, {avg_ratio:.2f}x realtime")
    print(f"  Per-frame: {avg/frames*1000:.0f}ms/frame")
    
    # Also test with fewer frames
    small_frames = 10
    print(f"\n  Testing {small_frames} frames ({small_frames*0.04:.1f}s audio)...")
    state = None
    t0 = time.time()
    wav, state = mrt.generate(style=style, frames=small_frames, state=state)
    elapsed = time.time() - t0
    ratio = (small_frames * 0.04) / elapsed
    print(f"  {small_frames} frames: {elapsed:.2f}s gen = {ratio:.2f}x realtime ({elapsed/small_frames*1000:.0f}ms/frame)")
    
    return avg_ratio

print("🧪 Model Speed Benchmark\n")
small_ratio = benchmark("mrt2_small", frames=25, runs=5)
base_ratio = benchmark("mrt2_base", frames=25, runs=5)

# Benchmark the fast variant if it exists
import os
fast_path = os.path.expanduser("~/Documents/Magenta/magenta-rt-v2/models/mrt2_base_fast")
if os.path.isdir(fast_path):
    fast_ratio = benchmark("mrt2_base_fast", frames=25, runs=5)
else:
    fast_ratio = None

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  mrt2_small:      {small_ratio:.2f}x realtime")
print(f"  mrt2_base:       {base_ratio:.2f}x realtime")
if fast_ratio is not None:
    print(f"  mrt2_base_fast:  {fast_ratio:.2f}x realtime (4-bit + 1 CFG)")
    speedup = fast_ratio / base_ratio
    print(f"  Speedup:         {speedup:.1f}x over base")
else:
    print(f"  mrt2_base_fast:  (not exported yet)")
print(f"{'='*60}")
