"""Cold test: generate audio continuously, write to WAV, measure timing precisely."""
import time
import sys
import os
import wave
import numpy as np

sys.path.insert(0, 'venv/lib/python3.12/site-packages')
import mlx.core as mx

# Ensure max GPU memory
mx.metal.set_cache_limit(80 * 1024 * 1024 * 1024)

from magenta_rt.mlx.system import MagentaRT2SystemMlxfn

MODEL = os.environ.get('MODEL', 'mrt2_base_fast')
RUNS = int(os.environ.get('RUNS', '60'))
FRAMES = int(os.environ.get('FRAMES', '25'))

print(f"🧪 Cold pipeline test: {MODEL}, {RUNS} chunks, {FRAMES} frames/chunk")
print(f"   {FRAMES} frames = {FRAMES * 0.04:.1f}s audio per chunk\n")

# Load model
t0 = time.time()
mrt = MagentaRT2SystemMlxfn(size=MODEL)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Embed style
style = mrt.embed_style('cool jazz, laid back, mellow trumpet, instrumental')

# Warmup
print("Warming up...")
wav, state = mrt.generate(style=style, frames=FRAMES)
print("Warmup done\n")

# Generate continuously
all_audio = []
times = []
state = None

print(f"{'chunk':>5} | {'gen_time':>8} | {'ratio':>6} | {'cumulative':>10} | {'audio_time':>10} | {'drift':>8}")
print("-" * 75)

t_start = time.time()
for i in range(RUNS):
    t0 = time.time()
    wav, state = mrt.generate(
        style=style,
        frames=FRAMES,
        state=state,
    )
    elapsed = time.time() - t0
    times.append(elapsed)
    
    samples = wav.samples.astype(np.float32)
    all_audio.append(samples)
    
    audio_sec = (i + 1) * FRAMES * 0.04
    wall_sec = time.time() - t_start
    drift = wall_sec - audio_sec
    
    if i % 10 == 0 or i == RUNS - 1:
        ratio = (FRAMES * 0.04) / elapsed
        print(f"{i+1:5d} | {elapsed:7.3f}s | {ratio:5.2f}x | {wall_sec:9.2f}s | {audio_sec:9.2f}s | {drift:+7.2f}s")

total_wall = time.time() - t_start
total_audio = RUNS * FRAMES * 0.04
avg_gen = sum(times) / len(times)
min_gen = min(times)
max_gen = max(times)
p95_gen = sorted(times)[int(len(times) * 0.95)]

print(f"\n{'='*60}")
print(f"RESULTS: {MODEL}")
print(f"  Chunks: {RUNS}")
print(f"  Total wall time:  {total_wall:.2f}s")
print(f"  Total audio time: {total_audio:.2f}s")
print(f"  Overall ratio:    {total_audio/total_wall:.2f}x realtime")
print(f"  Avg gen time:     {avg_gen*1000:.0f}ms")
print(f"  Min gen time:     {min_gen*1000:.0f}ms")
print(f"  Max gen time:     {max_gen*1000:.0f}ms")
print(f"  P95 gen time:     {p95_gen*1000:.0f}ms")
print(f"  Jitter (max-min): {(max_gen-min_gen)*1000:.0f}ms")

# Check for any slow chunks that would cause cuts
slow_chunks = [t for t in times if t > FRAMES * 0.04]
if slow_chunks:
    print(f"\n  ⚠️ {len(slow_chunks)} chunks slower than realtime!")
    print(f"  Slowest: {max(slow_chunks)*1000:.0f}ms (need <{(FRAMES*0.04)*1000:.0f}ms)")
else:
    print(f"\n  ✅ All chunks faster than realtime")

# Save WAV
print(f"\nSaving to output/cold_test_{MODEL}.wav...")
os.makedirs('output', exist_ok=True)
all_samples = np.concatenate(all_audio, axis=0)
# Normalize
all_samples = all_samples / max(1.0, np.max(np.abs(all_samples)))
int16_samples = (all_samples * 32767).astype(np.int16)

with wave.open(f'output/cold_test_{MODEL}.wav', 'w') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(48000)
    wf.writeframes(int16_samples.tobytes())

print(f"Saved {len(all_samples)/48000:.1f}s of audio")
print(f"{'='*60}")
