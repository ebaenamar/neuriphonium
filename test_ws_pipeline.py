"""Test WebSocket pipeline: measure chunk arrival times through the full stack."""
import asyncio
import websockets
import json
import time
import struct

async def test_pipeline():
    uri = "ws://localhost:8767"
    
    print("Connecting to ws://localhost:8767...")
    async with websockets.connect(uri) as ws:
        print("Connected!\n")
        
        # Send start_music command
        await ws.send(json.dumps({
            'action': 'start_music',
            'engine': 'local',
            'model_size': 'mrt2_base_fast',
        }))
        print("Sent start_music command\n")
        
        chunk_times = []
        chunk_ids = []
        t_start = time.time()
        
        # Receive messages for 60 seconds
        while time.time() - t_start < 60:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                t_recv = time.time() - t_start
                
                if isinstance(msg, bytes):
                    # Binary audio chunk
                    chunk_id = struct.unpack('<I', msg[:4])[0]
                    chunk_times.append(t_recv)
                    chunk_ids.append(chunk_id)
                    
                    if chunk_id % 10 == 0 or chunk_id <= 5:
                        if len(chunk_times) >= 2:
                            interval = chunk_times[-1] - chunk_times[-2]
                            print(f"  chunk #{chunk_id:3d} at {t_recv:6.2f}s | interval: {interval*1000:5.0f}ms | size: {len(msg)} bytes")
                        else:
                            print(f"  chunk #{chunk_id:3d} at {t_recv:6.2f}s | size: {len(msg)} bytes")
                else:
                    data = json.loads(msg)
                    if data.get('type') == 'music_started':
                        print(f"  music_started at {t_recv:.2f}s")
                    elif data.get('type') == 'music_status':
                        pass  # ignore
                    elif data.get('type') == 'eeg':
                        pass  # ignore
                    elif data.get('type') not in ('eeg', 'music_status'):
                        print(f"  {data.get('type', '?')} at {t_recv:.2f}s")
            except asyncio.TimeoutError:
                print(f"  [timeout at {time.time()-t_start:.1f}s]")
                break
        
        # Send stop
        await ws.send(json.dumps({'action': 'stop_music'}))
        
        # Analysis
        if len(chunk_times) >= 2:
            intervals = [chunk_times[i+1] - chunk_times[i] for i in range(len(chunk_times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            jitter = max_interval - min_interval
            
            print(f"\n{'='*60}")
            print(f"WEBSOCKET PIPELINE RESULTS")
            print(f"  Chunks received: {len(chunk_times)}")
            print(f"  Total time: {chunk_times[-1]:.2f}s")
            print(f"  Audio time: {len(chunk_times) * 1.0:.2f}s")
            print(f"  Throughput: {len(chunk_times) / chunk_times[-1]:.2f} chunks/s")
            print(f"  Avg interval: {avg_interval*1000:.0f}ms (need <1000ms for realtime)")
            print(f"  Min interval: {min_interval*1000:.0f}ms")
            print(f"  Max interval: {max_interval*1000:.0f}ms")
            print(f"  Jitter: {jitter*1000:.0f}ms")
            
            slow = [iv for iv in intervals if iv > 1.0]
            if slow:
                print(f"\n  ⚠️ {len(slow)} intervals > 1000ms (would cause cuts!)")
                for i, iv in enumerate(intervals):
                    if iv > 1.0:
                        print(f"    chunk {chunk_ids[i+1]}: {iv*1000:.0f}ms")
            else:
                print(f"\n  ✅ All intervals < 1000ms")
            print(f"{'='*60}")

asyncio.run(test_pipeline())
