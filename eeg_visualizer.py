#!/usr/bin/env python3
"""
EEG Visualizer - Visualización web en tiempo real de estados cerebrales MUSE
Dashboard interactivo para explorar y grabar estados
"""

import os
import json
import asyncio
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
SAMPLE_RATE = 256
WINDOW_SIZE = 256
DATA_DIR = "eeg_recordings"

EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


class EEGProcessor:
    """Procesador de señales EEG"""
    
    def __init__(self):
        self.band_history = {band: deque(maxlen=10) for band in EEG_BANDS}
        self.raw_history = deque(maxlen=500)  # ~2 segundos de datos raw
        
    def extract_bands(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Extraer potencias de bandas"""
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        band_powers = {}
        n_samples = eeg_data.shape[1]
        
        fft_vals = np.fft.fft(eeg_data, axis=1)
        freqs = np.fft.fftfreq(n_samples, 1.0 / SAMPLE_RATE)
        
        for band_name, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            power = np.mean(np.abs(fft_vals[:, mask]) ** 2)
            band_powers[band_name] = power
        
        total = sum(band_powers.values())
        if total > 0:
            band_powers = {k: v / total for k, v in band_powers.items()}
        
        for band, power in band_powers.items():
            self.band_history[band].append(power)
            band_powers[band] = np.mean(list(self.band_history[band]))
        
        return band_powers
    
    def calculate_metrics(self, bands: Dict[str, float]) -> Dict[str, float]:
        """Calcular métricas cognitivas"""
        d, t, a, b, g = [bands.get(x, 0.2) for x in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
        
        return {
            'arousal': np.clip(0.1*d + 0.2*t + 0.3*a + 0.5*b + 0.7*g, 0, 1),
            'valence': np.clip((a - 0.3*t - 0.2*abs(b-t) + 0.5), 0, 1),
            'focus': np.clip(b / (t + 0.01) / 3, 0, 1),
            'relaxation': np.clip(a / (b + 0.01) / 2, 0, 1),
            'meditation': np.clip((t + a) / (b + g + 0.01) / 2, 0, 1),
            'engagement': np.clip((b + g) / (a + t + 0.01) / 2, 0, 1),
        }


# HTML del dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 EEG Explorer - MUSE</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 20px; font-size: 2em; }
        h1 span { font-size: 0.5em; opacity: 0.7; display: block; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            animation: pulse 2s infinite;
        }
        .status.connected { background: #44ff44; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .band-bar {
            display: flex;
            align-items: center;
            margin: 10px 0;
            gap: 10px;
        }
        .band-label {
            width: 80px;
            font-size: 0.9em;
        }
        .band-track {
            flex: 1;
            height: 24px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            overflow: hidden;
        }
        .band-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        .band-value {
            width: 50px;
            text-align: right;
            font-family: monospace;
        }
        
        .delta .band-fill { background: linear-gradient(90deg, #9b59b6, #8e44ad); }
        .theta .band-fill { background: linear-gradient(90deg, #3498db, #2980b9); }
        .alpha .band-fill { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .beta .band-fill { background: linear-gradient(90deg, #f1c40f, #f39c12); }
        .gamma .band-fill { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 5px 0;
        }
        .metric-label { opacity: 0.7; font-size: 0.9em; }
        
        .arousal .metric-value { color: #e74c3c; }
        .valence .metric-value { color: #2ecc71; }
        .focus .metric-value { color: #f1c40f; }
        .relaxation .metric-value { color: #3498db; }
        .meditation .metric-value { color: #9b59b6; }
        .engagement .metric-value { color: #e67e22; }
        
        .state-indicator {
            text-align: center;
            padding: 30px;
            font-size: 1.5em;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            margin-top: 15px;
        }
        .state-emoji { font-size: 3em; display: block; margin-bottom: 10px; }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: transform 0.2s, opacity 0.2s;
        }
        button:hover { transform: scale(1.05); }
        button:active { transform: scale(0.95); }
        
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #2ecc71; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-secondary { background: rgba(255,255,255,0.2); color: white; }
        
        input[type="text"] {
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 1em;
            flex: 1;
        }
        input::placeholder { color: rgba(255,255,255,0.5); }
        
        .recordings {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .recording-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            margin-bottom: 5px;
        }
        
        .chart-container {
            height: 200px;
            margin-top: 15px;
        }
        
        .suggestions {
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid rgba(46, 204, 113, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
        }
        .suggestions h3 { color: #2ecc71; margin-bottom: 10px; }
        .suggestion-item {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .suggestion-item:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 EEG Explorer <span>Exploración de Estados Cerebrales con MUSE</span></h1>
        
        <div class="grid">
            <!-- Bandas de Frecuencia -->
            <div class="card">
                <h2><span class="status" id="status"></span> Bandas de Frecuencia</h2>
                
                <div class="band-bar delta">
                    <span class="band-label">δ Delta</span>
                    <div class="band-track"><div class="band-fill" id="delta-bar"></div></div>
                    <span class="band-value" id="delta-val">0.00</span>
                </div>
                <div class="band-bar theta">
                    <span class="band-label">θ Theta</span>
                    <div class="band-track"><div class="band-fill" id="theta-bar"></div></div>
                    <span class="band-value" id="theta-val">0.00</span>
                </div>
                <div class="band-bar alpha">
                    <span class="band-label">α Alpha</span>
                    <div class="band-track"><div class="band-fill" id="alpha-bar"></div></div>
                    <span class="band-value" id="alpha-val">0.00</span>
                </div>
                <div class="band-bar beta">
                    <span class="band-label">β Beta</span>
                    <div class="band-track"><div class="band-fill" id="beta-bar"></div></div>
                    <span class="band-value" id="beta-val">0.00</span>
                </div>
                <div class="band-bar gamma">
                    <span class="band-label">γ Gamma</span>
                    <div class="band-track"><div class="band-fill" id="gamma-bar"></div></div>
                    <span class="band-value" id="gamma-val">0.00</span>
                </div>
                
                <div class="chart-container">
                    <canvas id="bandChart"></canvas>
                </div>
            </div>
            
            <!-- Métricas Cognitivas -->
            <div class="card">
                <h2>🎯 Métricas Cognitivas</h2>
                
                <div class="metric-grid">
                    <div class="metric arousal">
                        <div class="metric-label">Arousal</div>
                        <div class="metric-value" id="arousal-val">0.00</div>
                    </div>
                    <div class="metric valence">
                        <div class="metric-label">Valence</div>
                        <div class="metric-value" id="valence-val">0.00</div>
                    </div>
                    <div class="metric focus">
                        <div class="metric-label">Focus</div>
                        <div class="metric-value" id="focus-val">0.00</div>
                    </div>
                    <div class="metric relaxation">
                        <div class="metric-label">Relaxation</div>
                        <div class="metric-value" id="relaxation-val">0.00</div>
                    </div>
                    <div class="metric meditation">
                        <div class="metric-label">Meditation</div>
                        <div class="metric-value" id="meditation-val">0.00</div>
                    </div>
                    <div class="metric engagement">
                        <div class="metric-label">Engagement</div>
                        <div class="metric-value" id="engagement-val">0.00</div>
                    </div>
                </div>
                
                <div class="state-indicator">
                    <span class="state-emoji" id="state-emoji">🧘</span>
                    <span id="state-text">Conectando...</span>
                </div>
            </div>
            
            <!-- Grabación -->
            <div class="card">
                <h2>🎬 Grabación de Estados</h2>
                
                <div class="controls">
                    <input type="text" id="label-input" placeholder="Etiqueta (ej: relajado, pensando_en_playa)">
                    <button class="btn-success" onclick="startRecording()">▶️ Grabar</button>
                    <button class="btn-danger" onclick="stopRecording()">⏹️ Parar</button>
                </div>
                
                <div id="recording-status" style="margin-top: 15px; text-align: center;"></div>
                
                <h3 style="margin-top: 20px;">📁 Sesiones Grabadas</h3>
                <div class="recordings" id="recordings-list"></div>
                
                <div class="controls" style="margin-top: 15px;">
                    <button class="btn-secondary" onclick="saveAll()">💾 Guardar Todo</button>
                    <button class="btn-secondary" onclick="compareSelected()">📊 Comparar</button>
                </div>
            </div>
            
            <!-- Sugerencias de Mapping -->
            <div class="card">
                <h2>🎵 Mapping EEG → Música</h2>
                
                <div class="suggestions">
                    <h3>💡 Sugerencias Actuales</h3>
                    <div id="mapping-suggestions">
                        <div class="suggestion-item">
                            <strong>Tempo:</strong> <span id="suggested-bpm">80</span> BPM
                        </div>
                        <div class="suggestion-item">
                            <strong>Género:</strong> <span id="suggested-genre">ambient</span>
                        </div>
                        <div class="suggestion-item">
                            <strong>Densidad:</strong> <span id="suggested-density">0.5</span>
                        </div>
                        <div class="suggestion-item">
                            <strong>Brillo:</strong> <span id="suggested-brightness">0.5</span>
                        </div>
                        <div class="suggestion-item">
                            <strong>Escala:</strong> <span id="suggested-scale">C Mayor</span>
                        </div>
                    </div>
                </div>
                
                <div class="suggestions" style="margin-top: 15px; background: rgba(52, 152, 219, 0.1); border-color: rgba(52, 152, 219, 0.3);">
                    <h3 style="color: #3498db;">🎤 Prompt Sugerido para Lyria</h3>
                    <p id="suggested-prompt" style="font-style: italic; margin-top: 10px;">
                        calm ambient meditation music with soft pads
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let isRecording = false;
        let recordings = {};
        let bandHistory = { delta: [], theta: [], alpha: [], beta: [], gamma: [] };
        const maxHistory = 60;
        
        // Chart
        const ctx = document.getElementById('bandChart').getContext('2d');
        const bandChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(maxHistory).fill(''),
                datasets: [
                    { label: 'Delta', data: [], borderColor: '#9b59b6', tension: 0.4, pointRadius: 0 },
                    { label: 'Theta', data: [], borderColor: '#3498db', tension: 0.4, pointRadius: 0 },
                    { label: 'Alpha', data: [], borderColor: '#2ecc71', tension: 0.4, pointRadius: 0 },
                    { label: 'Beta', data: [], borderColor: '#f1c40f', tension: 0.4, pointRadius: 0 },
                    { label: 'Gamma', data: [], borderColor: '#e74c3c', tension: 0.4, pointRadius: 0 },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        function connect() {
            ws = new WebSocket('ws://localhost:8766');
            
            ws.onopen = () => {
                document.getElementById('status').classList.add('connected');
                console.log('Connected to EEG server');
            };
            
            ws.onclose = () => {
                document.getElementById('status').classList.remove('connected');
                setTimeout(connect, 2000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDisplay(data);
            };
        }
        
        function updateDisplay(data) {
            const bands = data.bands;
            const metrics = data.metrics;
            
            // Update band bars
            for (const band of ['delta', 'theta', 'alpha', 'beta', 'gamma']) {
                const val = bands[band] || 0;
                document.getElementById(`${band}-bar`).style.width = `${val * 100}%`;
                document.getElementById(`${band}-val`).textContent = val.toFixed(2);
                
                // Update history
                bandHistory[band].push(val);
                if (bandHistory[band].length > maxHistory) bandHistory[band].shift();
            }
            
            // Update chart
            bandChart.data.datasets[0].data = bandHistory.delta;
            bandChart.data.datasets[1].data = bandHistory.theta;
            bandChart.data.datasets[2].data = bandHistory.alpha;
            bandChart.data.datasets[3].data = bandHistory.beta;
            bandChart.data.datasets[4].data = bandHistory.gamma;
            bandChart.update('none');
            
            // Update metrics
            for (const metric of ['arousal', 'valence', 'focus', 'relaxation', 'meditation', 'engagement']) {
                const val = metrics[metric] || 0;
                document.getElementById(`${metric}-val`).textContent = val.toFixed(2);
            }
            
            // Update state indicator
            updateStateIndicator(metrics);
            
            // Update music suggestions
            updateMusicSuggestions(bands, metrics);
        }
        
        function updateStateIndicator(metrics) {
            let emoji = '🧘';
            let text = 'Neutral';
            
            if (metrics.relaxation > 0.6) {
                emoji = '😌'; text = 'Relajado';
            } else if (metrics.focus > 0.6) {
                emoji = '🎯'; text = 'Enfocado';
            } else if (metrics.meditation > 0.6) {
                emoji = '🧘'; text = 'Meditativo';
            } else if (metrics.engagement > 0.6) {
                emoji = '⚡'; text = 'Alerta';
            } else if (metrics.arousal > 0.7) {
                emoji = '🔥'; text = 'Alta Energía';
            } else if (metrics.arousal < 0.3) {
                emoji = '😴'; text = 'Baja Energía';
            }
            
            document.getElementById('state-emoji').textContent = emoji;
            document.getElementById('state-text').textContent = text;
        }
        
        function updateMusicSuggestions(bands, metrics) {
            // BPM
            const bpm = Math.round(60 + metrics.arousal * 140);
            document.getElementById('suggested-bpm').textContent = bpm;
            
            // Genre
            let genre = 'ambient';
            if (metrics.relaxation > 0.6) genre = 'ambient, meditation';
            else if (metrics.focus > 0.6) genre = 'lo-fi, minimal';
            else if (metrics.engagement > 0.6) genre = 'electronic, upbeat';
            else if (metrics.meditation > 0.6) genre = 'drone, ambient';
            document.getElementById('suggested-genre').textContent = genre;
            
            // Density
            const density = (1 - metrics.relaxation).toFixed(2);
            document.getElementById('suggested-density').textContent = density;
            
            // Brightness
            const brightness = (bands.gamma * 2).toFixed(2);
            document.getElementById('suggested-brightness').textContent = brightness;
            
            // Scale
            const scale = metrics.valence > 0.5 ? 'Mayor (positivo)' : 'Menor (contemplativo)';
            document.getElementById('suggested-scale').textContent = scale;
            
            // Prompt
            let energy = metrics.arousal > 0.6 ? 'energetic' : (metrics.arousal > 0.3 ? 'moderate' : 'calm');
            let mood = metrics.valence > 0.6 ? 'uplifting' : (metrics.valence > 0.4 ? 'peaceful' : 'contemplative');
            let style = metrics.relaxation > 0.6 ? 'ambient meditation' : (metrics.focus > 0.6 ? 'minimal electronic' : 'lo-fi chill');
            document.getElementById('suggested-prompt').textContent = 
                `${energy} ${mood} ${style} music, ${bpm} BPM`;
        }
        
        function startRecording() {
            const label = document.getElementById('label-input').value.trim();
            if (!label) {
                alert('Por favor ingresa una etiqueta');
                return;
            }
            
            ws.send(JSON.stringify({ action: 'start_recording', label: label }));
            isRecording = true;
            document.getElementById('recording-status').innerHTML = 
                '<span style="color: #e74c3c;">🔴 Grabando: ' + label + '</span>';
        }
        
        function stopRecording() {
            ws.send(JSON.stringify({ action: 'stop_recording' }));
            isRecording = false;
            document.getElementById('recording-status').innerHTML = 
                '<span style="color: #2ecc71;">✅ Grabación completada</span>';
            updateRecordingsList();
        }
        
        function saveAll() {
            ws.send(JSON.stringify({ action: 'save_all' }));
            alert('Sesiones guardadas');
        }
        
        function compareSelected() {
            ws.send(JSON.stringify({ action: 'get_comparison' }));
        }
        
        function updateRecordingsList() {
            ws.send(JSON.stringify({ action: 'get_sessions' }));
        }
        
        // Connect on load
        connect();
    </script>
</body>
</html>
"""


async def run_websocket_server():
    """Servidor WebSocket para el dashboard"""
    import websockets
    
    processor = EEGProcessor()
    sessions = {}
    current_recording = None
    current_label = None
    
    # Conectar a MUSE
    inlet = None
    try:
        from pylsl import StreamInlet, resolve_byprop
        logger.info("🔍 Buscando stream MUSE...")
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if streams:
            inlet = StreamInlet(streams[0])
            logger.info(f"✅ Conectado a MUSE: {streams[0].name()}")
        else:
            logger.warning("⚠️ No se encontró MUSE. Usando datos simulados.")
    except Exception as e:
        logger.warning(f"⚠️ Error conectando a MUSE: {e}. Usando datos simulados.")
    
    connected_clients = set()
    
    async def handler(websocket):
        nonlocal current_recording, current_label, sessions
        
        connected_clients.add(websocket)
        logger.info(f"📱 Cliente conectado. Total: {len(connected_clients)}")
        
        try:
            # Tarea para enviar datos EEG
            async def send_eeg_data():
                buffer = []
                
                while True:
                    try:
                        if inlet:
                            sample, _ = inlet.pull_sample(timeout=0.1)
                            if sample:
                                buffer.append(sample[:4])
                        else:
                            # Datos simulados
                            t = time.time()
                            sample = [
                                np.sin(2*np.pi*10*t) + np.random.randn()*0.1,
                                np.sin(2*np.pi*10*t + 0.5) + np.random.randn()*0.1,
                                np.sin(2*np.pi*10*t + 1.0) + np.random.randn()*0.1,
                                np.sin(2*np.pi*10*t + 1.5) + np.random.randn()*0.1,
                            ]
                            buffer.append(sample)
                            await asyncio.sleep(1/SAMPLE_RATE)
                        
                        if len(buffer) >= WINDOW_SIZE:
                            eeg_data = np.array(buffer).T
                            bands = processor.extract_bands(eeg_data)
                            metrics = processor.calculate_metrics(bands)
                            
                            data = {
                                'bands': bands,
                                'metrics': metrics,
                                'timestamp': time.time()
                            }
                            
                            # Grabar si está activo
                            if current_recording is not None:
                                current_recording.append({
                                    **bands, **metrics,
                                    'timestamp': time.time(),
                                    'label': current_label
                                })
                            
                            await websocket.send(json.dumps(data))
                            buffer = buffer[WINDOW_SIZE // 2:]
                            
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        await asyncio.sleep(0.1)
            
            # Tarea para recibir comandos
            async def receive_commands():
                nonlocal current_recording, current_label, sessions
                
                async for message in websocket:
                    try:
                        cmd = json.loads(message)
                        action = cmd.get('action')
                        
                        if action == 'start_recording':
                            current_label = cmd.get('label', 'unknown')
                            current_recording = []
                            logger.info(f"🎬 Grabando: {current_label}")
                        
                        elif action == 'stop_recording':
                            if current_recording and current_label:
                                if current_label not in sessions:
                                    sessions[current_label] = []
                                sessions[current_label].extend(current_recording)
                                logger.info(f"✅ Guardado: {current_label} ({len(current_recording)} muestras)")
                            current_recording = None
                            current_label = None
                        
                        elif action == 'save_all':
                            os.makedirs(DATA_DIR, exist_ok=True)
                            filename = f"sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            with open(os.path.join(DATA_DIR, filename), 'w') as f:
                                json.dump(sessions, f, indent=2)
                            logger.info(f"💾 Guardado: {filename}")
                        
                        elif action == 'get_sessions':
                            await websocket.send(json.dumps({
                                'type': 'sessions',
                                'data': {k: len(v) for k, v in sessions.items()}
                            }))
                            
                    except Exception as e:
                        logger.error(f"Error procesando comando: {e}")
            
            # Ejecutar ambas tareas
            await asyncio.gather(
                send_eeg_data(),
                receive_commands()
            )
            
        finally:
            connected_clients.remove(websocket)
            logger.info(f"📱 Cliente desconectado. Total: {len(connected_clients)}")
    
    server = await websockets.serve(handler, "localhost", 8766)
    logger.info("🌐 WebSocket server en ws://localhost:8766")
    await server.wait_closed()


def save_dashboard_html():
    """Guardar el HTML del dashboard"""
    filepath = os.path.join(os.path.dirname(__file__), "eeg_dashboard.html")
    with open(filepath, 'w') as f:
        f.write(DASHBOARD_HTML)
    logger.info(f"📄 Dashboard guardado en: {filepath}")
    return filepath


def main():
    """Iniciar el servidor y abrir el dashboard"""
    import webbrowser
    
    print("🧠 EEG Visualizer - Dashboard en Tiempo Real")
    print("=" * 50)
    
    # Guardar HTML
    html_path = save_dashboard_html()
    
    # Abrir en navegador
    print(f"\n📱 Abriendo dashboard en navegador...")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    print("\n🔌 Asegúrate de que MUSE está conectado:")
    print("   muselsl stream")
    
    print("\n🌐 Iniciando servidor WebSocket...")
    
    # Ejecutar servidor
    asyncio.run(run_websocket_server())


if __name__ == "__main__":
    main()
