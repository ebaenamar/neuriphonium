#!/usr/bin/env python3
"""
MUSE Motion Sensor Dashboard
Visualizes accelerometer and gyroscope data from MUSE headband in real-time.

Uses MuseConnectionManager from muse_adapter.py for proper connection handling.

Usage:
    1. Start MUSE stream with motion sensors:
       muselsl stream -c -g
    2. Run this dashboard:
       venv/bin/python motion_sensor_dashboard.py
"""

import os
import asyncio
import json
import time
import logging
import webbrowser

import numpy as np

from muse_adapter import MuseConnectionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = 8768

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MUSE Motion Sensor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #e6edf3;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 6px; font-size: 1.8em; }
        h1 span { font-size: 0.5em; opacity: 0.6; display: block; font-weight: normal; }
        .subtitle { text-align: center; opacity: 0.5; margin-bottom: 20px; font-size: 0.85em; }

        .connection-bar {
            display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;
        }
        .conn-badge {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 16px; border-radius: 20px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.85em;
        }
        .conn-dot {
            width: 10px; height: 10px; border-radius: 50%; background: #ff4444;
        }
        .conn-dot.connected { background: #44ff44; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 20px;
        }
        .card {
            background: rgba(255,255,255,0.04);
            border-radius: 14px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .card h2 { font-size: 1.05em; margin-bottom: 14px; }
        .card.full { grid-column: 1 / -1; }

        .axis-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .axis-box {
            text-align: center; padding: 14px 8px;
            background: rgba(255,255,255,0.04); border-radius: 10px;
        }
        .axis-label { font-size: 0.75em; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; }
        .axis-value { font-size: 1.6em; font-weight: bold; font-family: monospace; margin: 4px 0; }
        .axis-unit { font-size: 0.7em; opacity: 0.5; }
        .axis-bar-container {
            margin-top: 8px; height: 4px; background: rgba(255,255,255,0.1);
            border-radius: 2px; overflow: hidden;
        }
        .axis-bar { height: 100%; border-radius: 2px; transition: width 0.1s; }
        .x-color { color: #ff6b6b; } .x-bar { background: #ff6b6b; }
        .y-color { color: #4ecdc4; } .y-bar { background: #4ecdc4; }
        .z-color { color: #ffe66d; } .z-bar { background: #ffe66d; }

        .magnitude-display {
            text-align: center; padding: 16px;
            background: rgba(255,255,255,0.04); border-radius: 10px;
        }
        .magnitude-value { font-size: 2em; font-weight: bold; font-family: monospace; }
        .magnitude-label { font-size: 0.75em; opacity: 0.5; }

        .stat-table { width: 100%; font-size: 0.82em; }
        .stat-table th, .stat-table td {
            text-align: center; padding: 6px 4px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .stat-table th { opacity: 0.5; font-weight: 600; }
        .stat-table td { font-family: monospace; }

        .chart-container { height: 200px; margin-top: 10px; }
        .chart-container.tall { height: 280px; }

        .precision-info { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .precision-item {
            padding: 10px; background: rgba(255,255,255,0.04);
            border-radius: 8px; text-align: center;
        }
        .precision-value { font-size: 1.3em; font-weight: bold; font-family: monospace; }
        .precision-label { font-size: 0.72em; opacity: 0.5; margin-top: 2px; }

        .viz-3d {
            display: flex; justify-content: center; align-items: center;
            height: 220px; position: relative;
        }
        .tilt-indicator {
            width: 180px; height: 180px; border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.15);
            position: relative; background: rgba(0,0,0,0.2);
        }
        .tilt-dot {
            width: 16px; height: 16px; border-radius: 50%;
            background: #4ecdc4; position: absolute;
            top: 50%; left: 50%; transform: translate(-50%, -50%);
            transition: transform 0.1s ease-out;
            box-shadow: 0 0 12px rgba(78, 205, 196, 0.6);
        }
        .tilt-cross-h, .tilt-cross-v { position: absolute; background: rgba(255,255,255,0.08); }
        .tilt-cross-h { width: 100%; height: 1px; top: 50%; }
        .tilt-cross-v { width: 1px; height: 100%; left: 50%; }
        .tilt-label { position: absolute; font-size: 0.65em; opacity: 0.4; }
        .tilt-label.top { top: 4px; left: 50%; transform: translateX(-50%); }
        .tilt-label.bottom { bottom: 4px; left: 50%; transform: translateX(-50%); }
        .tilt-label.left { left: 6px; top: 50%; transform: translateY(-50%); }
        .tilt-label.right { right: 6px; top: 50%; transform: translateY(-50%); }

        .info-row {
            display: flex; justify-content: space-between;
            padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.85em;
        }
        .info-row:last-child { border-bottom: none; }
        .info-label { opacity: 0.6; }
        .info-value { font-family: monospace; font-weight: 600; }

        .sample-rate { display: flex; gap: 12px; justify-content: center; margin-bottom: 16px; }
        .rate-badge {
            padding: 4px 12px; border-radius: 12px;
            background: rgba(78,205,196,0.15); border: 1px solid rgba(78,205,196,0.3);
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MUSE Motion Sensor Dashboard
            <span>Accelerometer & Gyroscope Precision Analysis</span>
        </h1>
        <p class="subtitle">MUSE 2 | Accel: 3-axis, 52 Hz, g | Gyro: 3-axis, 52 Hz, dps</p>

        <div class="connection-bar">
            <div class="conn-badge">
                <div class="conn-dot" id="acc-dot"></div>
                <span>Accelerometer</span>
            </div>
            <div class="conn-badge">
                <div class="conn-dot" id="gyro-dot"></div>
                <span>Gyroscope</span>
            </div>
            <div class="conn-badge">
                <span id="elapsed">0.0s</span>
            </div>
        </div>

        <div class="sample-rate">
            <div class="rate-badge" id="acc-rate">Accel: -- Hz</div>
            <div class="rate-badge" id="gyro-rate">Gyro: -- Hz</div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Accelerometer (g)</h2>
                <div class="axis-grid">
                    <div class="axis-box">
                        <div class="axis-label x-color">X</div>
                        <div class="axis-value x-color" id="acc-x">0.000</div>
                        <div class="axis-unit">g</div>
                        <div class="axis-bar-container"><div class="axis-bar x-bar" id="acc-x-bar" style="width:50%"></div></div>
                    </div>
                    <div class="axis-box">
                        <div class="axis-label y-color">Y</div>
                        <div class="axis-value y-color" id="acc-y">0.000</div>
                        <div class="axis-unit">g</div>
                        <div class="axis-bar-container"><div class="axis-bar y-bar" id="acc-y-bar" style="width:50%"></div></div>
                    </div>
                    <div class="axis-box">
                        <div class="axis-label z-color">Z</div>
                        <div class="axis-value z-color" id="acc-z">0.000</div>
                        <div class="axis-unit">g</div>
                        <div class="axis-bar-container"><div class="axis-bar z-bar" id="acc-z-bar" style="width:50%"></div></div>
                    </div>
                </div>
                <div class="magnitude-display" style="margin-top:14px;">
                    <div class="magnitude-value" id="acc-mag">0.000</div>
                    <div class="magnitude-label">|a| magnitude (g)</div>
                </div>
            </div>

            <div class="card">
                <h2>Gyroscope (dps)</h2>
                <div class="axis-grid">
                    <div class="axis-box">
                        <div class="axis-label x-color">X</div>
                        <div class="axis-value x-color" id="gyro-x">0.0</div>
                        <div class="axis-unit">dps</div>
                        <div class="axis-bar-container"><div class="axis-bar x-bar" id="gyro-x-bar" style="width:50%"></div></div>
                    </div>
                    <div class="axis-box">
                        <div class="axis-label y-color">Y</div>
                        <div class="axis-value y-color" id="gyro-y">0.0</div>
                        <div class="axis-unit">dps</div>
                        <div class="axis-bar-container"><div class="axis-bar y-bar" id="gyro-y-bar" style="width:50%"></div></div>
                    </div>
                    <div class="axis-box">
                        <div class="axis-label z-color">Z</div>
                        <div class="axis-value z-color" id="gyro-z">0.0</div>
                        <div class="axis-unit">dps</div>
                        <div class="axis-bar-container"><div class="axis-bar z-bar" id="gyro-z-bar" style="width:50%"></div></div>
                    </div>
                </div>
                <div class="magnitude-display" style="margin-top:14px;">
                    <div class="magnitude-value" id="gyro-mag">0.0</div>
                    <div class="magnitude-label">|w| magnitude (dps)</div>
                </div>
            </div>

            <div class="card">
                <h2>Tilt Orientation</h2>
                <div class="viz-3d">
                    <div class="tilt-indicator">
                        <div class="tilt-cross-h"></div>
                        <div class="tilt-cross-v"></div>
                        <div class="tilt-label top">Fwd</div>
                        <div class="tilt-label bottom">Back</div>
                        <div class="tilt-label left">Left</div>
                        <div class="tilt-label right">Right</div>
                        <div class="tilt-dot" id="tilt-dot"></div>
                    </div>
                </div>
                <div class="info-row">
                    <span class="info-label">Pitch (deg)</span>
                    <span class="info-value" id="pitch">0.0</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Roll (deg)</span>
                    <span class="info-value" id="roll">0.0</span>
                </div>
            </div>

            <div class="card">
                <h2>Precision & Noise Analysis</h2>
                <div class="precision-info">
                    <div class="precision-item">
                        <div class="precision-value" id="acc-noise">--</div>
                        <div class="precision-label">Accel Noise (mg)</div>
                    </div>
                    <div class="precision-item">
                        <div class="precision-value" id="gyro-noise">--</div>
                        <div class="precision-label">Gyro Noise (mdps)</div>
                    </div>
                    <div class="precision-item">
                        <div class="precision-value" id="acc-resolution">--</div>
                        <div class="precision-label">Accel Resolution (mg)</div>
                    </div>
                    <div class="precision-item">
                        <div class="precision-value" id="gyro-resolution">--</div>
                        <div class="precision-label">Gyro Resolution (mdps)</div>
                    </div>
                </div>
                <div style="margin-top:14px;">
                    <div class="info-row">
                        <span class="info-label">Accel Dynamic Range</span>
                        <span class="info-value">+/- 2.0 g</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Gyro Dynamic Range</span>
                        <span class="info-value">+/- 250 dps</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Accel Theoretical Resolution</span>
                        <span class="info-value">0.061 mg</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Gyro Theoretical Resolution</span>
                        <span class="info-value">8.75 mdps</span>
                    </div>
                </div>
            </div>

            <div class="card full">
                <h2>Accelerometer Time Series (last ~2s)</h2>
                <div class="chart-container tall">
                    <canvas id="acc-chart"></canvas>
                </div>
            </div>

            <div class="card full">
                <h2>Gyroscope Time Series (last ~2s)</h2>
                <div class="chart-container tall">
                    <canvas id="gyro-chart"></canvas>
                </div>
            </div>

            <div class="card full">
                <h2>Statistical Summary (10s window)</h2>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h3 style="font-size:0.9em; margin-bottom:8px; opacity:0.7;">Accelerometer (g)</h3>
                        <table class="stat-table">
                            <thead><tr><th></th><th>X</th><th>Y</th><th>Z</th><th>|a|</th></tr></thead>
                            <tbody>
                                <tr><td>Mean</td><td id="acc-mean-x">--</td><td id="acc-mean-y">--</td><td id="acc-mean-z">--</td><td id="acc-mean-mag">--</td></tr>
                                <tr><td>Std</td><td id="acc-std-x">--</td><td id="acc-std-y">--</td><td id="acc-std-z">--</td><td id="acc-std-mag">--</td></tr>
                                <tr><td>Min</td><td id="acc-min-x">--</td><td id="acc-min-y">--</td><td id="acc-min-z">--</td><td rowspan="3" style="vertical-align:middle; opacity:0.4;">--</td></tr>
                                <tr><td>Max</td><td id="acc-max-x">--</td><td id="acc-max-y">--</td><td id="acc-max-z">--</td></tr>
                                <tr><td>Range</td><td id="acc-range-x">--</td><td id="acc-range-y">--</td><td id="acc-range-z">--</td></tr>
                            </tbody>
                        </table>
                    </div>
                    <div>
                        <h3 style="font-size:0.9em; margin-bottom:8px; opacity:0.7;">Gyroscope (dps)</h3>
                        <table class="stat-table">
                            <thead><tr><th></th><th>X</th><th>Y</th><th>Z</th><th>|w|</th></tr></thead>
                            <tbody>
                                <tr><td>Mean</td><td id="gyro-mean-x">--</td><td id="gyro-mean-y">--</td><td id="gyro-mean-z">--</td><td id="gyro-mean-mag">--</td></tr>
                                <tr><td>Std</td><td id="gyro-std-x">--</td><td id="gyro-std-y">--</td><td id="gyro-std-z">--</td><td id="gyro-std-mag">--</td></tr>
                                <tr><td>Min</td><td id="gyro-min-x">--</td><td id="gyro-min-y">--</td><td id="gyro-min-z">--</td><td rowspan="3" style="vertical-align:middle; opacity:0.4;">--</td></tr>
                                <tr><td>Max</td><td id="gyro-max-x">--</td><td id="gyro-max-y">--</td><td id="gyro-max-z">--</td></tr>
                                <tr><td>Range</td><td id="gyro-range-x">--</td><td id="gyro-range-y">--</td><td id="gyro-range-z">--</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let accChart, gyroChart;
        const MAX_POINTS = 100;

        function initCharts() {
            const commonOpts = {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#e6edf3', font: { size: 11 } } } },
                elements: { point: { radius: 0 }, line: { borderWidth: 1.5 } }
            };

            accChart = new Chart(document.getElementById('acc-chart'), {
                type: 'line',
                data: {
                    labels: Array(MAX_POINTS).fill(''),
                    datasets: [
                        { label: 'X (g)', data: Array(MAX_POINTS).fill(null), borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,0.1)' },
                        { label: 'Y (g)', data: Array(MAX_POINTS).fill(null), borderColor: '#4ecdc4', backgroundColor: 'rgba(78,205,196,0.1)' },
                        { label: 'Z (g)', data: Array(MAX_POINTS).fill(null), borderColor: '#ffe66d', backgroundColor: 'rgba(255,230,109,0.1)' }
                    ]
                },
                options: { ...commonOpts, scales: { ...commonOpts.scales, y: { ...commonOpts.scales.y, suggestedMin: -2, suggestedMax: 2 } } }
            });

            gyroChart = new Chart(document.getElementById('gyro-chart'), {
                type: 'line',
                data: {
                    labels: Array(MAX_POINTS).fill(''),
                    datasets: [
                        { label: 'X (dps)', data: Array(MAX_POINTS).fill(null), borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,0.1)' },
                        { label: 'Y (dps)', data: Array(MAX_POINTS).fill(null), borderColor: '#4ecdc4', backgroundColor: 'rgba(78,205,196,0.1)' },
                        { label: 'Z (dps)', data: Array(MAX_POINTS).fill(null), borderColor: '#ffe66d', backgroundColor: 'rgba(255,230,109,0.1)' }
                    ]
                },
                options: { ...commonOpts, scales: { ...commonOpts.scales, y: { ...commonOpts.scales.y, suggestedMin: -250, suggestedMax: 250 } } }
            });
        }

        function fmt(v, dec) {
            if (dec === undefined) dec = 3;
            if (v === null || v === undefined || isNaN(v)) return '--';
            return v.toFixed(dec);
        }

        function setBar(id, val, max) {
            var el = document.getElementById(id);
            var pct = Math.min(100, Math.max(0, ((val + max) / (2 * max)) * 100));
            el.style.width = pct + '%';
        }

        function updateChart(chart, series) {
            if (!series || !chart) return;
            var n = series.x ? series.x.length : 0;
            var pad = MAX_POINTS - n;
            chart.data.datasets[0].data = (series.x || []).concat(Array(Math.max(0, pad)).fill(null));
            chart.data.datasets[1].data = (series.y || []).concat(Array(Math.max(0, pad)).fill(null));
            chart.data.datasets[2].data = (series.z || []).concat(Array(Math.max(0, pad)).fill(null));
            chart.update('none');
        }

        function connect() {
            ws = new WebSocket('ws://localhost:8768');

            ws.onopen = function() {
                console.log('WebSocket connected');
            };

            ws.onclose = function() {
                console.log('WebSocket disconnected - reconnecting...');
                setTimeout(connect, 2000);
            };

            ws.onmessage = function(event) {
                var d = JSON.parse(event.data);

                document.getElementById('acc-dot').classList.toggle('connected', d.sensors.acc.connected);
                document.getElementById('gyro-dot').classList.toggle('connected', d.sensors.gyro.connected);
                document.getElementById('elapsed').textContent = d.elapsed.toFixed(1) + 's';

                if (d.elapsed > 1) {
                    if (d.sensors.acc.connected) document.getElementById('acc-rate').textContent = 'Accel: ' + d.sensors.acc.measured_rate.toFixed(1) + ' Hz';
                    if (d.sensors.gyro.connected) document.getElementById('gyro-rate').textContent = 'Gyro: ' + d.sensors.gyro.measured_rate.toFixed(1) + ' Hz';
                }

                // Accel
                var acc = d.sensors.acc;
                if (acc.latest) {
                    document.getElementById('acc-x').textContent = fmt(acc.latest[0]);
                    document.getElementById('acc-y').textContent = fmt(acc.latest[1]);
                    document.getElementById('acc-z').textContent = fmt(acc.latest[2]);
                    setBar('acc-x-bar', acc.latest[0], 2);
                    setBar('acc-y-bar', acc.latest[1], 2);
                    setBar('acc-z-bar', acc.latest[2], 2);
                    document.getElementById('acc-mag').textContent = fmt(acc.magnitude_latest);

                    var pitch = Math.atan2(acc.latest[0], Math.abs(acc.latest[2])) * 180 / Math.PI;
                    var roll = Math.atan2(acc.latest[1], Math.abs(acc.latest[2])) * 180 / Math.PI;
                    document.getElementById('pitch').textContent = fmt(pitch, 1);
                    document.getElementById('roll').textContent = fmt(roll, 1);
                    var dotX = Math.max(-80, Math.min(80, -acc.latest[1] * 90));
                    var dotY = Math.max(-80, Math.min(80, acc.latest[0] * 90));
                    document.getElementById('tilt-dot').style.transform = 'translate(calc(-50% + ' + dotX + 'px), calc(-50% + ' + dotY + 'px))';

                    document.getElementById('acc-noise').textContent = fmt(acc.magnitude_std * 1000, 1);
                    document.getElementById('acc-resolution').textContent = '0.061';
                }

                // Gyro
                var gyro = d.sensors.gyro;
                if (gyro.latest) {
                    document.getElementById('gyro-x').textContent = fmt(gyro.latest[0], 1);
                    document.getElementById('gyro-y').textContent = fmt(gyro.latest[1], 1);
                    document.getElementById('gyro-z').textContent = fmt(gyro.latest[2], 1);
                    setBar('gyro-x-bar', gyro.latest[0], 250);
                    setBar('gyro-y-bar', gyro.latest[1], 250);
                    setBar('gyro-z-bar', gyro.latest[2], 250);
                    document.getElementById('gyro-mag').textContent = fmt(gyro.magnitude_latest, 1);
                    document.getElementById('gyro-noise').textContent = fmt(gyro.magnitude_std * 1000, 1);
                    document.getElementById('gyro-resolution').textContent = '8.75';
                }

                // Charts
                updateChart(accChart, d.acc_series);
                updateChart(gyroChart, d.gyro_series);

                // Stats
                if (acc.mean) {
                    document.getElementById('acc-mean-x').textContent = fmt(acc.mean[0]);
                    document.getElementById('acc-mean-y').textContent = fmt(acc.mean[1]);
                    document.getElementById('acc-mean-z').textContent = fmt(acc.mean[2]);
                    document.getElementById('acc-mean-mag').textContent = fmt(acc.magnitude_mean);
                    document.getElementById('acc-std-x').textContent = fmt(acc.std[0]);
                    document.getElementById('acc-std-y').textContent = fmt(acc.std[1]);
                    document.getElementById('acc-std-z').textContent = fmt(acc.std[2]);
                    document.getElementById('acc-std-mag').textContent = fmt(acc.magnitude_std);
                    document.getElementById('acc-min-x').textContent = fmt(acc.min[0]);
                    document.getElementById('acc-min-y').textContent = fmt(acc.min[1]);
                    document.getElementById('acc-min-z').textContent = fmt(acc.min[2]);
                    document.getElementById('acc-max-x').textContent = fmt(acc.max[0]);
                    document.getElementById('acc-max-y').textContent = fmt(acc.max[1]);
                    document.getElementById('acc-max-z').textContent = fmt(acc.max[2]);
                    document.getElementById('acc-range-x').textContent = fmt(acc.range[0]);
                    document.getElementById('acc-range-y').textContent = fmt(acc.range[1]);
                    document.getElementById('acc-range-z').textContent = fmt(acc.range[2]);
                }

                if (gyro.mean) {
                    document.getElementById('gyro-mean-x').textContent = fmt(gyro.mean[0], 1);
                    document.getElementById('gyro-mean-y').textContent = fmt(gyro.mean[1], 1);
                    document.getElementById('gyro-mean-z').textContent = fmt(gyro.mean[2], 1);
                    document.getElementById('gyro-mean-mag').textContent = fmt(gyro.magnitude_mean, 1);
                    document.getElementById('gyro-std-x').textContent = fmt(gyro.std[0], 1);
                    document.getElementById('gyro-std-y').textContent = fmt(gyro.std[1], 1);
                    document.getElementById('gyro-std-z').textContent = fmt(gyro.std[2], 1);
                    document.getElementById('gyro-std-mag').textContent = fmt(gyro.magnitude_std, 1);
                    document.getElementById('gyro-min-x').textContent = fmt(gyro.min[0], 1);
                    document.getElementById('gyro-min-y').textContent = fmt(gyro.min[1], 1);
                    document.getElementById('gyro-min-z').textContent = fmt(gyro.min[2], 1);
                    document.getElementById('gyro-max-x').textContent = fmt(gyro.max[0], 1);
                    document.getElementById('gyro-max-y').textContent = fmt(gyro.max[1], 1);
                    document.getElementById('gyro-max-z').textContent = fmt(gyro.max[2], 1);
                    document.getElementById('gyro-range-x').textContent = fmt(gyro.range[0], 1);
                    document.getElementById('gyro-range-y').textContent = fmt(gyro.range[1], 1);
                    document.getElementById('gyro-range-z').textContent = fmt(gyro.range[2], 1);
                }
            };
        }

        initCharts();
        connect();
    </script>
</body>
</html>
"""


def save_dashboard():
    filepath = os.path.join(os.path.dirname(__file__), "motion_dashboard.html")
    with open(filepath, 'w') as f:
        f.write(DASHBOARD_HTML)
    return filepath


async def run_server():
    import websockets

    mgr = MuseConnectionManager(history_seconds=10)
    results = mgr.connect_all()
    mgr.start()

    connected_clients = set()

    async def broadcast(message):
        if connected_clients:
            await asyncio.gather(*[client.send(json.dumps(message)) for client in connected_clients])

    async def handler(websocket):
        connected_clients.add(websocket)
        logger.info(f"Client connected ({len(connected_clients)})")
        try:
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            connected_clients.discard(websocket)
            logger.info(f"Client disconnected ({len(connected_clients)})")

    async def broadcast_loop():
        while True:
            status = mgr.get_status()
            payload = {
                'running': status['running'],
                'elapsed': status['elapsed'],
                'sensors': status['sensors'],
                'acc_series': mgr.get_series('acc', 100),
                'gyro_series': mgr.get_series('gyro', 100),
            }
            await broadcast(payload)
            await asyncio.sleep(0.1)

    asyncio.create_task(broadcast_loop())

    server = await websockets.serve(handler, "localhost", PORT)
    logger.info(f"Dashboard server on ws://localhost:{PORT}")
    await server.wait_closed()


def main():
    print("""
    +----------------------------------------------------------+
    |        MUSE Motion Sensor Dashboard                      |
    |   Accelerometer & Gyroscope Precision Analysis           |
    +----------------------------------------------------------+
    """)

    html_path = save_dashboard()
    print(f"Dashboard: {html_path}")

    print("\nMake sure MUSE is streaming: muselsl stream -c -g")

    webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print("\nStarting server...")
    asyncio.run(run_server())


if __name__ == '__main__':
    main()
