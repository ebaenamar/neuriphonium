#!/usr/bin/env python3
"""
EEG Explorer - Herramienta para analizar y grabar estados cerebrales con MUSE
Permite explorar diferentes estados mentales y diseñar mappings creativos para música
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuración
SAMPLE_RATE = 256
WINDOW_SIZE = 256  # 1 segundo de datos
DATA_DIR = "eeg_recordings"

# Bandas EEG
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Canales MUSE
MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']


class EEGExplorer:
    """
    Herramienta interactiva para explorar estados cerebrales con MUSE.
    
    Funcionalidades:
    - Grabación de sesiones etiquetadas
    - Visualización en tiempo real de bandas
    - Análisis de patrones por estado
    - Comparación entre estados
    """
    
    def __init__(self):
        self.inlet = None
        self.is_recording = False
        self.current_session = []
        self.sessions = {}  # {label: [recordings]}
        
        # Historial para suavizado
        self.band_history = {band: deque(maxlen=10) for band in EEG_BANDS}
        
        # Crear directorio de datos
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def connect_muse(self) -> bool:
        """Conectar al stream LSL de MUSE"""
        try:
            from pylsl import StreamInlet, resolve_byprop
            
            logger.info("🔍 Buscando stream MUSE LSL...")
            streams = resolve_byprop('type', 'EEG', timeout=10)
            
            if not streams:
                logger.error("❌ No se encontró stream EEG.")
                logger.info("   Asegúrate de que MUSE está conectado y ejecuta:")
                logger.info("   muselsl stream")
                return False
            
            self.inlet = StreamInlet(streams[0])
            logger.info(f"✅ Conectado a: {streams[0].name()}")
            return True
            
        except ImportError:
            logger.error("❌ pylsl no instalado. Ejecuta: pip install pylsl")
            return False
        except Exception as e:
            logger.error(f"❌ Error conectando: {e}")
            return False
    
    def extract_band_powers(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Extraer potencias de bandas de frecuencia"""
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        band_powers = {}
        n_samples = eeg_data.shape[1]
        
        # FFT
        fft_vals = np.fft.fft(eeg_data, axis=1)
        freqs = np.fft.fftfreq(n_samples, 1.0 / SAMPLE_RATE)
        
        for band_name, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            power = np.mean(np.abs(fft_vals[:, mask]) ** 2)
            band_powers[band_name] = power
        
        # Normalizar
        total = sum(band_powers.values())
        if total > 0:
            band_powers = {k: v / total for k, v in band_powers.items()}
        
        # Suavizar
        for band, power in band_powers.items():
            self.band_history[band].append(power)
            band_powers[band] = np.mean(list(self.band_history[band]))
        
        return band_powers
    
    def calculate_metrics(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Calcular métricas cognitivas derivadas"""
        d = band_powers.get('delta', 0.2)
        t = band_powers.get('theta', 0.2)
        a = band_powers.get('alpha', 0.2)
        b = band_powers.get('beta', 0.2)
        g = band_powers.get('gamma', 0.2)
        
        return {
            # Métricas básicas
            'arousal': 0.1*d + 0.2*t + 0.3*a + 0.5*b + 0.7*g,
            'valence': np.clip((a - 0.3*t - 0.2*abs(b-t) + 0.5), 0, 1),
            
            # Ratios clásicos
            'alpha_theta_ratio': a / (t + 0.01),  # Relajación vs somnolencia
            'beta_alpha_ratio': b / (a + 0.01),   # Alerta vs relajación
            'theta_beta_ratio': t / (b + 0.01),   # Creatividad/meditación
            
            # Estados derivados
            'focus': np.clip(b / (t + 0.01) / 3, 0, 1),
            'relaxation': np.clip(a / (b + 0.01) / 2, 0, 1),
            'meditation': np.clip((t + a) / (b + g + 0.01) / 2, 0, 1),
            'engagement': np.clip((b + g) / (a + t + 0.01) / 2, 0, 1),
            
            # Asimetría hemisférica (si tenemos datos por canal)
            'frontal_asymmetry': 0.5,  # Placeholder - calcular con AF7/AF8
        }
    
    def display_realtime(self, band_powers: Dict[str, float], metrics: Dict[str, float]):
        """Mostrar estado actual en tiempo real"""
        # Barras ASCII para bandas
        def bar(value, width=20):
            filled = int(value * width)
            return '█' * filled + '░' * (width - filled)
        
        # Limpiar pantalla (opcional)
        # print("\033[H\033[J", end="")
        
        print("\n" + "=" * 60)
        print("🧠 ESTADO CEREBRAL EN TIEMPO REAL")
        print("=" * 60)
        
        print("\n📊 BANDAS DE FRECUENCIA:")
        print(f"  δ Delta  (0.5-4Hz)  [{bar(band_powers['delta'])}] {band_powers['delta']:.2f}")
        print(f"  θ Theta  (4-8Hz)    [{bar(band_powers['theta'])}] {band_powers['theta']:.2f}")
        print(f"  α Alpha  (8-13Hz)   [{bar(band_powers['alpha'])}] {band_powers['alpha']:.2f}")
        print(f"  β Beta   (13-30Hz)  [{bar(band_powers['beta'])}] {band_powers['beta']:.2f}")
        print(f"  γ Gamma  (30-50Hz)  [{bar(band_powers['gamma'])}] {band_powers['gamma']:.2f}")
        
        print("\n🎯 MÉTRICAS COGNITIVAS:")
        print(f"  Arousal     [{bar(metrics['arousal'])}] {metrics['arousal']:.2f}")
        print(f"  Valence     [{bar(metrics['valence'])}] {metrics['valence']:.2f}")
        print(f"  Focus       [{bar(metrics['focus'])}] {metrics['focus']:.2f}")
        print(f"  Relaxation  [{bar(metrics['relaxation'])}] {metrics['relaxation']:.2f}")
        print(f"  Meditation  [{bar(metrics['meditation'])}] {metrics['meditation']:.2f}")
        print(f"  Engagement  [{bar(metrics['engagement'])}] {metrics['engagement']:.2f}")
        
        # Interpretación
        print("\n💭 INTERPRETACIÓN:")
        if metrics['relaxation'] > 0.6:
            print("  → Estado RELAJADO (alpha dominante)")
        elif metrics['focus'] > 0.6:
            print("  → Estado ENFOCADO (beta dominante)")
        elif metrics['meditation'] > 0.6:
            print("  → Estado MEDITATIVO (theta+alpha)")
        elif metrics['engagement'] > 0.6:
            print("  → Estado ALERTA (beta+gamma)")
        else:
            print("  → Estado NEUTRAL")
    
    def record_session(self, label: str, duration: int = 30, 
                      display: bool = True, interval: float = 1.0):
        """
        Grabar una sesión etiquetada.
        
        Args:
            label: Etiqueta del estado (ej: 'relajado', 'pensando_en_playa')
            duration: Duración en segundos
            display: Mostrar visualización en tiempo real
            interval: Intervalo entre muestras
        """
        if not self.inlet:
            if not self.connect_muse():
                return None
        
        logger.info(f"\n🎬 GRABANDO: '{label}' por {duration} segundos")
        logger.info("   Prepárate... comenzando en 3 segundos")
        time.sleep(3)
        logger.info("   ▶️ GRABANDO...")
        
        recordings = []
        buffer = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample:
                    buffer.append(sample[:4])
                    
                    if len(buffer) >= WINDOW_SIZE:
                        eeg_data = np.array(buffer).T
                        band_powers = self.extract_band_powers(eeg_data)
                        metrics = self.calculate_metrics(band_powers)
                        
                        # Guardar
                        record = {
                            'timestamp': time.time(),
                            'elapsed': time.time() - start_time,
                            'label': label,
                            **band_powers,
                            **metrics
                        }
                        recordings.append(record)
                        
                        # Mostrar
                        if display:
                            self.display_realtime(band_powers, metrics)
                        
                        # Overlap
                        buffer = buffer[WINDOW_SIZE // 2:]
                        
                        time.sleep(interval)
                        
            except KeyboardInterrupt:
                logger.info("\n⏹️ Grabación interrumpida")
                break
        
        logger.info(f"\n✅ Grabación completada: {len(recordings)} muestras")
        
        # Guardar en sesiones
        if label not in self.sessions:
            self.sessions[label] = []
        self.sessions[label].extend(recordings)
        
        return recordings
    
    def analyze_session(self, label: str) -> Dict:
        """Analizar estadísticas de una sesión grabada"""
        if label not in self.sessions:
            logger.error(f"❌ No hay datos para '{label}'")
            return None
        
        data = self.sessions[label]
        df = pd.DataFrame(data)
        
        # Estadísticas
        stats = {
            'label': label,
            'n_samples': len(data),
            'duration': df['elapsed'].max() if len(df) > 0 else 0,
            'bands': {},
            'metrics': {}
        }
        
        for band in EEG_BANDS:
            stats['bands'][band] = {
                'mean': df[band].mean(),
                'std': df[band].std(),
                'min': df[band].min(),
                'max': df[band].max()
            }
        
        for metric in ['arousal', 'valence', 'focus', 'relaxation', 'meditation', 'engagement']:
            if metric in df.columns:
                stats['metrics'][metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std()
                }
        
        return stats
    
    def compare_sessions(self, labels: List[str]) -> pd.DataFrame:
        """Comparar múltiples sesiones"""
        comparisons = []
        
        for label in labels:
            stats = self.analyze_session(label)
            if stats:
                row = {'label': label}
                for band, vals in stats['bands'].items():
                    row[f'{band}_mean'] = vals['mean']
                for metric, vals in stats['metrics'].items():
                    row[f'{metric}_mean'] = vals['mean']
                comparisons.append(row)
        
        return pd.DataFrame(comparisons)
    
    def save_sessions(self, filename: str = None):
        """Guardar todas las sesiones a archivo"""
        if not filename:
            filename = f"sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.sessions, f, indent=2)
        
        logger.info(f"💾 Sesiones guardadas en: {filepath}")
        return filepath
    
    def load_sessions(self, filename: str):
        """Cargar sesiones desde archivo"""
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'r') as f:
            self.sessions = json.load(f)
        
        logger.info(f"📂 Sesiones cargadas: {list(self.sessions.keys())}")
    
    def suggest_mappings(self) -> Dict:
        """
        Sugerir mappings creativos basados en los datos grabados.
        
        Retorna diccionario con sugerencias de mapping EEG → Música
        """
        if not self.sessions:
            logger.warning("⚠️ No hay sesiones grabadas para analizar")
            return {}
        
        # Analizar todas las sesiones
        all_stats = {label: self.analyze_session(label) for label in self.sessions}
        
        suggestions = {
            'basic_mappings': {
                'arousal → tempo': {
                    'description': 'Mayor activación = música más rápida',
                    'formula': 'BPM = 60 + arousal * 140',
                    'range': '60-200 BPM'
                },
                'relaxation → density': {
                    'description': 'Más relajado = música más espaciada',
                    'formula': 'density = 1 - relaxation',
                    'range': '0-1'
                },
                'focus → brightness': {
                    'description': 'Más enfocado = tonos más brillantes',
                    'formula': 'brightness = focus',
                    'range': '0-1'
                },
                'valence → scale': {
                    'description': 'Positivo = mayor, negativo = menor',
                    'formula': 'scale = MAJOR if valence > 0.5 else MINOR',
                    'options': 'C_MAJOR_A_MINOR, D_MINOR, etc.'
                }
            },
            'creative_mappings': {
                'meditation → genre': {
                    'description': 'Estado meditativo cambia el género',
                    'low': 'electronic, techno',
                    'medium': 'lo-fi, ambient',
                    'high': 'drone, meditation'
                },
                'engagement → instruments': {
                    'description': 'Engagement controla complejidad instrumental',
                    'low': 'soft pads, single instrument',
                    'medium': 'piano, strings',
                    'high': 'full band, drums, bass'
                },
                'alpha_theta_ratio → creativity_mode': {
                    'description': 'Ratio indica estado creativo',
                    'high_alpha': 'melodic, harmonic',
                    'high_theta': 'experimental, ambient'
                },
                'frontal_asymmetry → emotional_tone': {
                    'description': 'Asimetría frontal indica valencia emocional',
                    'left_dominant': 'approach, positive',
                    'right_dominant': 'withdrawal, melancholic'
                }
            },
            'advanced_mappings': {
                'band_ratios → prompt_weights': {
                    'description': 'Usar ratios de bandas como pesos de prompts',
                    'example': [
                        'WeightedPrompt("calm", weight=alpha)',
                        'WeightedPrompt("energetic", weight=beta)',
                        'WeightedPrompt("dreamy", weight=theta)'
                    ]
                },
                'state_transitions → music_transitions': {
                    'description': 'Detectar cambios de estado para transiciones suaves',
                    'trigger': 'Cuando arousal cambia > 0.2 en 5 segundos',
                    'action': 'Gradualmente cambiar prompts'
                },
                'personal_signatures': {
                    'description': 'Patrones únicos de tu cerebro',
                    'note': 'Analiza tus sesiones para encontrar tus "firmas" cerebrales'
                }
            }
        }
        
        # Agregar sugerencias personalizadas basadas en datos
        if all_stats:
            suggestions['personalized'] = {}
            
            for label, stats in all_stats.items():
                if stats and stats['bands']:
                    dominant_band = max(stats['bands'].items(), 
                                       key=lambda x: x[1]['mean'])[0]
                    suggestions['personalized'][label] = {
                        'dominant_band': dominant_band,
                        'suggested_genre': self._band_to_genre(dominant_band),
                        'suggested_tempo': self._metrics_to_tempo(stats['metrics'])
                    }
        
        return suggestions
    
    def _band_to_genre(self, band: str) -> str:
        """Mapear banda dominante a género sugerido"""
        mapping = {
            'delta': 'ambient drone, sleep music',
            'theta': 'meditation, lo-fi, dream pop',
            'alpha': 'chill, acoustic, soft jazz',
            'beta': 'electronic, pop, rock',
            'gamma': 'techno, EDM, high energy'
        }
        return mapping.get(band, 'ambient')
    
    def _metrics_to_tempo(self, metrics: Dict) -> int:
        """Calcular tempo sugerido basado en métricas"""
        if not metrics or 'arousal' not in metrics:
            return 80
        arousal = metrics['arousal']['mean']
        return int(60 + arousal * 140)


def interactive_session():
    """Sesión interactiva de exploración"""
    explorer = EEGExplorer()
    
    print("🧠 EEG EXPLORER - Exploración de Estados Cerebrales")
    print("=" * 60)
    print("""
    Comandos disponibles:
    
    1. connect     - Conectar a MUSE
    2. monitor     - Monitorear en tiempo real (sin grabar)
    3. record      - Grabar sesión etiquetada
    4. analyze     - Analizar sesión grabada
    5. compare     - Comparar sesiones
    6. mappings    - Ver sugerencias de mappings
    7. save        - Guardar sesiones
    8. load        - Cargar sesiones
    9. list        - Listar sesiones grabadas
    0. exit        - Salir
    
    Ejemplos de etiquetas para grabar:
    - 'relajado_ojos_cerrados'
    - 'pensando_en_playa'
    - 'resolviendo_problema'
    - 'escuchando_musica_favorita'
    - 'meditando'
    - 'estresado'
    """)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == '1' or cmd == 'connect':
                explorer.connect_muse()
            
            elif cmd == '2' or cmd == 'monitor':
                if not explorer.inlet:
                    explorer.connect_muse()
                if explorer.inlet:
                    print("Monitoreando... (Ctrl+C para parar)")
                    explorer.record_session('_monitor', duration=300, display=True)
            
            elif cmd == '3' or cmd == 'record':
                label = input("Etiqueta del estado: ").strip()
                duration = int(input("Duración (segundos) [30]: ") or "30")
                explorer.record_session(label, duration=duration)
            
            elif cmd == '4' or cmd == 'analyze':
                label = input("Etiqueta a analizar: ").strip()
                stats = explorer.analyze_session(label)
                if stats:
                    print(f"\n📊 Análisis de '{label}':")
                    print(f"   Muestras: {stats['n_samples']}")
                    print(f"   Duración: {stats['duration']:.1f}s")
                    print("\n   Bandas (media ± std):")
                    for band, vals in stats['bands'].items():
                        print(f"     {band}: {vals['mean']:.3f} ± {vals['std']:.3f}")
                    print("\n   Métricas:")
                    for metric, vals in stats['metrics'].items():
                        print(f"     {metric}: {vals['mean']:.3f}")
            
            elif cmd == '5' or cmd == 'compare':
                labels = input("Etiquetas a comparar (separadas por coma): ").split(',')
                labels = [l.strip() for l in labels]
                df = explorer.compare_sessions(labels)
                print("\n📊 Comparación:")
                print(df.to_string())
            
            elif cmd == '6' or cmd == 'mappings':
                suggestions = explorer.suggest_mappings()
                print("\n🎵 SUGERENCIAS DE MAPPINGS EEG → MÚSICA:")
                print(json.dumps(suggestions, indent=2))
            
            elif cmd == '7' or cmd == 'save':
                explorer.save_sessions()
            
            elif cmd == '8' or cmd == 'load':
                files = os.listdir(DATA_DIR)
                json_files = [f for f in files if f.endswith('.json')]
                print(f"Archivos disponibles: {json_files}")
                filename = input("Archivo a cargar: ").strip()
                explorer.load_sessions(filename)
            
            elif cmd == '9' or cmd == 'list':
                print(f"Sesiones grabadas: {list(explorer.sessions.keys())}")
                for label, data in explorer.sessions.items():
                    print(f"  - {label}: {len(data)} muestras")
            
            elif cmd == '0' or cmd == 'exit':
                print("👋 ¡Hasta luego!")
                break
            
            else:
                print("Comando no reconocido")
                
        except KeyboardInterrupt:
            print("\n")
            continue
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_session()
