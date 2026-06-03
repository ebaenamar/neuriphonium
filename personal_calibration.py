#!/usr/bin/env python3
"""
Sistema de calibración personalizada para mapeo individual de estados cognitivos
Cada cerebro es único - este sistema aprende TUS patrones específicos
"""

import json
import numpy as np
from collections import deque
from typing import Dict, Optional
from pathlib import Path
import time


class PersonalCalibration:
    """
    Calibración personalizada de thresholds neurocognitivos
    
    Aprende y se adapta a los patrones únicos de cada usuario:
    - Baseline personal de cada banda EEG
    - Thresholds dinámicos basados en percentiles
    - Detección de estados cognitivos personalizados
    - Persistencia de perfil entre sesiones
    """
    
    def __init__(self, user_id: str = "default", calibration_samples: int = 60):
        self.user_id = user_id
        self.calibration_samples = calibration_samples  # ~60 segundos @ 1Hz
        self.is_calibrated = False
        self.calibration_progress = 0.0
        
        # Historia de datos para calibración
        self.calibration_data = {
            'delta': deque(maxlen=calibration_samples),
            'theta': deque(maxlen=calibration_samples),
            'alpha': deque(maxlen=calibration_samples),
            'beta': deque(maxlen=calibration_samples),
            'gamma': deque(maxlen=calibration_samples),
            'arousal': deque(maxlen=calibration_samples),
            'valence': deque(maxlen=calibration_samples),
            'engagement': deque(maxlen=calibration_samples),
            'stress': deque(maxlen=calibration_samples)
        }
        
        # Baseline personal (media y std)
        self.baseline = {}
        
        # Thresholds personalizados (percentiles)
        self.thresholds = {}
        
        # Historia continua para adaptación
        self.continuous_history = {
            'flow_scores': deque(maxlen=300),  # 5 minutos
            'emotional_states': deque(maxlen=300)
        }
        
        # Intentar cargar perfil existente
        self.profile_path = Path(f"profiles/{user_id}_profile.json")
        self.load_profile()
    
    def add_sample(self, bands: Dict[str, float], metrics: Dict[str, float]):
        """
        Añadir muestra durante calibración
        """
        if self.is_calibrated:
            # Ya calibrado - solo actualizar historia continua
            if 'flow_score' in metrics:
                self.continuous_history['flow_scores'].append(metrics['flow_score'])
            if 'emotion' in metrics:
                self.continuous_history['emotional_states'].append(metrics['emotion'])
            return
        
        # Añadir a datos de calibración
        for band, power in bands.items():
            if band in self.calibration_data:
                self.calibration_data[band].append(power)
        
        for metric, value in metrics.items():
            if metric in self.calibration_data:
                self.calibration_data[metric].append(value)
        
        # Actualizar progreso
        samples_collected = len(self.calibration_data['alpha'])
        self.calibration_progress = min(1.0, samples_collected / self.calibration_samples)
        
        # Calibrar cuando tengamos suficientes muestras
        if samples_collected >= self.calibration_samples and not self.is_calibrated:
            self.calibrate()
    
    def calibrate(self):
        """
        Calcular baseline y thresholds personalizados
        """
        print(f"\n🎯 Calibrando perfil personal para {self.user_id}...")
        
        # 1. Calcular baseline (media y desviación estándar)
        for key, data in self.calibration_data.items():
            if len(data) > 0:
                data_array = np.array(list(data))
                self.baseline[key] = {
                    'mean': float(np.mean(data_array)),
                    'std': float(np.std(data_array)),
                    'median': float(np.median(data_array)),
                    'min': float(np.min(data_array)),
                    'max': float(np.max(data_array))
                }
        
        # 2. Calcular thresholds basados en percentiles PERSONALES
        self.thresholds = self._calculate_personal_thresholds()
        
        self.is_calibrated = True
        print(f"✅ Calibración completa!")
        print(f"\n📊 Tu perfil personal:")
        self._print_profile()
        
        # Guardar perfil
        self.save_profile()
    
    def _calculate_personal_thresholds(self) -> Dict:
        """
        Calcular thresholds basados en los percentiles de TUS datos
        No valores arbitrarios - valores basados en TU cerebro
        """
        thresholds = {}
        
        # Arousal thresholds (basados en tus percentiles)
        if 'arousal' in self.calibration_data and len(self.calibration_data['arousal']) > 0:
            arousal_data = np.array(list(self.calibration_data['arousal']))
            thresholds['arousal'] = {
                'very_low': float(np.percentile(arousal_data, 10)),
                'low': float(np.percentile(arousal_data, 30)),
                'medium': float(np.percentile(arousal_data, 50)),
                'high': float(np.percentile(arousal_data, 70)),
                'very_high': float(np.percentile(arousal_data, 90))
            }
        
        # Valence thresholds
        if 'valence' in self.calibration_data and len(self.calibration_data['valence']) > 0:
            valence_data = np.array(list(self.calibration_data['valence']))
            thresholds['valence'] = {
                'negative': float(np.percentile(valence_data, 30)),
                'neutral': float(np.percentile(valence_data, 50)),
                'positive': float(np.percentile(valence_data, 70))
            }
        
        # Engagement thresholds
        if 'engagement' in self.calibration_data and len(self.calibration_data['engagement']) > 0:
            engagement_data = np.array(list(self.calibration_data['engagement']))
            thresholds['engagement'] = {
                'low': float(np.percentile(engagement_data, 25)),
                'medium': float(np.percentile(engagement_data, 50)),
                'high': float(np.percentile(engagement_data, 75))
            }
        
        # Stress thresholds
        if 'stress' in self.calibration_data and len(self.calibration_data['stress']) > 0:
            stress_data = np.array(list(self.calibration_data['stress']))
            thresholds['stress'] = {
                'low': float(np.percentile(stress_data, 30)),
                'moderate': float(np.percentile(stress_data, 60)),
                'high': float(np.percentile(stress_data, 80))
            }
        
        # Alpha óptimo para flow (tu sweet spot personal)
        if 'alpha' in self.calibration_data and len(self.calibration_data['alpha']) > 0:
            alpha_data = np.array(list(self.calibration_data['alpha']))
            # Tu alpha óptimo es tu mediana (donde pasas más tiempo)
            thresholds['alpha_optimal'] = float(np.median(alpha_data))
            thresholds['alpha_range'] = {
                'low': float(np.percentile(alpha_data, 25)),
                'optimal': float(np.percentile(alpha_data, 50)),
                'high': float(np.percentile(alpha_data, 75))
            }
        
        # Beta/Alpha ratio para flow
        if 'beta' in self.calibration_data and 'alpha' in self.calibration_data:
            beta_data = np.array(list(self.calibration_data['beta']))
            alpha_data = np.array(list(self.calibration_data['alpha']))
            ratio_data = beta_data / (alpha_data + 0.01)
            thresholds['beta_alpha_ratio'] = {
                'relaxed': float(np.percentile(ratio_data, 25)),
                'balanced': float(np.percentile(ratio_data, 50)),
                'stressed': float(np.percentile(ratio_data, 75))
            }
        
        return thresholds
    
    def get_personalized_state(self, current_value: float, metric: str) -> str:
        """
        Clasificar estado usando TUS thresholds personalizados
        """
        if not self.is_calibrated or metric not in self.thresholds:
            return "unknown"
        
        t = self.thresholds[metric]
        
        if metric == 'arousal':
            if current_value < t['low']:
                return "very_low"
            elif current_value < t['medium']:
                return "low"
            elif current_value < t['high']:
                return "medium"
            elif current_value < t['very_high']:
                return "high"
            else:
                return "very_high"
        
        elif metric == 'valence':
            if current_value < t['negative']:
                return "negative"
            elif current_value < t['positive']:
                return "neutral"
            else:
                return "positive"
        
        elif metric == 'engagement':
            if current_value < t['low']:
                return "low"
            elif current_value < t['high']:
                return "medium"
            else:
                return "high"
        
        elif metric == 'stress':
            if current_value < t['low']:
                return "low"
            elif current_value < t['moderate']:
                return "moderate"
            else:
                return "high"
        
        return "unknown"
    
    def is_in_personal_flow(self, metrics: Dict) -> bool:
        """
        Detectar flow usando TUS thresholds personalizados
        """
        if not self.is_calibrated:
            return False
        
        # Criterios de flow personalizados
        flow_indicators = []
        
        # 1. Alpha en tu rango óptimo personal
        if 'alpha' in metrics and 'alpha_optimal' in self.thresholds:
            alpha = metrics['alpha']
            optimal = self.thresholds['alpha_optimal']
            alpha_range = self.thresholds.get('alpha_range', {})
            
            # Estás en flow si tu alpha está cerca de tu óptimo personal
            if alpha_range:
                in_optimal_range = (alpha_range['low'] <= alpha <= alpha_range['high'])
                flow_indicators.append(in_optimal_range)
        
        # 2. Beta/Alpha ratio en tu rango balanceado
        if 'beta' in metrics and 'alpha' in metrics and 'beta_alpha_ratio' in self.thresholds:
            current_ratio = metrics['beta'] / (metrics['alpha'] + 0.01)
            balanced = self.thresholds['beta_alpha_ratio']['balanced']
            relaxed = self.thresholds['beta_alpha_ratio']['relaxed']
            
            # Flow = ratio entre relajado y balanceado
            in_flow_ratio = (relaxed <= current_ratio <= balanced * 1.2)
            flow_indicators.append(in_flow_ratio)
        
        # 3. Engagement alto (para ti)
        if 'engagement' in metrics:
            engagement_state = self.get_personalized_state(metrics['engagement'], 'engagement')
            flow_indicators.append(engagement_state in ['medium', 'high'])
        
        # 4. Stress bajo (para ti)
        if 'stress_score' in metrics:
            stress_state = self.get_personalized_state(metrics['stress_score'], 'stress')
            flow_indicators.append(stress_state == 'low')
        
        # Estás en flow si cumples la mayoría de criterios
        if len(flow_indicators) >= 3:
            return sum(flow_indicators) >= 3
        elif len(flow_indicators) >= 2:
            return sum(flow_indicators) >= 2
        
        return False
    
    def get_z_score(self, current_value: float, metric: str) -> float:
        """
        Calcular z-score (desviaciones estándar desde tu baseline)
        Útil para detectar cambios significativos PARA TI
        """
        if not self.is_calibrated or metric not in self.baseline:
            return 0.0
        
        baseline = self.baseline[metric]
        mean = baseline['mean']
        std = baseline['std']
        
        if std == 0:
            return 0.0
        
        return (current_value - mean) / std
    
    def _print_profile(self):
        """Mostrar perfil personal"""
        print("\n🧠 Bandas EEG (tu baseline):")
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band in self.baseline:
                b = self.baseline[band]
                print(f"  {band:6s}: μ={b['mean']:.3f} σ={b['std']:.3f} [{b['min']:.3f}-{b['max']:.3f}]")
        
        print("\n🎯 Thresholds personalizados:")
        if 'arousal' in self.thresholds:
            t = self.thresholds['arousal']
            print(f"  Arousal: bajo<{t['low']:.2f} medio<{t['medium']:.2f} alto<{t['high']:.2f}")
        
        if 'valence' in self.thresholds:
            t = self.thresholds['valence']
            print(f"  Valence: neg<{t['negative']:.2f} neutral<{t['neutral']:.2f} pos>{t['positive']:.2f}")
        
        if 'alpha_optimal' in self.thresholds:
            print(f"  Alpha óptimo (tu flow): {self.thresholds['alpha_optimal']:.3f}")
        
        if 'beta_alpha_ratio' in self.thresholds:
            t = self.thresholds['beta_alpha_ratio']
            print(f"  β/α ratio: relajado<{t['relaxed']:.2f} balanceado~{t['balanced']:.2f} estrés>{t['stressed']:.2f}")
    
    def save_profile(self):
        """Guardar perfil para futuras sesiones"""
        self.profile_path.parent.mkdir(exist_ok=True)
        
        profile_data = {
            'user_id': self.user_id,
            'calibrated_at': time.time(),
            'baseline': self.baseline,
            'thresholds': self.thresholds,
            'calibration_samples': self.calibration_samples
        }
        
        with open(self.profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"\n💾 Perfil guardado en: {self.profile_path}")
    
    def load_profile(self) -> bool:
        """Cargar perfil existente"""
        if not self.profile_path.exists():
            print(f"ℹ️  No hay perfil previo. Iniciando calibración...")
            return False
        
        try:
            with open(self.profile_path, 'r') as f:
                profile_data = json.load(f)
            
            self.baseline = profile_data.get('baseline', {})
            self.thresholds = profile_data.get('thresholds', {})
            self.is_calibrated = True
            
            print(f"\n✅ Perfil cargado desde: {self.profile_path}")
            print(f"   Calibrado el: {time.ctime(profile_data.get('calibrated_at', 0))}")
            self._print_profile()
            
            return True
        except Exception as e:
            print(f"⚠️  Error cargando perfil: {e}")
            return False
    
    def recalibrate(self):
        """Forzar recalibración"""
        self.is_calibrated = False
        self.calibration_progress = 0.0
        for key in self.calibration_data:
            self.calibration_data[key].clear()
        print("🔄 Recalibrando...")
