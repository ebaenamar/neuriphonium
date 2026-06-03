#!/usr/bin/env python3
"""
Métricas neurocientíficas para detección de Flow State y estados emocionales
Basado en investigación de neurociencia cognitiva y afectiva

Referencias:
- Flow State: Ulrich et al. (2016), Katahira et al. (2018)
- Emociones: Russell's Circumplex Model (1980)
- Engagement: Pope et al. (1995) - Beta/Alpha+Theta ratio
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
from personal_calibration import PersonalCalibration


class NeuroMetrics:
    """
    Calculador de métricas neurocientíficas validadas
    """
    
    def __init__(self, history_length: int = 10, user_id: str = "default", use_personal_calibration: bool = True):
        self.history_length = history_length
        self.band_history = {
            'delta': deque(maxlen=history_length),
            'theta': deque(maxlen=history_length),
            'alpha': deque(maxlen=history_length),
            'beta': deque(maxlen=history_length),
            'gamma': deque(maxlen=history_length)
        }
        self.flow_history = deque(maxlen=30)
        self.baseline = None
        
        # Sistema de calibración personalizada
        self.use_personal_calibration = use_personal_calibration
        self.personal_cal = PersonalCalibration(user_id=user_id) if use_personal_calibration else None
        
    def update_bands(self, bands: Dict[str, float]):
        """Actualizar historia de bandas"""
        for band, power in bands.items():
            if band in self.band_history:
                self.band_history[band].append(power)
    
    def calculate_flow_state(self, bands: Dict[str, float]) -> Dict[str, float]:
        """
        Detectar estado de Flow basado en marcadores EEG validados
        
        Flow State Markers:
        1. Frontal Theta (Fm-theta): Atención sostenida
        2. Alpha moderado: Relajación alerta (no somnolencia)
        3. Beta frontal reducido: Menos autocrítica/monitoreo
        4. Gamma elevado: Integración cognitiva
        5. Alpha/Beta ratio alto: Balance esfuerzo-facilidad
        
        Returns:
            dict con flow_score (0-1) y componentes
        """
        theta = bands.get('theta', 0.2)
        alpha = bands.get('alpha', 0.2)
        beta = bands.get('beta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        # 1. Frontal Theta (proxy: theta general elevado)
        theta_component = np.clip(theta * 2.0, 0, 1)
        
        # 2. Alpha óptimo (sweet spot: 0.25-0.35)
        alpha_optimal = 1.0 - abs(alpha - 0.30) / 0.30
        alpha_component = np.clip(alpha_optimal, 0, 1)
        
        # 3. Beta reducido (menos autocrítica)
        beta_component = np.clip(1.0 - beta * 1.5, 0, 1)
        
        # 4. Gamma elevado (integración cognitiva)
        gamma_component = np.clip(gamma * 2.5, 0, 1)
        
        # 5. Alpha/Beta ratio (balance)
        alpha_beta_ratio = alpha / (beta + 0.01)
        ratio_component = np.clip(alpha_beta_ratio / 2.0, 0, 1)
        
        # Flow score ponderado
        flow_score = (
            0.25 * theta_component +
            0.20 * alpha_component +
            0.20 * beta_component +
            0.20 * gamma_component +
            0.15 * ratio_component
        )
        
        self.flow_history.append(flow_score)
        
        # Estabilidad del flow
        flow_stability = 1.0 - np.std(list(self.flow_history)) if len(self.flow_history) > 5 else 0.5
        
        return {
            'flow_score': flow_score,
            'flow_stability': flow_stability,
            'theta_attention': theta_component,
            'alpha_relaxation': alpha_component,
            'beta_reduced': beta_component,
            'gamma_integration': gamma_component,
            'alpha_beta_ratio': alpha_beta_ratio,
            'in_flow': flow_score > 0.65 and flow_stability > 0.7
        }
    
    def calculate_emotional_state(self, bands: Dict[str, float]) -> Dict[str, float]:
        """
        Modelo Circumplex de Russell para estados emocionales
        
        Arousal (Activación): Delta bajo, Beta/Gamma alto
        Valence (Valencia): Alpha alto = positivo
        """
        delta = bands.get('delta', 0.2)
        theta = bands.get('theta', 0.2)
        alpha = bands.get('alpha', 0.2)
        beta = bands.get('beta', 0.2)
        gamma = bands.get('gamma', 0.2)
        
        # Arousal (Activación)
        arousal = np.clip(
            0.4 * beta + 0.4 * gamma - 0.2 * delta,
            0, 1
        )
        
        # Valence (Valencia emocional)
        valence = np.clip(
            0.6 * alpha - 0.3 * theta + 0.1 * gamma + 0.3,
            0, 1
        )
        
        emotion = self._classify_emotion(arousal, valence)
        
        return {
            'arousal': arousal,
            'valence': valence,
            'emotion': emotion,
            'arousal_label': self._arousal_label(arousal),
            'valence_label': self._valence_label(valence)
        }
    
    def calculate_cognitive_load(self, bands: Dict[str, float]) -> Dict[str, float]:
        """
        Engagement Index (Pope et al., 1995)
        Engagement = Beta / (Alpha + Theta)
        """
        theta = bands.get('theta', 0.2)
        alpha = bands.get('alpha', 0.2)
        beta = bands.get('beta', 0.2)
        
        engagement = beta / (alpha + theta + 0.01)
        engagement_normalized = np.clip(engagement / 3.0, 0, 1)
        
        workload = np.clip(beta * 2.0 - theta * 0.5, 0, 1)
        
        return {
            'engagement': engagement_normalized,
            'workload': workload,
            'engagement_raw': engagement,
            'cognitive_state': self._classify_cognitive_state(engagement_normalized, workload)
        }
    
    def calculate_stress_markers(self, bands: Dict[str, float]) -> Dict[str, float]:
        """
        Marcadores de estrés y ansiedad
        Beta/Alpha ratio alto = estrés
        """
        alpha = bands.get('alpha', 0.2)
        beta = bands.get('beta', 0.2)
        
        stress_ratio = beta / (alpha + 0.01)
        stress_score = np.clip(stress_ratio / 4.0, 0, 1)
        relaxation = 1.0 - stress_score
        
        return {
            'stress_score': stress_score,
            'relaxation': relaxation,
            'beta_alpha_ratio': stress_ratio,
            'stress_level': self._classify_stress(stress_score)
        }
    
    def get_comprehensive_state(self, bands: Dict[str, float]) -> Dict:
        """Análisis completo del estado mental con calibración personalizada"""
        self.update_bands(bands)
        
        flow = self.calculate_flow_state(bands)
        emotion = self.calculate_emotional_state(bands)
        cognitive = self.calculate_cognitive_load(bands)
        stress = self.calculate_stress_markers(bands)
        
        result = {
            **flow,
            **emotion,
            **cognitive,
            **stress,
            'bands': bands
        }
        
        # Añadir calibración personalizada
        if self.use_personal_calibration and self.personal_cal:
            # Añadir muestra para calibración
            self.personal_cal.add_sample(bands, result)
            
            # Si ya está calibrado, añadir métricas personalizadas
            if self.personal_cal.is_calibrated:
                result['personal_arousal_state'] = self.personal_cal.get_personalized_state(
                    result['arousal'], 'arousal'
                )
                result['personal_valence_state'] = self.personal_cal.get_personalized_state(
                    result['valence'], 'valence'
                )
                result['personal_engagement_state'] = self.personal_cal.get_personalized_state(
                    result['engagement'], 'engagement'
                )
                result['personal_stress_state'] = self.personal_cal.get_personalized_state(
                    result['stress_score'], 'stress'
                )
                result['in_personal_flow'] = self.personal_cal.is_in_personal_flow(result)
                
                # Z-scores (desviaciones desde TU baseline)
                result['arousal_z'] = self.personal_cal.get_z_score(result['arousal'], 'arousal')
                result['alpha_z'] = self.personal_cal.get_z_score(bands.get('alpha', 0), 'alpha')
                
            result['calibration_progress'] = self.personal_cal.calibration_progress
            result['is_calibrated'] = self.personal_cal.is_calibrated
        
        return result
    
    # Helper methods
    
    def _classify_emotion(self, arousal: float, valence: float) -> str:
        """Clasificar emoción en modelo circumplex"""
        if arousal > 0.6:
            if valence > 0.6:
                return "excited"
            elif valence > 0.4:
                return "alert"
            else:
                return "tense"
        elif arousal > 0.4:
            if valence > 0.6:
                return "pleased"
            elif valence > 0.4:
                return "neutral"
            else:
                return "frustrated"
        else:
            if valence > 0.6:
                return "calm"
            elif valence > 0.4:
                return "relaxed"
            else:
                return "sad"
    
    def _arousal_label(self, arousal: float) -> str:
        if arousal > 0.7:
            return "very_high"
        elif arousal > 0.5:
            return "high"
        elif arousal > 0.3:
            return "medium"
        else:
            return "low"
    
    def _valence_label(self, valence: float) -> str:
        if valence > 0.6:
            return "positive"
        elif valence > 0.4:
            return "neutral"
        else:
            return "negative"
    
    def _classify_cognitive_state(self, engagement: float, workload: float) -> str:
        if engagement > 0.7 and workload > 0.7:
            return "high_load"
        elif engagement > 0.5 and workload < 0.4:
            return "focused"
        elif engagement < 0.3:
            return "mind_wandering"
        else:
            return "moderate"
    
    def _classify_stress(self, stress_score: float) -> str:
        if stress_score > 0.7:
            return "high"
        elif stress_score > 0.4:
            return "moderate"
        else:
            return "low"
