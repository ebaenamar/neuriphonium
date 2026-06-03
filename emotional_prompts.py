#!/usr/bin/env python3
"""
Sistema de prompts evolutivos que construyen narrativas emocionales
Genera prompts complejos multi-capa que evolucionan con el estado mental
"""

import os
import time
import random
import asyncio
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EmotionalPromptGenerator:
    """
    Genera prompts complejos que evolucionan con el estado emocional
    
    Características:
    - Prompts multi-capa (base + emocional + transición + textura)
    - Historia emocional (memoria de estados previos)
    - Construcción progresiva (añadir/quitar elementos)
    - Coherencia narrativa musical
    """
    
    def __init__(self):
        self.emotional_history = deque(maxlen=20)
        self.current_layers = set()
        self.transition_count = 0
        self.session_start = time.time()
        
        # Género base seleccionado aleatoriamente al inicio
        self.base_genre = None
        self.genre_locked = False  # Se bloquea después de la primera generación
        
    def generate_layered_prompt(self, state: Dict) -> Dict:
        """
        Genera prompt complejo con múltiples capas
        
        Estructura:
        [Base Layer] + [Emotional Layer] + [Transition Layer] + [Texture Layer]
        """
        
        # Detectar transición emocional
        transition = self._detect_emotional_transition(state)
        
        # Construir capas
        base_layer = self._get_base_layer(state)
        emotional_layer = self._get_emotional_layer(state)
        transition_layer = self._get_transition_layer(transition)
        texture_layer = self._get_texture_layer(state)
        
        # Combinar en prompt coherente
        full_prompt = self._compose_prompt(
            base_layer, 
            emotional_layer, 
            transition_layer, 
            texture_layer
        )
        
        # Actualizar historia
        self.emotional_history.append({
            'state': state,
            'emotion': state.get('emotion', 'neutral'),
            'timestamp': time.time()
        })
        
        if transition.get('type') == 'shift':
            self.transition_count += 1
        
        return {
            'prompt': full_prompt,
            'layers': {
                'base': base_layer,
                'emotional': emotional_layer,
                'transition': transition_layer,
                'texture': texture_layer
            },
            'transition': transition,
            'emotion': state.get('emotion', 'neutral')
        }
    
    def _get_base_layer(self, state: Dict) -> str:
        """
        Base Layer - Fundamento musical con variedad aleatoria inicial
        
        Múltiples géneros por cuadrante emocional:
        - Primera vez: selecciona aleatoriamente
        - Después: evoluciona el género seleccionado según el estado
        """
        arousal = state.get('arousal', 0.5)
        valence = state.get('valence', 0.5)
        
        arousal_level = 'high' if arousal > 0.6 else ('medium' if arousal > 0.4 else 'low')
        valence_level = 'positive' if valence > 0.6 else ('neutral' if valence > 0.4 else 'negative')
        
        # Fondos dinámicos para improvisación de violín
        # Musicalmente interesantes pero sin melodías dominantes
        genre_options = {
            ('high', 'positive'): [
                "Energetic funk groove with syncopated bass, rhythm guitar stabs, and bright horn hits",
                "Upbeat afrobeat with polyrhythmic percussion, kalimba patterns, and driving bass",
                "Vibrant electronic dance with pulsing synth bass, filtered chords, and crisp hi-hats",
                "Joyful gypsy jazz rhythm with acoustic guitar comping and upright bass walking",
                "Euphoric trance with arpeggiated synths, rolling bassline, and atmospheric pads"
            ],
            ('high', 'neutral'): [
                "Driving techno with hypnotic bass pattern, evolving synth textures, and steady kick",
                "Intense drum and bass with rapid breakbeats, deep sub-bass, and atmospheric pads",
                "Progressive house with building synth layers, filtered chords, and rhythmic pulse",
                "Post-rock with layered guitars, dynamic drums, and crescendoing bass",
                "Rhythmic minimal with percussive elements, bass groove, and subtle harmonic shifts"
            ],
            ('high', 'negative'): [
                "Dark industrial with pounding drums, distorted bass, and ominous synth swells",
                "Intense orchestral with dramatic string sections, timpani rolls, and brass stabs",
                "Heavy psytrance with driving bassline, dark pads, and tribal percussion",
                "Aggressive dubstep with wobble bass, sharp snares, and tense build-ups",
                "Chaotic breakcore with frenetic drums, distorted bass, and dissonant synths"
            ],
            ('medium', 'positive'): [
                "Smooth bossa nova with nylon guitar, gentle percussion, and warm upright bass",
                "Mellow jazz with piano comping, brushed drums, and walking bass",
                "Uplifting neo-soul with Rhodes piano, subtle drums, and groovy bass",
                "Cheerful indie folk with fingerpicked guitar, light percussion, and acoustic bass",
                "Sunny reggae with organ skanks, one-drop drums, and deep bass"
            ],
            ('medium', 'neutral'): [
                "Downtempo trip-hop with dusty beats, deep bass, and ethereal pads",
                "Ambient electronic with evolving textures, subtle pulse, and harmonic drones",
                "Lo-fi hip-hop with vinyl crackle, mellow bass, and warm chord samples",
                "Post-classical with piano, cello harmonies, and atmospheric strings",
                "World fusion with hand drums, acoustic bass, and modal guitar"
            ],
            ('medium', 'negative'): [
                "Melancholic chamber music with cello, viola, and piano in minor keys",
                "Moody slowcore with sparse guitar, heavy bass, and minimal drums",
                "Dark jazz with muted trumpet, walking bass, and somber piano chords",
                "Brooding post-punk with angular bass, atmospheric guitar, and steady drums",
                "Wistful folk with fingerpicked guitar, cello drones, and subtle percussion"
            ],
            ('low', 'positive'): [
                "Peaceful classical guitar with gentle arpeggios and warm bass notes",
                "Serene ambient with flowing synth pads, soft chimes, and nature sounds",
                "Tranquil new age with harp glissandos, flute whispers, and gentle strings",
                "Soothing chillout with soft Rhodes, mellow bass, and relaxed beats",
                "Gentle bossa with quiet nylon guitar, brush drums, and warm bass"
            ],
            ('low', 'neutral'): [
                "Minimal ambient with subtle drones, field recordings, and sparse piano",
                "Meditative soundscape with singing bowls, deep bass, and long reverbs",
                "Contemplative modern classical with prepared piano and string harmonics",
                "Sparse electroacoustic with granular textures and delicate tones",
                "Quiet experimental with tape loops, resonant objects, and silence"
            ],
            ('low', 'negative'): [
                "Dark ambient with deep bass drones, eerie pads, and distant rumbles",
                "Haunting doom with slow heavy riffs, crushing bass, and sparse drums",
                "Desolate soundscape with industrial noise, low frequencies, and decay",
                "Bleak drone with dissonant harmonics, sub-bass, and harsh textures",
                "Ominous dark jazz with contrabass, muted piano, and brushed cymbals"
            ]
        }
        
        quadrant = (arousal_level, valence_level)
        
        # Primera vez: seleccionar género aleatorio
        if not self.genre_locked:
            options = genre_options.get(quadrant, ["Balanced instrumental music"])
            self.base_genre = random.choice(options)
            self.genre_locked = True
            print(f"🎲 Género inicial seleccionado: {self.base_genre[:50]}...")
        
        # Después: evolucionar el género según el nuevo cuadrante
        else:
            # Si cambiamos de cuadrante, adaptar el género gradualmente
            current_options = genre_options.get(quadrant, ["Balanced instrumental music"])
            
            # Buscar si hay un género similar en el nuevo cuadrante
            # (mantener coherencia pero adaptarse al nuevo estado)
            if self.base_genre not in current_options:
                # Transición: mencionar evolución
                new_genre = random.choice(current_options)
                print(f"🔄 Género evolucionando hacia: {new_genre[:50]}...")
                self.base_genre = new_genre
        
        return self.base_genre
    
    def _get_emotional_layer(self, state: Dict) -> str:
        """
        Emotional Layer - Matices emocionales con vocabulario rico
        """
        emotion = state.get('emotion', 'neutral')
        flow_score = state.get('flow_score', 0.5)
        stress = state.get('stress_score', 0.5)
        
        # Vocabulario emocional rico por estado
        emotional_descriptors = {
            'excited': [
                "euphoric and exhilarating",
                "vibrant with infectious energy",
                "bright, sparkling, and celebratory",
                "dynamic with soaring melodies"
            ],
            'alert': [
                "focused and purposeful",
                "crisp with clear intention",
                "sharp and attentive",
                "precise with controlled energy"
            ],
            'tense': [
                "anxious with building tension",
                "restless and uneasy",
                "edgy with dissonant undertones",
                "nervous energy with irregular rhythms"
            ],
            'pleased': [
                "content and satisfied",
                "warm with gentle optimism",
                "pleasant and uplifting",
                "cheerful with light textures"
            ],
            'neutral': [
                "balanced and centered",
                "steady with even flow",
                "contemplative and observant",
                "measured and thoughtful"
            ],
            'frustrated': [
                "agitated with unresolved tension",
                "conflicted with contrasting elements",
                "turbulent and searching",
                "restless with shifting patterns"
            ],
            'calm': [
                "serene and peaceful",
                "tranquil with flowing harmonies",
                "soothing and meditative",
                "gentle with spacious atmosphere"
            ],
            'relaxed': [
                "laid-back and easy-going",
                "comfortable with smooth grooves",
                "mellow and unhurried",
                "soft with warm tones"
            ],
            'sad': [
                "melancholic and introspective",
                "somber with deep emotion",
                "wistful and reflective",
                "tender with aching beauty"
            ]
        }
        
        # Seleccionar descriptor (rotar para variedad)
        descriptors = emotional_descriptors.get(emotion, ["balanced and neutral"])
        descriptor = descriptors[self.transition_count % len(descriptors)]
        
        # Añadir modificador de flow si está alto
        if flow_score > 0.65:
            descriptor += ", flowing effortlessly with immersive quality"
        
        # Añadir modificador de estrés si está alto
        if stress > 0.7:
            descriptor += ", with underlying tension"
        
        return f"The mood is {descriptor}"
    
    def _detect_emotional_transition(self, current_state: Dict) -> Dict:
        """
        Detecta transiciones emocionales comparando con historia
        """
        if len(self.emotional_history) < 3:
            return {'type': 'stable', 'direction': None}
        
        prev_states = list(self.emotional_history)[-3:]
        prev_emotion = prev_states[-1]['emotion']
        current_emotion = current_state.get('emotion', 'neutral')
        
        # Detectar cambio emocional
        if prev_emotion != current_emotion:
            return {
                'type': 'shift',
                'from': prev_emotion,
                'to': current_emotion,
                'speed': 'gradual'
            }
        
        # Detectar intensificación
        prev_arousal = prev_states[-1]['state'].get('arousal', 0.5)
        current_arousal = current_state.get('arousal', 0.5)
        arousal_delta = current_arousal - prev_arousal
        
        if abs(arousal_delta) > 0.15:
            return {
                'type': 'intensify' if arousal_delta > 0 else 'diminish',
                'magnitude': abs(arousal_delta)
            }
        
        return {'type': 'stable', 'direction': None}
    
    def _get_transition_layer(self, transition: Dict) -> str:
        """
        Transition Layer - Describe cómo está evolucionando la música
        """
        trans_type = transition.get('type')
        
        if trans_type == 'stable':
            return "maintaining consistent energy and mood"
        
        elif trans_type == 'shift':
            from_emotion = transition.get('from', '')
            to_emotion = transition.get('to', '')
            return f"gradually transitioning with smooth crossfade, evolving from {from_emotion} to {to_emotion} qualities"
        
        elif trans_type == 'intensify':
            return "building in intensity with added layers and rising energy"
        
        elif trans_type == 'diminish':
            return "settling down with elements gradually fading and softening"
        
        return ""
    
    def _get_texture_layer(self, state: Dict) -> str:
        """
        Texture Layer - Instrumentación dinámica basada en bandas EEG
        """
        bands = state.get('bands', {})
        flow_score = state.get('flow_score', 0.5)
        valence = state.get('valence', 0.5)
        
        instruments = []
        
        # Añadir instrumentos según bandas dominantes
        if bands.get('delta', 0) > 0.3:
            instruments.append("deep sub-bass")
        
        if bands.get('theta', 0) > 0.25:
            instruments.append("warm synth pads")
        
        if bands.get('alpha', 0) > 0.25:
            if valence > 0.5:
                instruments.append("acoustic guitar")
            else:
                instruments.append("electric piano")
        
        if bands.get('beta', 0) > 0.3:
            instruments.append("crisp percussion")
        
        if bands.get('gamma', 0) > 0.2:
            instruments.append("bright bells and chimes")
        
        # Añadir capas según flow
        if flow_score > 0.65:
            instruments.append("layered strings creating depth")
        
        # Construir frase
        if len(instruments) == 0:
            return "Featuring minimal instrumentation"
        elif len(instruments) == 1:
            return f"Featuring {instruments[0]}"
        else:
            instruments_str = ", ".join(instruments[:-1]) + f", and {instruments[-1]}"
            return f"Featuring {instruments_str}"
    
    def _compose_prompt(self, base: str, emotional: str, transition: str, texture: str) -> str:
        """
        Combina todas las capas en un prompt coherente y natural
        
        Estructura:
        [Base]. [Emotional]. [Texture]. [Transition]. [Production]. Instrumental.
        """
        prompt_parts = []
        
        # 1. Base style
        prompt_parts.append(base)
        
        # 2. Emotional layer
        prompt_parts.append(emotional)
        
        # 3. Instrumentation
        prompt_parts.append(texture)
        
        # 4. Transition (si hay)
        if transition:
            prompt_parts.append(transition)
        
        # 5. Production quality
        prompt_parts.append("High-quality stereo production with spatial depth and clarity")
        
        # 6. Instrumental
        prompt_parts.append("Instrumental")
        
        # Unir con puntos
        full_prompt = ". ".join(prompt_parts) + "."
        
        return full_prompt


class FlowInducer:
    """
    Sistema para inducir Flow State mediante feedback musical adaptativo
    
    Estrategia:
    1. Detectar estado actual
    2. Ajustar música para guiar hacia flow
    3. Mantener en flow una vez alcanzado
    """
    
    def __init__(self):
        self.target_flow = 0.7
        self.in_flow_threshold = 0.65
        self.flow_maintenance_mode = False
        
    def get_music_adjustments(self, state: Dict) -> Dict[str, any]:
        """
        Ajustar parámetros musicales para inducir/mantener flow
        """
        flow_score = state.get('flow_score', 0.5)
        stress = state.get('stress_score', 0.5)
        engagement = state.get('engagement', 0.5)
        
        adjustments = {}
        
        # Modo: Inducción vs Mantenimiento
        if flow_score > self.in_flow_threshold:
            # MANTENER FLOW
            self.flow_maintenance_mode = True
            adjustments['strategy'] = 'maintain_flow'
            adjustments['tempo_multiplier'] = 1.0
            adjustments['complexity_change'] = 0.0
            adjustments['smoothing'] = 15
            
        else:
            # INDUCIR FLOW
            self.flow_maintenance_mode = False
            
            # Si hay estrés alto: primero relajar
            if stress > 0.6:
                adjustments['strategy'] = 'reduce_stress'
                adjustments['tempo_multiplier'] = 0.8
                adjustments['complexity_change'] = -0.2
                adjustments['smoothing'] = 12
                
            # Si engagement bajo: activar
            elif engagement < 0.3:
                adjustments['strategy'] = 'increase_engagement'
                adjustments['tempo_multiplier'] = 1.2
                adjustments['complexity_change'] = 0.2
                adjustments['smoothing'] = 6
                
            # Si cerca de flow: ajuste fino
            elif flow_score > 0.5:
                adjustments['strategy'] = 'approach_flow'
                flow_gap = self.target_flow - flow_score
                adjustments['tempo_multiplier'] = 1.0 + (flow_gap * 0.1)
                adjustments['complexity_change'] = 0.0
                adjustments['smoothing'] = 10
                
            else:
                adjustments['strategy'] = 'balance'
                adjustments['tempo_multiplier'] = 1.0
                adjustments['complexity_change'] = 0.0
                adjustments['smoothing'] = 8
        
        return adjustments
    
    def generate_flow_optimized_prompt(self, state: Dict, adjustments: Dict) -> str:
        """
        Generar prompt de Lyria optimizado para inducir flow
        """
        strategy = adjustments.get('strategy', 'balance')
        
        prompts = {
            'maintain_flow': (
                "Smooth, flowing instrumental music with consistent rhythm and gradual evolution. "
                "Steady tempo with subtle variations. Layered textures that maintain engagement "
                "without distraction. Balanced mix with clear melodic focus. High-quality production. Instrumental."
            ),
            'reduce_stress': (
                "Calm, soothing ambient music with gentle pads and soft acoustic instruments. "
                "Slow, relaxed tempo with spacious arrangement. Warm tones and peaceful atmosphere. "
                "Minimal percussion, emphasis on sustained notes and flowing melodies. Instrumental."
            ),
            'increase_engagement': (
                "Energetic, rhythmic music with clear pulse and engaging melodic patterns. "
                "Moderate to fast tempo with driving beat. Dynamic arrangement with interesting "
                "harmonic progressions. Bright tones and crisp production. Instrumental."
            ),
            'approach_flow': (
                "Balanced instrumental music with steady groove and evolving textures. "
                "Moderate tempo with consistent energy. Clear structure with smooth transitions. "
                "Engaging but not overwhelming. Clean, focused production. Instrumental."
            ),
            'balance': (
                "Versatile instrumental music with moderate tempo and balanced dynamics. "
                "Mix of rhythmic and melodic elements. Neutral emotional tone with room for variation. Instrumental."
            )
        }
        
        return prompts.get(strategy, prompts['balance'])


class GeminiPromptEvolver:
    """
    Usa un LLM (OpenRouter o Gemini) para evolucionar el prompt musical en tiempo real.
    
    Corre en background (async) para no bloquear el audio.
    El sistema usa el último prompt generado mientras el LLM prepara el siguiente.
    """

    SYSTEM_PROMPT = """You are a musical director creating real-time backing music for violin improvisation.
Your job: given a musician's live EEG cognitive state, generate a SHORT musical description (2-3 sentences max)
that describes a background soundscape the violinist can improvise over.

Rules:
- NO lead melodies or dominant melodic lines — the violin is the lead
- Focus on: harmony, rhythm, texture, atmosphere, bass
- Be specific about genre, instruments, feel
- Evolve naturally from the previous prompt (don't jump abruptly)
- Output ONLY the musical description, no explanations

Example output:
"Modal jazz with sparse piano comping in Dorian, walking upright bass at 70 BPM. Brushed snare gives a loose, breathing pulse. Dark and introspective, harmonic space open for violin to breathe."
"""

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL   = "openai/gpt-4o-mini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Preferir OpenRouter; si no, Gemini nativo
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.use_openrouter = bool(os.getenv("OPENROUTER_API_KEY") or api_key)
        self.model = model or self.DEFAULT_MODEL
        self._client = None
        self._current_prompt: Optional[str] = None
        self._pending: bool = False
        self._last_call: float = 0
        self._min_interval: float = 12.0  # Mínimo segundos entre llamadas
        self._history: deque = deque(maxlen=5)
        self._enabled: bool = bool(self.api_key)

        if not self._enabled:
            logger.warning("⚠️  GeminiPromptEvolver: sin API key, usando fallback estático")
        else:
            backend = "OpenRouter" if self.use_openrouter else "Gemini"
            logger.info(f"🤖 GeminiPromptEvolver listo — backend: {backend}, modelo: {self.model}")

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                if self.use_openrouter:
                    self._client = OpenAI(
                        base_url=self.OPENROUTER_BASE,
                        api_key=self.api_key,
                    )
                else:
                    # Gemini via openai-compatible endpoint
                    self._client = OpenAI(
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        api_key=self.api_key,
                    )
            except ImportError:
                logger.error("openai package no instalado: pip install openai")
                self._enabled = False
        return self._client

    def _build_user_message(self, state: Dict) -> str:
        arousal   = state.get('arousal', 0.5)
        valence   = state.get('valence', 0.5)
        flow      = state.get('flow_score', 0.5)
        stress    = state.get('stress_score', 0.3)
        emotion   = state.get('emotion', 'neutral')
        bands     = state.get('bands', {})
        theta     = bands.get('theta', 0.25)
        alpha     = bands.get('alpha', 0.1)
        beta      = bands.get('beta', 0.1)
        gamma     = bands.get('gamma', 0.05)

        prev = f'\nPrevious prompt: "{self._current_prompt}"' if self._current_prompt else ""
        history_str = ""
        if len(self._history) > 1:
            history_str = f"\nRecent evolution: {' → '.join(list(self._history)[-3:])}"

        return (
            f"EEG state snapshot:\n"
            f"  arousal={arousal:.2f}  valence={valence:.2f}  flow={flow:.2f}  stress={stress:.2f}\n"
            f"  emotion={emotion}\n"
            f"  theta={theta:.3f}  alpha={alpha:.3f}  beta={beta:.3f}  gamma={gamma:.3f}\n"
            f"{prev}{history_str}\n\n"
            f"Generate the next evolved musical background for violin improvisation."
        )

    async def evolve_async(self, state: Dict) -> Optional[str]:
        """
        Llama a Gemini async. Devuelve el nuevo prompt o None si no toca llamar todavía.
        """
        if not self._enabled:
            return None

        now = time.time()
        if self._pending or (now - self._last_call) < self._min_interval:
            return None  # Todavía no toca / ya hay una llamada en curso

        self._pending = True
        self._last_call = now

        try:
            client = self._get_client()
            if not client:
                return None

            user_msg = self._build_user_message(state)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=120,
                    temperature=0.85,
                )
            )

            msg = response.choices[0].message
            new_prompt = (msg.content or "").strip()
            if not new_prompt:
                logger.warning(f"⚠️  LLM devolvió content vacío (modelo: {response.model})")
                return None
            if new_prompt:
                self._history.append(new_prompt[:60] + "...")
                self._current_prompt = new_prompt
                logger.info(f"🤖 Gemini → {new_prompt[:80]}...")
                return new_prompt

        except Exception as e:
            logger.warning(f"⚠️  GeminiPromptEvolver error: {e}")
        finally:
            self._pending = False

        return None

    @property
    def current_prompt(self) -> Optional[str]:
        return self._current_prompt

    @property
    def is_ready(self) -> bool:
        return self._current_prompt is not None

    def set_interval(self, seconds: float):
        """Ajustar frecuencia de llamadas a Gemini (mínimo 8s)"""
        self._min_interval = max(8.0, seconds)
