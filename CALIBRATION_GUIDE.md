# 🎯 Guía de Calibración Personalizada

## ¿Por qué calibración personalizada?

**Cada cerebro es único.** Los thresholds genéricos no funcionan para todos:

- Tu "arousal alto" puede ser diferente al mío
- Tu alpha óptimo para flow es personal
- Tus patrones de estrés son únicos
- Tu baseline EEG es individual

## 🧠 Cómo Funciona

### Fase 1: Calibración Inicial (60 segundos)

Cuando inicias por primera vez, el sistema:

1. **Recopila 60 muestras** de tus ondas cerebrales (~1 minuto)
2. **Calcula TU baseline personal**:
   - Media y desviación estándar de cada banda EEG
   - Tus rangos naturales de arousal, valence, engagement
3. **Determina TUS thresholds** usando percentiles:
   - No valores arbitrarios (0.6, 0.7, etc.)
   - Valores basados en TUS datos reales

**Durante calibración:**
- Relájate y respira normalmente
- No intentes forzar ningún estado mental
- Deja que el sistema aprenda tu "estado neutral"

### Fase 2: Uso Personalizado

Una vez calibrado, el sistema:

- ✅ Detecta estados usando **TUS thresholds**
- ✅ Compara con **TU baseline**
- ✅ Calcula z-scores (desviaciones desde TU normal)
- ✅ Detecta flow basado en **TUS patrones**

## 📊 Métricas Personalizadas

### Estados Personalizados

```python
# Genérico (para todos)
arousal > 0.6  # ¿Alto para quién?

# Personalizado (para TI)
arousal > tu_percentil_70  # Alto para TI específicamente
```

### Ejemplo Real

**Usuario A (persona calmada):**
```
Arousal baseline: 0.35
Arousal "alto": > 0.50
Alpha óptimo: 0.32
```

**Usuario B (persona energética):**
```
Arousal baseline: 0.65
Arousal "alto": > 0.80
Alpha óptimo: 0.28
```

**Mismo valor absoluto, diferente significado:**
- Arousal = 0.60
  - Para Usuario A: **MUY ALTO** (z-score = +2.5)
  - Para Usuario B: **NORMAL** (z-score = -0.5)

## 🎯 Detección de Flow Personalizada

### Criterios Genéricos (antes)
```python
flow = (
    alpha == 0.30 AND  # ¿Por qué 0.30?
    flow_score > 0.65 AND  # ¿Por qué 0.65?
    stability > 0.70  # ¿Por qué 0.70?
)
```

### Criterios Personalizados (ahora)
```python
flow = (
    alpha in TU_RANGO_OPTIMO AND  # Tu sweet spot personal
    beta/alpha ~ TU_RATIO_BALANCEADO AND  # Tu balance único
    engagement in ['medium', 'high'] PARA_TI AND  # Alto para ti
    stress == 'low' PARA_TI  # Bajo para ti
)
```

## 💾 Persistencia de Perfil

Tu perfil se guarda en:
```
profiles/[user_id]_profile.json
```

Contiene:
- Tu baseline de cada banda EEG
- Tus thresholds personalizados
- Fecha de calibración

**Próxima sesión:** El sistema carga tu perfil automáticamente.

## 🔄 Recalibración

**¿Cuándo recalibrar?**

- Después de cambios significativos (medicación, sueño, etc.)
- Si sientes que el sistema no te representa
- Cada 1-2 semanas para ajuste fino

**Cómo recalibrar:**
```python
# En el código
neuro_metrics.personal_cal.recalibrate()

# O borra tu perfil
rm profiles/[user_id]_profile.json
```

## 📈 Ventajas del Sistema

### 1. **Precisión Individual**
- Thresholds basados en TUS datos
- No promedios poblacionales

### 2. **Adaptación Continua**
- Aprende de tus patrones
- Se ajusta a cambios graduales

### 3. **Detección Mejorada**
- Flow detection más precisa
- Estados emocionales más exactos

### 4. **Z-Scores**
```python
arousal_z = +2.5  # Estás 2.5 desviaciones sobre TU normal
alpha_z = -1.2    # Estás 1.2 desviaciones bajo TU normal
```

Útil para detectar:
- Cambios significativos PARA TI
- Anomalías en TUS patrones
- Progreso en entrenamiento mental

## 🎮 Uso en Neuriphonium

El sistema usa tu calibración para:

1. **Prompts más precisos**
   - "Estás más energético de lo normal PARA TI"
   - No "estás energético" genérico

2. **Flow induction personalizada**
   - Guía hacia TU estado de flow
   - No un estado de flow genérico

3. **Música adaptada a TI**
   - Responde a TUS cambios
   - No cambios absolutos

## 🔬 Base Científica

**Diferencias individuales en EEG:**
- Klimesch et al. (1999): "Individual alpha frequency"
- Bazanova & Vernon (2014): "Alpha power variability"
- Ros et al. (2013): "Neurofeedback personalization"

**Conclusión:** Los thresholds deben ser individualizados para precisión óptima.

## 📝 Ejemplo de Salida

```json
{
  "arousal": 0.58,
  "personal_arousal_state": "high",  // Alto PARA TI
  "arousal_z": 1.8,  // 1.8 desviaciones sobre TU normal
  
  "flow_score": 0.62,
  "in_flow": false,  // Genérico
  "in_personal_flow": true,  // Personalizado - ¡Estás en flow PARA TI!
  
  "calibration_progress": 1.0,
  "is_calibrated": true
}
```

## 🎯 Próximos Pasos

1. **Inicia sesión** - El sistema calibra automáticamente
2. **Espera 60 segundos** - Mantente relajado
3. **Perfil guardado** - Listo para futuras sesiones
4. **Disfruta** - Música adaptada a TU cerebro único

---

**Tu cerebro es único. Tu música también debería serlo.** 🧠🎵
