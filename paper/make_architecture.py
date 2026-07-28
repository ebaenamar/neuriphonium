"""Generate a clean, minimal architecture diagram for the Neuriphonium paper.

Design: vertical pipeline with 5 clear stages, left-side detail annotations,
and a separate validation branch. No nested boxes — everything is a flat,
left-to-right or top-to-bottom flow.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(12, 14), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')

# ── Colors (clean, high-contrast on white) ─────────────────────────────
C_BG     = '#ffffff'
C_EEG    = '#2ecc71'
C_PROC   = '#3498db'
C_STYLE  = '#e74c3c'
C_GEN    = '#f39c12'
C_WEB    = '#9b59b6'
C_VAL    = '#1abc9c'
C_TEXT   = '#2c3e50'
C_SUB    = '#7f8c8d'
C_ARROW  = '#bdc3c7'

fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── Helpers ────────────────────────────────────────────────────────────
def stage(x, y, w, h, num, title, color, details):
    """Draw a numbered pipeline stage with title and detail lines."""
    # Main box
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                       facecolor=color, edgecolor='white', alpha=0.15, zorder=2)
    ax.add_patch(b)
    # Number circle
    circle = plt.Circle((x + 0.45, y + h - 0.45), 0.28,
                        facecolor=color, edgecolor='white', zorder=3)
    ax.add_patch(circle)
    ax.text(x + 0.45, y + h - 0.45, str(num), ha='center', va='center',
            fontsize=11, color='white', fontweight='bold', zorder=4)
    # Title
    ax.text(x + 0.95, y + h - 0.45, title, ha='left', va='center',
            fontsize=11, color=color, fontweight='bold', zorder=4)
    # Detail lines
    for i, line in enumerate(details):
        ax.text(x + 0.95, y + h - 0.9 - i * 0.32, line, ha='left', va='center',
                fontsize=8, color=C_SUB, zorder=4)

def arrow_v(x, y1, y2, color=C_ARROW, lw=2):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle='->', color=color,
                        lw=lw, zorder=1, mutation_scale=18)
    ax.add_patch(a)

def arrow_h(x1, x2, y, color=C_ARROW, lw=2):
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle='->', color=color,
                        lw=lw, zorder=1, mutation_scale=18)
    ax.add_patch(a)

def side_label(x, y, text, color=C_SUB, fs=8):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=color, style='italic', zorder=5)

# ── Title ──────────────────────────────────────────────────────────────
ax.text(6, 13.6, 'Neuriphonium System Architecture',
        ha='center', va='center', fontsize=15, color=C_TEXT, fontweight='bold')
ax.text(6, 13.25, 'Real-time EEG-to-music pipeline with offline validation',
        ha='center', va='center', fontsize=9.5, color=C_SUB, style='italic')

# ── Main pipeline (vertical, centered) ─────────────────────────────────
PX = 3.0   # pipeline x
PW = 6.0   # pipeline width

# Stage 1: EEG Acquisition
stage(PX, 11.5, PW, 1.4, 1, 'EEG Acquisition', C_EEG, [
    'Muse headband: TP9, AF7, AF8, TP10 (256 Hz)',
    'LSL streaming via muselsl',
    '256-sample windows (1s), 50% overlap',
    'Auto-reconnect on stream loss',
])
arrow_v(PX + PW/2, 11.5, 10.9, C_EEG)

# Stage 2: Signal Processing
stage(PX, 9.4, PW, 1.5, 2, 'Signal Processing', C_PROC, [
    'FFT band extraction: delta, theta, alpha, beta, gamma',
    'Affective metrics: arousal, valence, focus, relaxation',
    'Sensitivity & smoothing (user-configurable)',
    'Output: normalized band powers [0,1] + 4 metrics',
])
arrow_v(PX + PW/2, 9.4, 8.8, C_PROC)

# Stage 3: Vibe Prompt + Parameter Mapping
stage(PX, 7.2, PW, 1.6, 3, 'EEG-to-Music Mapping', C_STYLE, [
    'Parameter mapping (band ratios):',
    '  temperature = theta/alpha ratio  [0.3, 3.0]',
    '  top-k = gamma/beta ratio  [5, 100]',
    '  CFG guidance = f(focus, arousal)  [1.5, 6.5]',
    '  drums = 1 if arousal > 0.4, else -1',
    'Layered vibe prompt: energy + mood + texture + space + instruments',
])
arrow_v(PX + PW/2, 7.2, 6.6, C_STYLE)

# Stage 4: Style Embedding + Generation
stage(PX, 4.7, PW, 1.9, 4, 'Neural Audio Generation', C_GEN, [
    'MusicCoCa embed_style(): text -> 768-dim style embedding',
    'Style interpolation: blend = 0.4, vibe history = 10',
    'Magenta RealTime 2 (mrt2_small, 230M params, MLX)',
    'threading.Lock guards TFLite interpreters',
    '25 frames/chunk, 48 kHz stereo, 16-bit PCM output',
    'Real-time ratio: 2.5-3.0x on Apple Silicon',
])
arrow_v(PX + PW/2, 4.7, 4.1, C_GEN)

# Stage 5: Web Playback
stage(PX, 2.3, PW, 1.8, 5, 'Web Dashboard & Playback', C_WEB, [
    'WebSocket server: PCM chunks as base64',
    'Web Audio API: buffer 3s, lookahead 10s, 20ms polling',
    'Catch-up logic if nextPlayTime < currentTime',
    'Chart.js: real-time EEG band visualization',
    'User controls: sensitivity sliders, start/stop',
])

# ── Feedback loop arrow (right side) ───────────────────────────────────
# From Stage 5 back up to Stage 1 (EEG data broadcast)
fb_x = PX + PW + 0.8
arrow_h(PX + PW, fb_x, 3.2, C_EEG, 1.5)
arrow_v(fb_x, 3.2, 12.2, C_EEG, 1.5)
arrow_h(fb_x, PX + PW, 12.2, C_EEG, 1.5)
side_label(fb_x + 0.3, 7.7, 'EEG data\n(fire-and-forget\nbroadcast)', C_EEG, 7.5)

# ── Validation branch (left side) ──────────────────────────────────────
val_x = 0.3
val_w = 2.2

# Arrow from Stage 1 to validation
arrow_h(PX, val_x + val_w, 12.2, C_VAL, 1.5)
arrow_v(val_x + val_w, 12.2, 10.8, C_VAL, 1.5)
arrow_h(val_x + val_w, val_x + val_w, 10.8, C_VAL, 1.5)

# Validation box
vb = FancyBboxPatch((val_x, 8.8), val_w, 2.0, boxstyle="round,pad=0.2",
                    facecolor=C_VAL, edgecolor='white', alpha=0.15, zorder=2)
ax.add_patch(vb)
ax.text(val_x + val_w/2, 10.5, 'Offline Validation', ha='center', va='center',
        fontsize=10, color=C_VAL, fontweight='bold', zorder=4)
val_details = [
    'EEG Fingerprinting',
    'LaBraM pretrained transformer',
    '3000 samples -> CLS token',
    'UMAP / PCA visualization',
    'KNN: 97-100% accuracy',
    '',
    'Validates separability of',
    'listening-state neural signatures',
]
for i, line in enumerate(val_details):
    wt = 'bold' if i == 0 else 'normal'
    fs = 8 if i == 0 else 7
    ax.text(val_x + val_w/2, 10.1 - i * 0.28, line, ha='center', va='center',
            fontsize=fs, color=C_SUB if i > 0 else C_VAL, fontweight=wt, zorder=4)

# ── Update flow annotations (right side) ───────────────────────────────
side_label(PX + PW + 0.45, 11.1, 'raw EEG\nsamples', C_EEG, 7.5)
side_label(PX + PW + 0.45, 9.0, 'bands +\nmetrics', C_PROC, 7.5)
side_label(PX + PW + 0.45, 6.8, 'params +\nprompt text', C_STYLE, 7.5)
side_label(PX + PW + 0.45, 4.3, 'audio\nchunks', C_GEN, 7.5)

# ── Update interval annotation ─────────────────────────────────────────
ax.text(PX + PW/2, 8.95, 'update every 3s',
        ha='center', va='center', fontsize=7.5, color=C_SUB,
        style='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_ARROW, alpha=0.9))

# ── Save ───────────────────────────────────────────────────────────────
fig.savefig('architecture.png', dpi=300, facecolor=C_BG, bbox_inches='tight',
            pad_inches=0.5)
print("Saved: architecture.png (300 DPI, 12x14 inches)")
