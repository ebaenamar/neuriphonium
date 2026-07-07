"""
EEG Fingerprint Analysis — Extract embeddings from Muse EEG recordings
and visualize whether different music genres produce separable neural signatures.

Data: reads CSV files from ~/Desktop/MuseData/Documents/
Output: saves UMAP plot + embeddings to ./output/fingerprint/

NOTE: Muse app records at ~16Hz (not 256Hz). This limits frequency analysis
to delta (1-4Hz) and theta (4-8Hz) bands. LaBraM embeddings use heavy
upsampling which may reduce fidelity but can still capture temporal patterns.

Usage:
    venv/bin/python eeg_fingerprint.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import umap

MUSE_DATA_DIR = os.path.expanduser("~/Desktop/MuseData/Documents")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "fingerprint")
EPOCH_SECONDS = 10.0
TARGET_SFREQ = 200
MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

LABEL_MAP = {
    'chopsuey': 'trash_metal',
    'meditation': 'gregoriano',
    'people': 'silencio',
    'peolpe': 'silencio',
}


def load_sessions(data_dir):
    sessions = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "muse_*.csv")))
    
    for csv_path in csv_files:
        meta_path = csv_path.replace('.csv', '_meta.json')
        if not os.path.exists(meta_path):
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        tag = meta.get('activity_tag', 'unknown').lower()
        label = LABEL_MAP.get(tag, tag)
        
        df = pd.read_csv(csv_path)
        eeg_cols = ['tp9', 'af7', 'af8', 'tp10']
        if not all(c in df.columns for c in eeg_cols):
            continue
        
        eeg = df[eeg_cols].values.astype(np.float32)
        offsets = df['session_offset_ms'].values
        duration_s = (offsets.max() - offsets.min()) / 1000.0
        
        if duration_s < 30:
            print(f"  SKIP {os.path.basename(csv_path)}: too short ({duration_s:.0f}s)")
            continue
        
        actual_sfreq = len(eeg) / duration_s
        
        sessions.append({
            'name': os.path.basename(csv_path),
            'label': label,
            'eeg': eeg,
            'sfreq': actual_sfreq,
            'duration_s': duration_s,
            'n_samples': len(eeg),
        })
        
        print(f"  LOADED {os.path.basename(csv_path)}: {label} | {duration_s:.1f}s | {actual_sfreq:.1f}Hz | {len(eeg)} samples")
    
    return sessions


def epoch_eeg(eeg, sfreq, epoch_seconds=10.0):
    epoch_len = int(epoch_seconds * sfreq)
    n_epochs = len(eeg) // epoch_len
    if n_epochs == 0:
        return np.array([])
    epochs = []
    for i in range(n_epochs):
        start = i * epoch_len
        end = start + epoch_len
        epochs.append(eeg[start:end])
    return np.array(epochs)


def resample_epochs(epochs, orig_sfreq, target_sfreq):
    n_epochs, n_samples, n_chans = epochs.shape
    target_samples = int(n_samples * target_sfreq / orig_sfreq)
    resampled = np.zeros((n_epochs, target_samples, n_chans), dtype=np.float32)
    for i in range(n_epochs):
        for c in range(n_chans):
            resampled[i, :, c] = resample(epochs[i, :, c], target_samples)
    return resampled


def extract_labram_embeddings(epochs, batch_size=4):
    from braindecode.models import Labram
    
    print("  Loading pretrained LaBraM...")
    model = Labram.from_pretrained('braindecode/labram-pretrained')
    model.eval()
    model.reset_classifier(0)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"  Device: {device}")
    
    all_embeddings = []
    n_epochs = len(epochs)
    
    with torch.no_grad():
        for i in range(0, n_epochs, batch_size):
            batch = epochs[i:i+batch_size]
            x = torch.from_numpy(batch).float().to(device)
            x = x.permute(0, 2, 1)
            
            result = model(x, ch_names=MUSE_CHANNELS, return_features=True)
            cls = result['cls_token'].cpu().numpy()
            all_embeddings.append(cls)
            
            print(f"    Embedded {min(i+batch_size, n_epochs)}/{n_epochs}")
    
    return np.vstack(all_embeddings)


def extract_bandpower_features(epochs, sfreq):
    nyquist = sfreq / 2.0
    all_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }
    bands = {name: (lo, hi) for name, (lo, hi) in all_bands.items() if hi < nyquist}
    
    print(f"  Available bands at {sfreq:.1f}Hz (Nyquist={nyquist:.1f}Hz): {list(bands.keys())}")
    
    n_epochs = len(epochs)
    n_features = len(bands) * 4
    features = np.zeros((n_epochs, n_features), dtype=np.float32)
    nperseg = min(256, epochs.shape[1])
    
    for i in range(n_epochs):
        for ch in range(4):
            freqs, psd = welch(epochs[i, :, ch], fs=sfreq, nperseg=nperseg)
            for j, (band, (low, high)) in enumerate(bands.items()):
                mask = (freqs >= low) & (freqs <= high)
                if mask.any():
                    power = np.mean(psd[mask])
                    features[i, ch * len(bands) + j] = np.log10(power + 1e-10)
    
    return features, list(bands.keys())


def extract_temporal_features(epochs, sfreq):
    n_epochs = len(epochs)
    features = []
    
    for i in range(n_epochs):
        epoch = epochs[i]
        ch_features = []
        for ch in range(4):
            signal = epoch[:, ch]
            ch_features.extend([
                np.mean(signal),
                np.std(signal),
                np.max(signal) - np.min(signal),
                np.mean(np.abs(np.diff(signal))),
                np.mean(signal ** 2),
                np.std(np.diff(signal)),
                np.std(np.diff(np.diff(signal))) / (np.std(np.diff(signal)) + 1e-10),
                len(np.where(np.diff(np.sign(signal)))[0]) / len(signal),
            ])
        features.append(ch_features)
    
    return np.array(features, dtype=np.float32)


def run_umap(embeddings, labels, title, save_path):
    n = len(embeddings)
    n_neighbors = min(15, n - 1)
    if n_neighbors < 2:
        print(f"  Too few samples ({n}) for UMAP")
        return None
    
    print(f"  UMAP: {n} samples, n_neighbors={n_neighbors}")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 2)))
    
    for idx, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()
    
    return embedding_2d


def run_pca(embeddings, labels, title, save_path):
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 2)))
    
    for idx, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    
    ax.set_title(f"{title} ({pca.explained_variance_ratio_.sum():.1%} variance)", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()
    
    return emb_2d


def classify(embeddings, labels, name):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    if len(set(y)) < 2:
        print(f"  {name}: only one class, skipping")
        return 0.0
    
    n = len(y)
    k = min(5, n - 1)
    cv = min(5, n)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X, y, cv=cv)
    acc = scores.mean()
    print(f"  {name}: KNN accuracy = {acc:.2%} (±{scores.std():.2%})")
    return acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("EEG Fingerprint Analysis")
    print("=" * 60)
    
    print("\n[1/5] Loading Muse sessions...")
    sessions = load_sessions(MUSE_DATA_DIR)
    if not sessions:
        print("No valid sessions found!")
        return
    
    print(f"\n[2/5] Epoching EEG data ({EPOCH_SECONDS}s epochs)...")
    EPOCH_SAMPLES_ORIG = 160  # Fixed: ~16Hz * 10s
    # LaBraM pretrained expects 3000 samples (15s at 200Hz)
    EPOCH_SAMPLES_200_LABRAM = 3000
    
    all_epochs_orig = []
    all_epochs_200 = []
    all_labels = []
    
    for sess in sessions:
        # Epoch at original rate (10s epochs)
        epochs_orig = epoch_eeg(sess['eeg'], sess['sfreq'], EPOCH_SECONDS)
        if len(epochs_orig) == 0:
            print(f"  SKIP {sess['name']}: not enough samples for {EPOCH_SECONDS}s epochs")
            continue
        
        # Resample each epoch to fixed sample count (160 for 10s@16Hz)
        n_chans = epochs_orig.shape[2]
        epochs_orig_fixed = np.zeros((len(epochs_orig), EPOCH_SAMPLES_ORIG, n_chans), dtype=np.float32)
        for i in range(len(epochs_orig)):
            for c in range(n_chans):
                epochs_orig_fixed[i, :, c] = resample(epochs_orig[i, :, c], EPOCH_SAMPLES_ORIG)
        
        # For LaBraM: combine 3 epochs (30s) to get 15s at 200Hz after resampling
        # Actually: resample 10s@16Hz (160 samples) to 3000 samples (15s@200Hz)
        # This stretches 10s of data to fill 15s — not ideal but preserves temporal patterns
        epochs_200 = np.zeros((len(epochs_orig_fixed), EPOCH_SAMPLES_200_LABRAM, n_chans), dtype=np.float32)
        for i in range(len(epochs_orig_fixed)):
            for c in range(n_chans):
                epochs_200[i, :, c] = resample(epochs_orig_fixed[i, :, c], EPOCH_SAMPLES_200_LABRAM)
        
        all_epochs_orig.append(epochs_orig_fixed)
        all_epochs_200.append(epochs_200)
        all_labels.extend([sess['label']] * len(epochs_orig_fixed))
        
        print(f"  {sess['name']}: {len(epochs_orig_fixed)} epochs -> {EPOCH_SAMPLES_200_LABRAM} samples for LaBraM ({sess['label']})")
    
    all_epochs_orig = np.vstack(all_epochs_orig)
    all_epochs_200 = np.vstack(all_epochs_200)
    all_labels = np.array(all_labels)
    
    print(f"\nTotal: {len(all_labels)} epochs")
    for label in sorted(set(all_labels)):
        count = sum(all_labels == label)
        print(f"  {label}: {count} epochs")
    
    print("\n[3/5] Extracting LaBraM embeddings (upsampled to 200Hz)...")
    labram_embeddings = extract_labram_embeddings(all_epochs_200, batch_size=4)
    print(f"  LaBraM embeddings: {labram_embeddings.shape}")
    
    actual_sfreq = sessions[0]['sfreq']
    print(f"\n[4/5] Extracting band-power + temporal features @ {actual_sfreq:.1f}Hz...")
    bandpower_features, available_bands = extract_bandpower_features(all_epochs_orig, actual_sfreq)
    temporal_features = extract_temporal_features(all_epochs_orig, actual_sfreq)
    combined_features = np.hstack([bandpower_features, temporal_features])
    print(f"  Band-power: {bandpower_features.shape} (bands: {available_bands})")
    print(f"  Temporal: {temporal_features.shape}")
    print(f"  Combined: {combined_features.shape}")
    
    print("\n[5/5] Visualization + classification...")
    
    umap_labram = run_umap(
        labram_embeddings, all_labels,
        "LaBraM Embeddings - EEG Fingerprint (Muse ~16Hz, upscaled)",
        os.path.join(OUTPUT_DIR, "umap_labram.png")
    )
    
    umap_combined = run_umap(
        combined_features, all_labels,
        f"Band-Power + Temporal - EEG Fingerprint ({available_bands} only)",
        os.path.join(OUTPUT_DIR, "umap_combined.png")
    )
    
    pca_labram = run_pca(
        labram_embeddings, all_labels,
        "LaBraM Embeddings (PCA)",
        os.path.join(OUTPUT_DIR, "pca_labram.png")
    )
    
    print("\nClassification results (KNN cross-val):")
    print("-" * 50)
    acc_labram = classify(labram_embeddings, all_labels, "LaBraM (200d)")
    acc_bands = classify(bandpower_features, all_labels, f"Band-power ({len(available_bands)} bands)")
    acc_temporal = classify(temporal_features, all_labels, "Temporal (32d)")
    acc_combined = classify(combined_features, all_labels, "Combined (band+temporal)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Epochs: {len(all_labels)} ({EPOCH_SECONDS}s each)")
    print(f"Sample rate: ~{actual_sfreq:.1f}Hz (Muse app downsampled)")
    print(f"Available bands: {available_bands}")
    print(f"Classes: {sorted(set(all_labels))}")
    print()
    print("KNN Accuracy:")
    print(f"  LaBraM embeddings:      {acc_labram:.2%}")
    print(f"  Band-power only:        {acc_bands:.2%}")
    print(f"  Temporal only:          {acc_temporal:.2%}")
    print(f"  Combined (band+temp):   {acc_combined:.2%}")
    print()
    print(f"Plots saved to: {OUTPUT_DIR}/")
    
    np.savez(
        os.path.join(OUTPUT_DIR, "embeddings.npz"),
        labram=labram_embeddings,
        bandpower=bandpower_features,
        temporal=temporal_features,
        combined=combined_features,
        labels=all_labels,
        available_bands=available_bands,
        sfreq=actual_sfreq,
    )
    print(f"Embeddings saved to: {os.path.join(OUTPUT_DIR, 'embeddings.npz')}")
    
    if actual_sfreq < 50:
        print(f"\nWARNING: Sample rate is {actual_sfreq:.1f}Hz - below Nyquist for alpha/beta/gamma bands.")
        print("   For better results, record with: muselsl stream (256Hz) + muselsl record")


if __name__ == "__main__":
    main()
