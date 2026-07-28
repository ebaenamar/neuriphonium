"""NMED-T paired EEG/audio dataset for cross-modal alignment training.

Layout expected under the dataset root:
    root/data_processed/song{21..30}_Imputed.mat   (aggregated NMED-T EEG)
    root/music/{song_name}.wav                     (user-provided stimulus audio)

Each .mat holds data{song_id} of shape (125, n_samples, n_subjects) at 125 Hz
and subs{song_id} with subject ids. Windows of EEG are paired with the
time-aligned MusicCoCa clip embedding of the same song segment.
"""

import os
import re

import numpy as np
import resampy
import scipy.io
import soundfile as sf
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

NMED_T_SONGS = {
    "first_fires": 21,
    "oino": 22,
    "tiptoes": 23,
    "careless_love": 24,
    "lebanese_blonde": 25,
    "canopee": 26,
    "doing_yoga": 27,
    "until_the_sun_needs_to_rise": 28,
    "silent_shout": 29,
    "the_last_thing_you_should_do": 30,
}

EEG_SR = 125
LABRAM_SR = 200
AUDIO_SR = 16000
N_EEG_CHANNELS = 125


def _normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    return waveform / (np.max(np.abs(waveform)) + 1e-8) * 0.5


def egi_chs_info(n_channels=N_EEG_CHANNELS):
    """MNE GSN-HydroCel-128 positions for the first n EGI channels."""
    import mne
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    ch_pos = montage.get_positions()["ch_pos"]
    info = []
    for i in range(1, n_channels + 1):
        name = f"E{i}"
        if name not in ch_pos:
            raise ValueError(f"Channel {name!r} not in GSN-HydroCel-128 montage")
        info.append({"ch_name": name, "kind": "eeg",
                     "loc": np.asarray(ch_pos[name], dtype=float)})
    return info


class NMEDTAlignDataset(Dataset):
    """Yields (eeg_window, target_embedding, song_idx) triples.

    eeg_window: float32 tensor (n_channels, window_seconds * LABRAM_SR),
    z-scored per channel over the full song, resampled 125 -> 200 Hz.
    target_embedding: float32 (768,) MusicCoCa clip embedding aligned in time.
    """

    def __init__(self, root, embedder, window_seconds=10.0, hop_seconds=10.0,
                 songs=None, subjects=None):
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.win_eeg = int(window_seconds * EEG_SR)
        self.win_labram = int(window_seconds * LABRAM_SR)
        self.embedder = embedder

        music_dir = os.path.join(root, "music")
        eeg_dir = os.path.join(root, "data_processed")
        if not os.path.isdir(music_dir):
            raise FileNotFoundError(f"Audio dir not found: {music_dir}")
        if not os.path.isdir(eeg_dir):
            raise FileNotFoundError(f"EEG dir not found: {eeg_dir}")

        wanted = songs if songs is not None else list(NMED_T_SONGS)
        self.song_names = [s for s in wanted if s in NMED_T_SONGS]
        self.song_index = {s: i for i, s in enumerate(self.song_names)}

        self.items = []
        for song_name in self.song_names:
            song_id = NMED_T_SONGS[song_name]
            wav_path = os.path.join(music_dir, f"{song_name}.wav")
            if not os.path.isfile(wav_path):
                print(f"  Skipping {song_name}: missing {wav_path}")
                continue
            samples, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            audio = samples.mean(axis=1)
            if sr != AUDIO_SR:
                audio = resampy.resample(audio, sr, AUDIO_SR)
            audio = _normalize_wav(audio)

            mat_path = None
            for f in os.listdir(eeg_dir):
                if f.endswith(".mat") and re.findall(r"\d+", f) and \
                        int(re.findall(r"\d+", f)[0]) == song_id:
                    mat_path = os.path.join(eeg_dir, f)
                    break
            if mat_path is None:
                print(f"  Skipping {song_name}: no .mat for song id {song_id}")
                continue

            mat = scipy.io.loadmat(mat_path)
            data = mat[f"data{song_id}"].astype(np.float32)  # (ch, ts, subs)
            subs = [str(s[0]) for s in mat[f"subs{song_id}"].flatten()]
            data = (data - data.mean(axis=1, keepdims=True)) / \
                   (data.std(axis=1, keepdims=True) + 1e-6)

            emb = embedder.embed_song(song_name, audio, hop_seconds)
            n_clips = emb.shape[0]
            hop_eeg = int(hop_seconds * EEG_SR)

            n_windows = 0
            for sub_idx, sub in enumerate(subs):
                if subjects is not None and sub not in subjects:
                    continue
                for w, start in enumerate(range(0, data.shape[1] - self.win_eeg + 1, hop_eeg)):
                    if w >= n_clips:
                        break
                    self.items.append({
                        "song": song_name,
                        "song_idx": self.song_index[song_name],
                        "sub": sub,
                        "sub_idx": sub_idx,
                        "eeg_start": start,
                        "clip_idx": w,
                        "_song_data": song_name,
                    })
                    n_windows += 1
            self._data = getattr(self, "_data", {})
            self._data[song_name] = data
            print(f"  {song_name}: {n_windows} windows "
                  f"({data.shape[2]} subs x {data.shape[1] / EEG_SR:.0f}s, {n_clips} clips)")

        if not self.items:
            raise RuntimeError("No paired EEG/audio windows found. Check dataset root.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        data = self._data[it["_song_data"]]  # (ch, ts, subs)
        seg = data[:, it["eeg_start"]:it["eeg_start"] + self.win_eeg, it["sub_idx"]]
        seg = resample(seg, self.win_labram, axis=1).astype(np.float32)
        target = self.embedder.clip_embedding(it["song"], self.hop_seconds, it["clip_idx"])
        return (torch.from_numpy(seg),
                torch.from_numpy(np.asarray(target, dtype=np.float32)),
                it["song_idx"])
