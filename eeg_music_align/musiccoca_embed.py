"""MusicCoCa audio embedding extraction with disk cache.

Uses the frozen MusicCoCa audio tower (TFLite, from magenta-rt) to embed
song audio into the shared 768-dim style space consumed by Magenta RT2.
Embeddings are computed per-song with a configurable hop and cached to
.npz so training never re-runs the TFLite encoders.
"""

from pathlib import Path

import numpy as np

MUSICCOCA_SR = 16000
CLIP_SECONDS = 10.0
EMBEDDING_DIM = 768


class MusicCoCaSongEmbedder:
    """Precomputes MusicCoCa embeddings for full songs on a fixed hop grid."""

    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self._model = None
        self._cache = {}
        if self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as z:
                for key in z.files:
                    self._cache[key] = z[key]
            print(f"  MusicCoCa cache loaded: {len(self._cache)} songs from {self.cache_path}")

    def _load_model(self):
        if self._model is None:
            from magenta_rt.musiccoca import MusicCoCa
            print("  Loading MusicCoCa (TFLite)...")
            self._model = MusicCoCa(lazy=False)
        return self._model

    @staticmethod
    def _key(song_name, hop_seconds):
        return f"{song_name}@hop{hop_seconds:g}"

    def n_clips(self, song_name, hop_seconds):
        key = self._key(song_name, hop_seconds)
        return self._cache[key].shape[0] if key in self._cache else 0

    def clip_embedding(self, song_name, hop_seconds, clip_idx):
        return self._cache[self._key(song_name, hop_seconds)][clip_idx]

    def song_embeddings(self, song_name, hop_seconds):
        return self._cache[self._key(song_name, hop_seconds)]

    def embed_song(self, song_name, samples_16k, hop_seconds):
        """Embeds a full mono 16kHz song into (n_clips, 768) on the hop grid."""
        key = self._key(song_name, hop_seconds)
        if key in self._cache:
            return self._cache[key]
        from magenta_rt.audio import Waveform
        model = self._load_model()
        wave = Waveform(np.asarray(samples_16k, dtype=np.float32), MUSICCOCA_SR)
        emb = model.embed_batch_audio(
            [wave], hop_length=hop_seconds, pool_across_time=False, pad_end=False
        )[0]
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape != (emb.shape[0], EMBEDDING_DIM):
            raise RuntimeError(f"Unexpected MusicCoCa output shape {emb.shape}")
        self._cache[key] = emb
        print(f"  Embedded {song_name}: {emb.shape[0]} clips @ hop {hop_seconds:g}s")
        return emb

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, **self._cache)
        print(f"  MusicCoCa cache saved: {len(self._cache)} songs -> {self.cache_path}")
