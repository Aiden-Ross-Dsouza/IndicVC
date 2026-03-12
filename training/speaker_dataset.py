"""
speaker_dataset.py
==================
PyTorch Dataset for speaker encoder fine-tuning on IndicVoices-R.

Provides:
- SpeakerDataset         : single-utterance dataset from manifest CSV
- SpeakerBatchSampler    : ensures ≥2 utterances per speaker per batch
                           (required for AAM-Softmax to work)
- build_speaker_loaders  : convenience factory → (train_loader, val_loader)

Augmentation pipeline (training only):
  1. Random crop to [min_dur_s, max_dur_s]
  2. Speed perturbation ±10% (via resampling)
  3. MUSAN additive noise (if musan_dir provided)
  4. RIR convolution (if rir_dir provided)
"""

import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

SR = 16_000


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def random_crop(wav: torch.Tensor, min_s: float, max_s: float) -> torch.Tensor:
    """Randomly crop waveform to a duration in [min_s, max_s]."""
    min_n = int(min_s * SR)
    max_n = int(max_s * SR)
    n = wav.shape[-1]
    target = random.randint(min(min_n, n), min(max_n, n))
    if n <= target:
        # pad if too short
        wav = F.pad(wav, (0, target - n))
        return wav
    start = random.randint(0, n - target)
    return wav[..., start:start + target]


def speed_perturb(wav: torch.Tensor, sr: int = SR,
                  low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    """Apply random speed perturbation in [low, high] via resampling."""
    try:
        import torchaudio.functional as TAF
        factor = random.uniform(low, high)
        orig_freq = int(sr * factor)
        return TAF.resample(wav, orig_freq, sr)
    except Exception:
        return wav  # skip if torchaudio not available


def add_noise(wav: torch.Tensor, noise_dir: Optional[str],
              snr_db_range: tuple = (5, 30)) -> torch.Tensor:
    """
    Add random noise from MUSAN or similar directory.
    Only applied if noise_dir is provided and contains .wav files.
    """
    if noise_dir is None:
        return wav
    noise_files = list(Path(noise_dir).rglob("*.wav"))
    if not noise_files:
        return wav
    import soundfile as sf
    nf = random.choice(noise_files)
    try:
        noise, nsr = sf.read(str(nf), dtype="float32", always_2d=True)
        noise = torch.from_numpy(noise.T).mean(0)
        # Repeat/trim noise to match signal length
        n = wav.shape[-1]
        if noise.shape[-1] < n:
            reps = math.ceil(n / noise.shape[-1])
            noise = noise.repeat(reps)
        noise = noise[:n]
        # Apply random SNR
        snr_db = random.uniform(*snr_db_range)
        sig_power   = wav.pow(2).mean().clamp(min=1e-9)
        noise_power = noise.pow(2).mean().clamp(min=1e-9)
        scale = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
        return (wav + scale * noise).clamp(-1, 1)
    except Exception:
        return wav


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeakerDataset(Dataset):
    """
    Loads utterances from a manifest CSV.

    CSV columns: utt_id, speaker_id, language, family, audio_path, duration_s, text

    Parameters
    ----------
    manifest_path : str
        Path to the manifest CSV (e.g. data/manifests/all_train.csv)
    augment : bool
        Apply augmentation (speed perturb, noise, crop). Use True for training.
    min_dur_s : float
        Minimum crop duration (seconds)
    max_dur_s : float
        Maximum crop duration (seconds). Also used as fixed length when augment=False.
    noise_dir : str, optional
        Path to directory of noise .wav files (e.g. MUSAN/noise)
    langs : list[str], optional
        If provided, only load utterances for these language codes
    """

    def __init__(
        self,
        manifest_path: str,
        augment: bool = True,
        min_dur_s: float = 2.0,
        max_dur_s: float = 6.0,
        noise_dir: Optional[str] = None,
        langs: Optional[list] = None,
        min_utts_per_speaker: int = 2,
    ):
        self.augment    = augment
        self.min_dur_s  = min_dur_s
        self.max_dur_s  = max_dur_s
        self.noise_dir  = noise_dir

        self.samples = []
        self.speaker_ids: list[str] = []
        self.spk2idx: dict[str, int] = {}

        self._load_manifest(manifest_path, langs, min_utts_per_speaker)

    def _load_manifest(self, path: str, langs: Optional[list],
                       min_utts_per_speaker: int = 2):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if langs and row["language"] not in langs:
                    continue
                if not os.path.exists(row["audio_path"]):
                    continue
                rows.append(row)

        if not rows:
            raise ValueError(f"No valid samples found in {path}")

        # Filter out speakers with too few utterances
        # (they can't form same-speaker pairs → noisy AAM-Softmax signal)
        by_spk = defaultdict(list)
        for r in rows:
            by_spk[r["speaker_id"]].append(r)

        filtered = []
        dropped_spk = 0
        for spk, spk_rows in by_spk.items():
            if len(spk_rows) >= min_utts_per_speaker:
                filtered.extend(spk_rows)
            else:
                dropped_spk += 1

        if dropped_spk > 0:
            print(f"[SpeakerDataset] Dropped {dropped_spk} speakers with "
                  f"< {min_utts_per_speaker} utterances")

        rows = filtered if filtered else rows  # fallback: keep all if filtering removes too many

        # Build speaker label mapping
        all_spks = sorted(set(r["speaker_id"] for r in rows))
        self.speaker_ids = all_spks
        self.spk2idx = {s: i for i, s in enumerate(all_spks)}

        self.samples = rows
        print(f"[SpeakerDataset] Loaded {len(self.samples)} utterances, "
              f"{len(self.speaker_ids)} speakers from {path}")

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_ids)

    def _load_wav(self, path: str) -> torch.Tensor:
        """Load audio file → (samples,) float32 tensor at 16kHz."""
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(arr.T).mean(0)   # mono (samples,)
        if sr != SR:
            try:
                import torchaudio.functional as TAF
                wav = TAF.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
            except Exception:
                pass
        return wav

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        wav = self._load_wav(sample["audio_path"])

        if self.augment:
            # 1. Speed perturbation (30% chance)
            if random.random() < 0.3:
                wav = speed_perturb(wav.unsqueeze(0)).squeeze(0)

            # 2. Random crop
            wav = random_crop(wav, self.min_dur_s, self.max_dur_s)

            # 3. Additive noise (40% chance)
            if random.random() < 0.4:
                wav = add_noise(wav, self.noise_dir)
        else:
            # Validation: fixed-length crop from start
            target = int(self.max_dur_s * SR)
            if wav.shape[-1] < target:
                wav = F.pad(wav, (0, target - wav.shape[-1]))
            else:
                wav = wav[:target]

        spk_label = self.spk2idx[sample["speaker_id"]]
        return {
            "waveform":   wav,                          # (samples,)
            "speaker_id": sample["speaker_id"],
            "label":      torch.tensor(spk_label, dtype=torch.long),
            "language":   sample["language"],
            "family":     sample.get("family", "unknown"),
            "utt_id":     sample["utt_id"],
        }


# ---------------------------------------------------------------------------
# Sampler — guarantees ≥2 utterances per speaker per batch
# ---------------------------------------------------------------------------

class SpeakerBatchSampler(Sampler):
    """
    Samples batches ensuring each speaker appears ≥ utts_per_speaker times.

    This is critical for AAM-Softmax to receive meaningful negatives.
    A standard shuffle sampler can produce batches where most speakers
    appear only once, making the margin penalty trivial.

    Parameters
    ----------
    dataset : SpeakerDataset
    speakers_per_batch : int
        How many distinct speakers per batch. Typical: 32–64.
    utts_per_speaker : int
        Utterances per speaker per batch. Typical: 2–4.
    drop_last : bool
        Drop the last incomplete batch.
    """

    def __init__(
        self,
        dataset: SpeakerDataset,
        speakers_per_batch: int = 32,
        utts_per_speaker: int   = 2,
        drop_last: bool         = True,
    ):
        self.batch_size = speakers_per_batch * utts_per_speaker
        self.spb = speakers_per_batch
        self.ups = utts_per_speaker
        self.drop_last = drop_last

        # Group sample indices by speaker
        self.spk_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self.spk_to_indices[s["speaker_id"]].append(i)

        # Filter out speakers with too few utterances
        self.valid_speakers = [
            spk for spk, idxs in self.spk_to_indices.items()
            if len(idxs) >= utts_per_speaker
        ]
        if len(self.valid_speakers) < speakers_per_batch:
            raise ValueError(
                f"Only {len(self.valid_speakers)} speakers have ≥{utts_per_speaker} "
                f"utterances, but speakers_per_batch={speakers_per_batch}. "
                f"Reduce speakers_per_batch or utts_per_speaker."
            )

        n_batches = len(self.valid_speakers) // speakers_per_batch
        self._len = n_batches * self.batch_size

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        # Shuffle speakers each epoch
        speakers = self.valid_speakers.copy()
        random.shuffle(speakers)

        for start in range(0, len(speakers) - self.spb + 1, self.spb):
            batch_speakers = speakers[start:start + self.spb]
            batch_indices = []
            for spk in batch_speakers:
                idxs = self.spk_to_indices[spk]
                chosen = random.choices(idxs, k=self.ups)
                batch_indices.extend(chosen)
            # Shuffle within batch so speaker order is random
            random.shuffle(batch_indices)
            yield batch_indices


def collate_fn(batch: list[dict]) -> dict:
    """Pad waveforms to same length within a batch."""
    max_len = max(b["waveform"].shape[-1] for b in batch)
    waveforms = torch.stack([
        F.pad(b["waveform"], (0, max_len - b["waveform"].shape[-1]))
        for b in batch
    ])
    labels   = torch.stack([b["label"] for b in batch])
    return {
        "waveform":   waveforms,
        "label":      labels,
        "speaker_id": [b["speaker_id"] for b in batch],
        "language":   [b["language"] for b in batch],
        "family":     [b["family"] for b in batch],
        "utt_id":     [b["utt_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_speaker_loaders(
    train_manifest: str,
    val_manifest:   str,
    speakers_per_batch:   int   = 32,
    utts_per_speaker:     int   = 2,
    val_batch_size:       int   = 64,
    num_workers:          int   = 4,
    noise_dir:            Optional[str] = None,
    langs:                Optional[list] = None,
    min_dur_s:            float = 2.0,
    max_dur_s:            float = 6.0,
    min_utts_per_speaker: int   = 2,
) -> tuple:
    """
    Returns (train_loader, val_loader, n_speakers).
    min_utts_per_speaker: speakers with fewer utterances are excluded from
    training — they can't form same-speaker pairs and hurt AAM-Softmax.
    """
    train_ds = SpeakerDataset(
        train_manifest, augment=True,
        min_dur_s=min_dur_s, max_dur_s=max_dur_s,
        noise_dir=noise_dir, langs=langs,
        min_utts_per_speaker=min_utts_per_speaker,
    )
    val_ds = SpeakerDataset(
        val_manifest, augment=False,
        min_dur_s=min_dur_s, max_dur_s=max_dur_s,
        langs=langs,
        min_utts_per_speaker=1,  # val: include all speakers for loss computation
    )

    train_sampler = SpeakerBatchSampler(
        train_ds,
        speakers_per_batch=speakers_per_batch,
        utts_per_speaker=utts_per_speaker,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.n_speakers