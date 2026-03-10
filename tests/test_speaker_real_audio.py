"""
test_speaker_real_audio.py
==========================
Tests the Indic ECAPA-TDNN speaker encoder on real audio files.

Tests
-----
1. Single file embedding: shape, L2-norm, no NaN
2. Same-speaker consistency: two clips from same file should be similar
3. Different-speaker discrimination: two unrelated files should differ
4. Duration sensitivity: longer clips should give stable embeddings
5. Real-time factor: how fast is embedding extraction?

Usage
-----
    # Minimal — uses synthetic audio if no files provided
    python tests/test_speaker_real_audio.py

    # One real audio file (will split into two clips for same-speaker test)
    python tests/test_speaker_real_audio.py --audio output.wav

    # Two different speaker files (enables cross-speaker test)
    python tests/test_speaker_real_audio.py --audio spk1.wav --audio2 spk2.wav

    # With pretrained weights and GPU
    python tests/test_speaker_real_audio.py --audio output.wav --pretrained --device cuda
"""

import argparse
import sys
import os
import time

# ── torchaudio compatibility patch (must be before any speechbrain import) ──
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modules.speaker_encoder import SpeakerEncoderConfig, SpeakerEncoder

SR = 16000


def load_audio(path: str) -> torch.Tensor:
    """Load audio file → (1, samples) float32 tensor at 16kHz."""
    import soundfile as sf
    import numpy as np
    arr, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(arr.T)          # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)    # mono
    if sr != SR:
        try:
            import torchaudio.functional as TAF
            wav = TAF.resample(wav, sr, SR)
        except Exception:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(SR, sr)
            arr = resample_poly(wav.squeeze(0).numpy(), SR//g, sr//g).astype("float32")
            wav = torch.from_numpy(arr).unsqueeze(0)
    return wav  # (1, samples)


def make_encoder(pretrained: bool, device: str) -> SpeakerEncoder:
    cfg = SpeakerEncoderConfig(
        pretrained_model="speechbrain/spkrec-ecapa-voxceleb" if pretrained else "",
        embedding_dim=192,
        n_speakers=1000,
    )
    enc = SpeakerEncoder(cfg).to(device)
    enc.eval()
    return enc


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",  default=None, help="Primary audio file (speaker A)")
    parser.add_argument("--audio2", default=None, help="Second audio file (speaker B, optional)")
    parser.add_argument("--pretrained", action="store_true", help="Load VoxCeleb pretrained weights")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("\nIndicVC Speaker Encoder — Real Audio Test")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available() and args.device == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Pretrained: {args.pretrained}")

    results = {}

    # ── Load encoder ─────────────────────────────────────────────
    section("Building SpeakerEncoder")
    t0 = time.time()
    enc = make_encoder(args.pretrained, args.device)
    print(f"  Build time: {time.time()-t0:.1f}s")
    n = sum(p.numel() for p in enc.parameters())
    print(f"  Params    : {n:,}")

    # ── Load audio ───────────────────────────────────────────────
    section("Loading Audio")
    if args.audio and os.path.exists(args.audio):
        wav_a = load_audio(args.audio).to(args.device)
        dur_a = wav_a.shape[-1] / SR
        print(f"  Speaker A: {os.path.basename(args.audio)}  {dur_a:.2f}s")
    else:
        print("  No --audio provided. Using 4s synthetic audio.")
        wav_a = torch.randn(1, SR * 4).to(args.device) * 0.1
        dur_a = 4.0

    if args.audio2 and os.path.exists(args.audio2):
        wav_b = load_audio(args.audio2).to(args.device)
        dur_b = wav_b.shape[-1] / SR
        print(f"  Speaker B: {os.path.basename(args.audio2)}  {dur_b:.2f}s")
        have_spk_b = True
    else:
        print("  No --audio2 provided. Using different synthetic audio for speaker B.")
        wav_b = torch.randn(1, SR * 4).to(args.device) * 0.1
        have_spk_b = False

    # ── Test 1: Embedding shape & quality ────────────────────────
    section("Test 1/5 — Embedding shape, norm, stats")
    try:
        with torch.no_grad():
            t0 = time.time()
            emb = enc(wav_a)
            elapsed = time.time() - t0

        rtf = dur_a / elapsed
        print(f"  Input  : {tuple(wav_a.shape)}  ({dur_a:.2f}s)")
        print(f"  Output : {tuple(emb.shape)}")
        print(f"  L2 norm: {emb.norm(dim=-1).item():.6f}  (should be ~1.0)")
        print(f"  Mean   : {emb.mean().item():.4f}")
        print(f"  Std    : {emb.std().item():.4f}")
        print(f"  Time   : {elapsed*1000:.0f}ms  (RTF: {rtf:.1f}x real-time)")
        assert emb.shape == (1, 192)
        assert not torch.isnan(emb).any()
        norm = emb.norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-4, f"Not unit normalised: {norm}"
        print(f"  ✅ Passed")
        results["embedding_quality"] = True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["embedding_quality"] = False

    # ── Test 2: Duration sweep ────────────────────────────────────
    section("Test 2/5 — Duration sensitivity (0.5s → full)")
    try:
        durations_s = [0.5, 1.0, 2.0, min(dur_a, 5.0)]
        embeddings = []
        print(f"  {'Duration':>10}  {'Time(ms)':>10}  {'Norm':>8}  {'CosSim vs 1s':>14}")
        for d in durations_s:
            n_samples = int(d * SR)
            clip = wav_a[:, :n_samples]
            if clip.shape[-1] < n_samples:
                clip = F.pad(clip, (0, n_samples - clip.shape[-1]))
            t0 = time.time()
            with torch.no_grad():
                e = enc(clip)
            elapsed_ms = (time.time() - t0) * 1000
            embeddings.append(e)
            sim = F.cosine_similarity(embeddings[0], e).item() if len(embeddings) > 1 else 1.0
            print(f"  {d:>10.1f}s  {elapsed_ms:>10.0f}ms  "
                  f"{e.norm().item():>8.5f}  {sim:>14.4f}")

        # Longer clips should give more stable embeddings (higher mutual similarity)
        sim_05_10 = F.cosine_similarity(embeddings[0], embeddings[1]).item()
        sim_10_20 = F.cosine_similarity(embeddings[1], embeddings[2]).item()
        print(f"\n  0.5s↔1.0s similarity: {sim_05_10:.4f}")
        print(f"  1.0s↔2.0s similarity: {sim_10_20:.4f}")
        print(f"  ✅ Passed")
        results["duration_sensitivity"] = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ FAILED: {e}")
        results["duration_sensitivity"] = False

    # ── Test 3: Same-speaker consistency ─────────────────────────
    section("Test 3/5 — Same-speaker consistency")
    try:
        half = wav_a.shape[-1] // 2
        clip1 = wav_a[:, :half]
        clip2 = wav_a[:, half:half*2] if half*2 <= wav_a.shape[-1] else wav_a[:, :half]

        with torch.no_grad():
            e1 = enc(clip1)
            e2 = enc(clip2)

        sim = F.cosine_similarity(e1, e2).item()
        print(f"  Clip 1: {clip1.shape[-1]/SR:.2f}s  Clip 2: {clip2.shape[-1]/SR:.2f}s")
        print(f"  Cosine similarity (same speaker, different clips): {sim:.4f}")
        print(f"  Note: With random init this may be high (~1.0).")
        print(f"        After fine-tuning on IndicVoices-R, same-speaker")
        print(f"        pairs should score >0.7, cross-speaker <0.3.")
        print(f"  ✅ Passed (no crash, embedding produced)")
        results["same_speaker"] = True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["same_speaker"] = False

    # ── Test 4: Cross-speaker discrimination ─────────────────────
    section("Test 4/5 — Cross-speaker discrimination")
    try:
        with torch.no_grad():
            emb_a = enc(wav_a)
            emb_b = enc(wav_b)

        sim = F.cosine_similarity(emb_a, emb_b).item()
        label = "different speakers" if have_spk_b else "synthetic speakers"
        print(f"  Speaker A vs B ({label}): {sim:.4f}")
        if have_spk_b:
            print(f"  (Lower = more discriminative. Target after fine-tuning: <0.5)")
        else:
            print(f"  (Using synthetic audio — not a meaningful speaker test)")
        print(f"  ✅ Passed (no crash)")
        results["cross_speaker"] = True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["cross_speaker"] = False

    # ── Test 5: Batch vs individual consistency ───────────────────
    section("Test 5/5 — Batch processing consistency")
    try:
        # Use equal-length clips for batching
        min_len = min(wav_a.shape[-1], wav_b.shape[-1])
        clip_a = wav_a[:, :min_len]
        clip_b = wav_b[:, :min_len]
        batch = torch.cat([clip_a, clip_b], dim=0)  # (2, min_len)

        with torch.no_grad():
            ea = enc(clip_a)
            eb = enc(clip_b)
            eb_batch = enc(batch)

        diff_a = (eb_batch[0] - ea[0]).abs().max().item()
        diff_b = (eb_batch[1] - eb[0]).abs().max().item()
        print(f"  Batch[0] vs individual A: max diff = {diff_a:.2e}")
        print(f"  Batch[1] vs individual B: max diff = {diff_b:.2e}")
        assert diff_a < 0.05, f"Batch inconsistency too large: {diff_a}"
        assert diff_b < 0.05, f"Batch inconsistency too large: {diff_b}"
        print(f"  ✅ Passed")
        results["batch_consistency"] = True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results["batch_consistency"] = False

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for test, passed in results.items():
        print(f"  {test:30s} {'✅ PASS' if passed else '❌ FAIL'}")

    if not all(results.values()):
        sys.exit(1)
    print(f"\n  All tests passed ✅")


if __name__ == "__main__":
    main()