"""
test_real_audio.py
==================
Tests both content encoder backends on a real audio file.
Downloads a sample Hindi utterance from IndicVoices-R if no file is provided.

Usage
-----
    # Use a sample from IndicVoices-R (auto-downloaded, ~5MB)
    python tests/test_real_audio.py

    # Use your own audio file
    python tests/test_real_audio.py --audio path/to/audio.wav

    # Test specific backend only
    python tests/test_real_audio.py --backend indicwhisper
    python tests/test_real_audio.py --backend indicconformer

    # Test specific language
    python tests/test_real_audio.py --lang ta
"""

import argparse
import sys
import time
import os

import torch

sys.path.insert(0, ".")
from modules.content_encoder import ContentEncoderConfig, ContentEncoder


TARGET_SR = 16000  # All models expect 16kHz


def load_audio(path: str) -> torch.Tensor:
    """
    Load an audio file and resample to 16kHz mono.
    Uses soundfile (no FFmpeg required) with scipy fallback for resampling.
    Returns tensor of shape (1, samples).
    """
    import soundfile as sf
    import numpy as np

    arr, sr = sf.read(path, dtype="float32", always_2d=True)
    # arr shape: (samples, channels) — transpose to (channels, samples)
    wav = torch.from_numpy(arr.T)  # (channels, samples)

    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        print(f"  Resampling from {sr}Hz → {TARGET_SR}Hz...")
        try:
            import torchaudio.functional as AF
            wav = AF.resample(wav, sr, TARGET_SR)
        except Exception:
            # Fallback: scipy resample
            from scipy.signal import resample_poly
            import math
            g = math.gcd(TARGET_SR, sr)
            arr_mono = wav.squeeze(0).numpy()
            arr_resampled = resample_poly(arr_mono, TARGET_SR // g, sr // g).astype(np.float32)
            wav = torch.from_numpy(arr_resampled).unsqueeze(0)

    return wav  # (1, samples)


def download_sample_audio(lang: str = "hi") -> str:
    """
    Download a single real utterance from IndicVoices-R HuggingFace.
    Saves to tests/sample_audio/<lang>_sample.wav
    Returns the file path.
    """
    out_dir = os.path.join("tests", "sample_audio")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang}_sample.wav")

    if os.path.exists(out_path):
        print(f"  Using cached sample: {out_path}")
        return out_path

    print(f"  Downloading one {lang} sample from IndicVoices-R...")
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np

        # Stream just 1 sample — no large download
        dataset = load_dataset(
            f"PharynxAI/IndicVoices-{_lang_to_name(lang)}-2000",
            split="train",
            streaming=True,
        )
        sample = next(iter(dataset))
        audio = sample["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr  = audio["sampling_rate"]

        sf.write(out_path, arr, sr)
        print(f"  Saved to: {out_path}  ({len(arr)/sr:.2f}s @ {sr}Hz)")
        return out_path

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print("  Falling back to synthetic audio for shape validation.")
        return None


def _lang_to_name(lang: str) -> str:
    mapping = {
        "hi": "hindi", "ta": "tamil", "te": "telugu",
        "kn": "kannada", "ml": "malayalam", "bn": "bengali",
        "mr": "marathi", "gu": "gujarati", "or": "odia", "pa": "punjabi",
    }
    return mapping.get(lang, "hindi")


def run_test(backend: str, wav: torch.Tensor, lang: str,
             audio_path: str, duration_s: float):
    print(f"\n{'─'*60}")
    print(f"  Backend  : {backend}")
    print(f"  Language : {lang}")
    print(f"  Audio    : {os.path.basename(audio_path) if audio_path else 'synthetic'}")
    print(f"  Duration : {duration_s:.2f}s  ({wav.shape[-1]} samples @ {TARGET_SR}Hz)")
    print(f"{'─'*60}")

    cfg = ContentEncoderConfig(
        backend=backend,
        chunk_size_ms=200,
        lookahead_ms=40,
        output_dim=1024,
        freeze_encoder=True,
        languages=[lang],
    )

    print("\n[1/3] Loading encoder...")
    t0 = time.time()
    enc = ContentEncoder(cfg)
    load_time = time.time() - t0
    print(f"      Load time: {load_time:.1f}s")

    # ── Full utterance ──────────────────────────────────────────
    print("\n[2/3] Full utterance encoding...")
    # ContentEncoder expects (B, samples) — add batch dim
    wav_batch = wav  # already (1, samples)

    t0 = time.time()
    with torch.no_grad():
        features = enc(wav_batch, lang_id=lang)
    elapsed = time.time() - t0

    B, T, D = features.shape
    real_time_factor = duration_s / elapsed
    print(f"      Input  : ({B}, {wav.shape[-1]})  →  {duration_s:.2f}s audio")
    print(f"      Output : ({B}, {T}, {D})")
    print(f"      T frames: {T}  ({duration_s/T*1000:.1f}ms per frame)")
    print(f"      Encoding time : {elapsed*1000:.0f}ms")
    print(f"      Real-time factor: {real_time_factor:.2f}x  "
          f"({'faster' if real_time_factor > 1 else 'slower'} than real-time)")
    print(f"      Feature stats — mean: {features.mean():.4f}  "
          f"std: {features.std():.4f}  "
          f"max: {features.abs().max():.4f}")
    assert not torch.isnan(features).any(), "❌ NaN in features!"
    assert not torch.isinf(features).any(), "❌ Inf in features!"
    print(f"      ✅ No NaN/Inf")

    # ── Streaming chunks ────────────────────────────────────────
    print("\n[3/3] Streaming simulation (200ms chunks)...")
    chunk_samples = 200 * TARGET_SR // 1000  # 3200 samples
    total_samples = wav.shape[-1]
    chunks = [
        wav[:, i:i+chunk_samples]
        for i in range(0, total_samples, chunk_samples)
        if wav[:, i:i+chunk_samples].shape[-1] == chunk_samples
    ]

    if not chunks:
        print("      ⚠️  Audio too short for streaming test (need >200ms)")
    else:
        print(f"      Processing {len(chunks)} chunks of 200ms each...")
        all_chunk_features = []
        chunk_times = []

        for idx, chunk in enumerate(chunks):
            t0 = time.time()
            with torch.no_grad():
                cf = enc.stream(chunk, lang_id=lang)
            chunk_times.append(time.time() - t0)
            all_chunk_features.append(cf)

        avg_chunk_ms = sum(chunk_times) / len(chunk_times) * 1000
        max_chunk_ms = max(chunk_times) * 1000
        total_chunk_frames = sum(cf.shape[1] for cf in all_chunk_features)

        print(f"      Chunks processed    : {len(chunks)}")
        print(f"      Avg chunk latency   : {avg_chunk_ms:.1f}ms")
        print(f"      Max chunk latency   : {max_chunk_ms:.1f}ms")
        print(f"      Total frames (stream): {total_chunk_frames}")
        print(f"      Total frames (full)  : {T}")

        latency_ok = avg_chunk_ms < 300
        print(f"      Latency {'✅ OK (<300ms)' if latency_ok else '⚠️  >300ms (CPU expected)'}")

    print(f"\n✅  {backend} passed on real audio\n")
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None,
                        help="Path to a .wav file. If not provided, downloads a sample.")
    parser.add_argument("--backend", default="both",
                        choices=["indicwhisper", "indicconformer", "both"])
    parser.add_argument("--lang", default="hi")
    args = parser.parse_args()

    print(f"\nIndicVC — Real Audio Content Encoder Test")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")

    # ── Get audio ───────────────────────────────────────────────
    audio_path = args.audio
    if audio_path is None:
        print(f"\nNo --audio provided. Downloading sample for lang='{args.lang}'...")
        audio_path = download_sample_audio(args.lang)

    if audio_path and os.path.exists(audio_path):
        print(f"\nLoading audio: {audio_path}")
        wav = load_audio(audio_path)
        duration_s = wav.shape[-1] / TARGET_SR
        print(f"Loaded: {wav.shape}  duration={duration_s:.2f}s  sr={TARGET_SR}Hz")
    else:
        print("\nUsing 3-second synthetic audio (real download failed)")
        wav = torch.randn(1, TARGET_SR * 3) * 0.1
        duration_s = 3.0
        audio_path = "synthetic"

    # ── Run tests ───────────────────────────────────────────────
    backends = (
        ["indicwhisper", "indicconformer"]
        if args.backend == "both"
        else [args.backend]
    )

    results = {}
    all_features = {}
    for backend in backends:
        try:
            feat = run_test(backend, wav, args.lang, audio_path, duration_s)
            results[backend] = True
            all_features[backend] = feat
        except Exception as e:
            import traceback
            print(f"\n❌  {backend} FAILED: {e}")
            traceback.print_exc()
            results[backend] = False

    # ── Compare backends if both ran ────────────────────────────
    if len(all_features) == 2:
        print("\n" + "="*60)
        print("  Backend Comparison")
        print("="*60)
        f1 = all_features["indicwhisper"]
        f2 = all_features["indicconformer"]
        print(f"  IndicWhisper   output: {tuple(f1.shape)}")
        print(f"  IndicConformer output: {tuple(f2.shape)}")
        print(f"  Time frames — Whisper: {f1.shape[1]}  Conformer: {f2.shape[1]}")
        print(f"  Frame rate  — Whisper: {duration_s/f1.shape[1]*1000:.1f}ms/frame  "
              f"Conformer: {duration_s/f2.shape[1]*1000:.1f}ms/frame")
        print(f"\n  NOTE: Different frame rates = different temporal resolution.")
        print(f"  Both output 1024-dim features after projection.")
        print(f"  Which gives better VC quality is your ablation result.")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for backend, passed in results.items():
        print(f"  {backend:20s} {'✅ PASS' if passed else '❌ FAIL'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()