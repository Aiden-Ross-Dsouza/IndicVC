"""
test_content_encoder.py
=======================
Smoke test for the IndicVC content encoder.
Run this FIRST after cloning the repo and installing dependencies
to verify your environment and GPU are set up correctly.

Usage
-----
    python test_content_encoder.py                          # test both backends
    python test_content_encoder.py --backend indicwhisper   # whisper only
    python test_content_encoder.py --backend indicconformer # conformer only
    python test_content_encoder.py --lang ta                # Tamil
"""

import argparse
import sys
import time

import torch

# Allow running from repo root
sys.path.insert(0, ".")
from modules.content_encoder import ContentEncoderConfig, ContentEncoder


def test_encoder(backend: str, lang: str, device: str):
    print(f"\n{'='*60}")
    print(f"  Backend : {backend}")
    print(f"  Language: {lang}")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    if device == "cpu":
        print("\n⚠️  WARNING: Running on CPU. This will be slow.")
        print("   On Quadro 6000, move your repo there and run with --device cuda")
        print("   Continuing anyway for validation...\n")

    cfg = ContentEncoderConfig(
        backend=backend,
        chunk_size_ms=200,
        lookahead_ms=40,
        output_dim=1024,
        freeze_encoder=True,   # Freeze for inference test
        languages=[lang],
    )

    print("[1/4] Building ContentEncoder...")
    enc = ContentEncoder(cfg).to(device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"      Total params (projection + embeddings): {n_params:,}")

    # Simulate 2-second audio at 16kHz
    B, samples = 2, 32000
    waveform = torch.randn(B, samples).to(device)
    print(f"\n[2/4] Running full-utterance forward pass...")
    print(f"      Input  shape: {waveform.shape}  (2s audio @ 16kHz)")

    t0 = time.time()
    with torch.no_grad():
        out = enc(waveform, lang_id=lang)
    elapsed = time.time() - t0

    # T_real is variable — 2s audio → ~200 mel frames → ~100 encoder frames
    # after Whisper's 2x conv downsampling. Exact value depends on padding.
    print(f"      Output shape: {out.shape}  — expected (2, T_real, 1024)")
    print(f"      T_real (encoder frames for 2s audio): {out.shape[1]}")
    print(f"      Forward pass time: {elapsed*1000:.1f}ms")

    assert out.shape[0] == B,    f"Batch dim wrong: {out.shape[0]} != {B}"
    assert out.shape[2] == 1024, f"Feature dim wrong: {out.shape[2]} != 1024"
    assert out.shape[1] > 0,     "Zero time frames in output!"
    assert out.shape[1] <= 1500, f"T_real {out.shape[1]} > max Whisper frames 1500"
    assert not torch.isnan(out).any(), "NaN detected in output!"
    print("      ✅ Shape and NaN checks passed")

    # Streaming test — single 200ms chunk
    chunk_samples = 200 * 16   # 200ms @ 16kHz = 3200 samples
    chunk = torch.randn(1, chunk_samples).to(device)
    print(f"\n[3/4] Running streaming (single 200ms chunk) forward pass...")
    print(f"      Chunk input shape: {chunk.shape}")

    t0 = time.time()
    with torch.no_grad():
        chunk_out = enc.stream(chunk, lang_id=lang)
    elapsed = time.time() - t0

    print(f"      Chunk output shape: {chunk_out.shape}")
    print(f"      Chunk T_real: {chunk_out.shape[1]} encoder frames")
    print(f"      Latency: {elapsed*1000:.1f}ms", end="  ")

    # Latency check only meaningful on GPU
    if device == "cuda":
        if elapsed < 0.3:
            print("✅ Under 300ms target")
        else:
            print(f"⚠️  Over 300ms target (expected on first run due to CUDA warmup)")
    else:
        print("(CPU — latency check skipped)")

    assert chunk_out.shape[0] == 1,    "Batch dim wrong for streaming"
    assert chunk_out.shape[2] == 1024, "Feature dim wrong for streaming"
    assert chunk_out.shape[1] > 0,     "Zero frames in chunk output"

    # Cross-lingual test
    other_lang = "ta" if lang == "hi" else "hi"
    print(f"\n[4/4] Cross-lingual feature test ({lang} encoder vs {other_lang} encoder)...")
    cfg2 = ContentEncoderConfig(
        backend=backend,
        languages=[lang, other_lang],
        freeze_encoder=True,
    )
    enc2 = ContentEncoder(cfg2).to(device)
    with torch.no_grad():
        out_src = enc2(waveform, lang_id=lang)
        out_tgt = enc2(waveform, lang_id=other_lang)

    print(f"      {lang:2s} encoder output: {out_src.shape}")
    print(f"      {other_lang:2s} encoder output: {out_tgt.shape}")

    # Align time dims for comparison (may differ due to different real frame counts)
    T = min(out_src.shape[1], out_tgt.shape[1])
    diff = (out_src[:, :T] - out_tgt[:, :T]).abs().mean().item()
    print(f"      Mean abs diff between language outputs: {diff:.6f}")
    print(f"      (Should be > 0 — family embedding creates distinction)")
    assert diff > 0, "Cross-lingual outputs identical — family embedding not working!"

    print(f"\n✅  All tests passed for backend='{backend}', lang='{lang}'\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["indicwhisper", "indicconformer", "both"],
                        default="indicwhisper")
    parser.add_argument("--lang", default="hi",
                        help="BCP-47 language code, e.g. hi, ta, te")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\nIndicVC Content Encoder — Smoke Test")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    backends = (
        ["indicwhisper", "indicconformer"]
        if args.backend == "both"
        else [args.backend]
    )

    if "indicconformer" in backends:
        print("\n⚠️  NOTE: IndicConformer-600M download is ~2.4GB on first run.")
        print("   Ensure you have enough disk space and a stable connection.")
        print("   The model will be cached at ~/.cache/huggingface after download.\n")

    results = {}
    for backend in backends:
        try:
            results[backend] = test_encoder(backend, args.lang, args.device)
        except Exception as e:
            print(f"\n❌  {backend} FAILED: {e}")
            results[backend] = False

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for backend, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {backend:20s} {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
