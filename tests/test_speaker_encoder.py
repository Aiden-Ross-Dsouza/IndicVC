"""
test_speaker_encoder.py
=======================
Smoke tests for the Indic ECAPA-TDNN speaker encoder.

Tests
-----
1. Build: instantiate SpeakerEncoder (with and without speechbrain pretrained)
2. Forward: waveform → embedding shape/dtype/normalisation checks
3. Similarity: same speaker > different speaker
4. AAM-Softmax loss: backward pass works
5. Streaming/short audio: padding to min_duration_s works
6. Batch consistency: batched == individual forward passes

Run
---
    python tests/test_speaker_encoder.py
    python tests/test_speaker_encoder.py --no-pretrained   # skip SpeechBrain download
"""

import argparse
import sys
import time
import traceback

# ── torchaudio compatibility patch ──────────────────────────────────────────
# Must run before ANY speechbrain import (including transitive imports).
# SpeechBrain 1.x calls torchaudio.list_audio_backends() at import time,
# which was removed in torchaudio 2.x. Patch it here at process startup.
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modules.speaker_encoder import SpeakerEncoderConfig, SpeakerEncoder

SR = 16000


def make_encoder(pretrained: bool = False) -> SpeakerEncoder:
    cfg = SpeakerEncoderConfig(
        pretrained_model="speechbrain/spkrec-ecapa-voxceleb" if pretrained else "",
        embedding_dim=192,
        n_speakers=64,          # Small for testing
        freeze_backbone=False,
    )
    return SpeakerEncoder(cfg)


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true",
                        help="Load SpeechBrain VoxCeleb pretrained weights (~80MB download)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("\nIndicVC Speaker Encoder — Smoke Test")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    print(f"Pretrained: {args.pretrained}")

    results = {}

    # ── Test 1: Build ────────────────────────────────────────────────────────
    section("Test 1/6 — Build SpeakerEncoder")
    try:
        t0 = time.time()
        enc = make_encoder(pretrained=args.pretrained)
        enc = enc.to(args.device)
        enc.eval()
        build_time = time.time() - t0

        n_backbone = sum(p.numel() for p in enc.backbone.parameters())
        n_total    = sum(p.numel() for p in enc.parameters())
        print(f"  Backbone params  : {n_backbone:>10,}")
        print(f"  Total params     : {n_total:>10,}")
        print(f"  Build time       : {build_time*1000:.0f}ms")
        print(f"  Pretrained loaded: {enc._pretrained_loaded}")
        print(f"  ✅ Build passed")
        results["build"] = True
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Build FAILED: {e}")
        results["build"] = False
        sys.exit(1)

    # ── Test 2: Forward pass ─────────────────────────────────────────────────
    section("Test 2/6 — Forward pass (2s audio, batch=2)")
    try:
        wav = torch.randn(2, SR * 2).to(args.device)  # 2s audio, batch=2
        t0 = time.time()
        with torch.no_grad():
            emb = enc(wav)
        elapsed = time.time() - t0

        print(f"  Input  : {tuple(wav.shape)}")
        print(f"  Output : {tuple(emb.shape)}")
        print(f"  Time   : {elapsed*1000:.0f}ms")
        assert emb.shape == (2, 192), f"Expected (2, 192), got {emb.shape}"
        assert not torch.isnan(emb).any(), "NaN in embeddings"
        assert not torch.isinf(emb).any(), "Inf in embeddings"

        # L2 norm check — must be ~1.0 (unit hypersphere)
        norms = emb.norm(dim=-1)
        print(f"  L2 norms: {norms.tolist()}  (should be ~1.0)")
        assert (norms - 1.0).abs().max() < 1e-5, f"Embeddings not unit-normalised: {norms}"

        print(f"  ✅ Forward pass passed")
        results["forward"] = True
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Forward FAILED: {e}")
        results["forward"] = False

    # ── Test 3: Similarity ───────────────────────────────────────────────────
    section("Test 3/6 — Speaker similarity")
    try:
        # Same speaker: two clips from same random speaker (same base wav + noise)
        spk_a_clip1 = torch.randn(1, SR * 3).to(args.device)
        spk_a_clip2 = spk_a_clip1 + torch.randn_like(spk_a_clip1) * 0.01  # tiny noise
        spk_b        = torch.randn(1, SR * 3).to(args.device)  # completely different

        with torch.no_grad():
            emb_a1 = enc(spk_a_clip1)
            emb_a2 = enc(spk_a_clip2)
            emb_b  = enc(spk_b)

        sim_same = enc.similarity(emb_a1, emb_a2).item()
        sim_diff = enc.similarity(emb_a1, emb_b).item()
        print(f"  Same speaker similarity : {sim_same:.4f}  (should be >> diff)")
        print(f"  Diff speaker similarity : {sim_diff:.4f}")
        print(f"  Gap                     : {sim_same - sim_diff:.4f}")

        # With random init, same-speaker sim > diff-speaker sim only if
        # the near-identical audio maps to similar embeddings — this is
        # a model sanity check, not a speaker verification threshold test
        assert sim_same > sim_diff, \
            f"Same-speaker ({sim_same:.3f}) should > diff-speaker ({sim_diff:.3f})"
        print(f"  ✅ Similarity check passed")
        results["similarity"] = True
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Similarity FAILED: {e}")
        results["similarity"] = False

    # ── Test 4: AAM-Softmax loss + backward ──────────────────────────────────
    section("Test 4/6 — AAM-Softmax loss + backward pass")
    try:
        enc.train()
        wav   = torch.randn(4, SR * 2).to(args.device)
        labels = torch.tensor([0, 1, 2, 3]).to(args.device)

        emb  = enc(wav)
        loss = enc.aam_softmax_loss(emb, labels)

        print(f"  Loss value : {loss.item():.4f}  (should be finite positive)")
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "NaN loss"

        loss.backward()
        # Check gradients flowed through backbone
        grad_norms = [
            p.grad.norm().item()
            for p in enc.backbone.parameters()
            if p.grad is not None
        ]
        print(f"  Grad tensors with grads : {len(grad_norms)}")
        print(f"  Avg grad norm           : {sum(grad_norms)/max(len(grad_norms),1):.6f}")
        assert len(grad_norms) > 0, "No gradients in backbone — backward failed"
        print(f"  ✅ Loss + backward passed")
        results["loss"] = True
        enc.eval()
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Loss FAILED: {e}")
        results["loss"] = False

    # ── Test 5: Short audio padding ──────────────────────────────────────────
    section("Test 5/6 — Short audio padding (200ms < 1s minimum)")
    try:
        short_wav = torch.randn(1, SR // 5).to(args.device)  # 200ms
        print(f"  Input duration : {short_wav.shape[-1]/SR*1000:.0f}ms  "
              f"({short_wav.shape[-1]} samples)")
        print(f"  Min duration   : {enc.cfg.min_duration_s*1000:.0f}ms")

        with torch.no_grad():
            emb = enc(short_wav)

        assert emb.shape == (1, 192), f"Wrong shape: {emb.shape}"
        assert not torch.isnan(emb).any(), "NaN in short-audio embedding"
        print(f"  Output : {tuple(emb.shape)}")
        print(f"  ✅ Short audio padding passed")
        results["short_audio"] = True
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Short audio FAILED: {e}")
        results["short_audio"] = False

    # ── Test 6: Batch consistency ─────────────────────────────────────────────
    section("Test 6/6 — Batch consistency (batched == individual)")
    try:
        wav1 = torch.randn(1, SR * 2).to(args.device)
        wav2 = torch.randn(1, SR * 2).to(args.device)
        wav_batch = torch.cat([wav1, wav2], dim=0)

        with torch.no_grad():
            emb1       = enc(wav1)
            emb2       = enc(wav2)
            emb_batch  = enc(wav_batch)

        diff1 = (emb_batch[0] - emb1[0]).abs().max().item()
        diff2 = (emb_batch[1] - emb2[0]).abs().max().item()
        print(f"  Max diff (item 0): {diff1:.2e}  (should be ~0)")
        print(f"  Max diff (item 1): {diff2:.2e}  (should be ~0)")
        # BatchNorm behaves differently in batch vs individual — allow small diff
        threshold = 0.05
        assert diff1 < threshold, f"Batch inconsistency too large: {diff1}"
        assert diff2 < threshold, f"Batch inconsistency too large: {diff2}"
        print(f"  ✅ Batch consistency passed")
        results["batch"] = True
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Batch consistency FAILED: {e}")
        results["batch"] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    all_pass = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test:25s} {status}")
        if not passed:
            all_pass = False

    if not all_pass:
        sys.exit(1)
    print(f"\n  All tests passed ✅")


if __name__ == "__main__":
    main()