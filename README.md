# IndicVC — Zero-Shot Streaming Cross-Lingual Emotional Voice Conversion for Indic Languages

> **Target venue:** ICASSP 2027  
> **Hardware:** NVIDIA Quadro RTX 6000 (24 GB VRAM)  
> **Status:** Phase 1 — Speaker Encoder fine-tuning

---

## What This Is

IndicVC converts a speaker's voice to sound like a target speaker while preserving the **source speaker's emotion** and **spoken content**, working across 10 Indian languages without needing paired training data (zero-shot). It streams output in real-time with under 300 ms latency.

### Why It's Hard

Standard voice conversion systems fail on Indic languages for three reasons:

1. **Script and phoneme diversity** — 10 languages across two unrelated families (Indo-Aryan and Dravidian) have fundamentally different phoneme inventories and prosodic patterns.
2. **Emotion in Indic speech** — Indic emotional expression differs from the Western speech corpora that existing emotion recognisers and VC decoders were trained on.
3. **Zero-shot + streaming simultaneously** — Most VC systems either require target speaker fine-tuning *or* are causal/streaming — not both.

### Five Novel Contributions

| # | Contribution | Where |
|---|---|---|
| 1 | Causal IndicWhisper content encoder (chunk-wise causal attention) | `modules/content_encoder.py` |
| 2 | Indic ECAPA-TDNN speaker encoder fine-tuned on IndicVoices-R | `modules/speaker_encoder.py` |
| 3 | emotion2vec fine-tuned on AI4Bharat Rasa (12 Indic emotions) | `modules/emotion_encoder.py` *(upcoming)* |
| 4 | Language-family MoE (Mixture-of-Experts) decoder | `modules/moe_router.py` *(upcoming)* |
| 5 | Phoneme-boundary-aware streaming chunking | `modules/content_encoder.py` |

---

## Architecture

```
SOURCE AUDIO (any of 10 Indic languages)
        │
        ├──► Content Encoder (IndicWhisper or IndicConformer)
        │       Extracts WHAT is being said — language-agnostic tokens
        │       Causal attention for streaming (chunk_size=200ms, lookahead=40ms)
        │
        ├──► Speaker Encoder (ECAPA-TDNN)
        │       Extracts WHO is speaking — 192-dim L2-normalised embedding
        │       Pre-trained on VoxCeleb2, fine-tuned on IndicVoices-R
        │
        └──► Emotion Encoder (emotion2vec)
                Extracts HOW it's being said — 256-dim emotion embedding
                Fine-tuned on AI4Bharat Rasa (12 Indic emotions)

TARGET SPEAKER AUDIO
        │
        └──► Speaker Encoder ──► target speaker embedding

All three encodings + target speaker embedding
        │
        ▼
Language-Family MoE Router
        Soft-routes features through Indo-Aryan expert or Dravidian expert
        Learns language-family-specific prosodic mapping
        │
        ▼
Streaming DiT Decoder  (initialised from Seed-VC)
        Diffusion Transformer conditioned on:
          - content tokens (from source)
          - target speaker embedding
          - source emotion embedding
        Generates target mel-spectrogram frame-by-frame
        │
        ▼
BigVGAN Vocoder
        Converts mel-spectrogram → waveform
        │
        ▼
OUTPUT AUDIO
  Same content + same emotion as source
  Voice of target speaker
  Language of source speaker
```

---

## Supported Languages

| Code | Language | Family | Script |
|------|----------|--------|--------|
| `hi` | Hindi | Indo-Aryan | Devanagari |
| `mr` | Marathi | Indo-Aryan | Devanagari |
| `bn` | Bengali | Indo-Aryan | Bengali |
| `gu` | Gujarati | Indo-Aryan | Gujarati |
| `pa` | Punjabi | Indo-Aryan | Gurmukhi |
| `or` | Odia | Indo-Aryan | Odia |
| `ta` | Tamil | Dravidian | Tamil |
| `kn` | Kannada | Dravidian | Kannada |
| `te` | Telugu | Dravidian | Telugu |
| `ml` | Malayalam | Dravidian | Malayalam |

**4-language pilot** (current phase): `hi`, `mr`, `ta`, `kn`  
**Full scale** (later): all 10 above, and potentially all 22 scheduled languages.

---

## File Structure

```
IndicVC/
│
├── README.md                          ← This file
│
├── config/
│   └── model.yaml                     ← Full model hyperparameter config
│
├── modules/                           ← Core neural network modules (all tested)
│   ├── __init__.py
│   ├── content_encoder.py             ✅ Complete — IndicWhisper + IndicConformer
│   ├── speaker_encoder.py             ✅ Complete — ECAPA-TDNN, pretrained transfer 73%
│   ├── emotion_encoder.py             🔲 Upcoming — emotion2vec + Rasa fine-tune
│   ├── moe_router.py                  🔲 Upcoming — language-family Mixture of Experts
│   └── dit_decoder.py                 🔲 Upcoming — Streaming DiT, init from Seed-VC
│
├── training/                          ← Training scripts
│   ├── __init__.py
│   ├── speaker_dataset.py             ✅ Complete — Dataset, SpeakerBatchSampler, augmentation
│   └── train_speaker_encoder.py       ✅ Complete — AAM-Softmax, W&B, crash recovery
│
├── tools/                             ← Utilities
│   ├── download_data.py               ✅ Complete — PharynxAI + AI4Bharat full download
│   ├── inspect_conformer.py           ✅ Diagnostic — IndicConformer architecture inspector
│   ├── inspect_conformer_deep.py      ✅ Diagnostic — ONNX backend inspector
│   ├── inspect_ckpt_keys.py           ✅ Diagnostic — SpeechBrain checkpoint key mapper
│   └── read_conformer_source.py       ✅ Diagnostic — Source reader for ONNX model
│
├── tests/                             ← Smoke tests (run before any training)
│   ├── test_content_encoder.py        ✅ Tests both backends, streaming, cross-lingual
│   ├── test_speaker_encoder.py        ✅ Tests build, forward, similarity, loss, batch
│   ├── test_real_audio.py             ✅ Tests content encoder on real .wav files
│   └── test_speaker_real_audio.py     ✅ Tests speaker encoder on real .wav files
│
├── data/                              ← Created by download_data.py (gitignored)
│   ├── audio/
│   │   ├── hi/                        ← {lang}_{index:06d}.wav
│   │   ├── mr/
│   │   ├── ta/
│   │   └── kn/
│   └── manifests/
│       ├── hi_train.csv               ← utt_id, speaker_id, language, family,
│       ├── hi_val.csv                 │  audio_path, duration_s, text, emotion
│       ├── mr_train.csv
│       ├── mr_val.csv
│       ├── ta_train.csv
│       ├── ta_val.csv
│       ├── kn_train.csv
│       ├── kn_val.csv
│       ├── all_train.csv              ← All languages combined
│       └── all_val.csv
│
└── checkpoints/                       ← Created during training (gitignored)
    └── speaker/
        ├── last.pt                    ← Saved every epoch — safe crash recovery
        ├── best.pt                    ← Best val EER checkpoint
        ├── speaker_encoder_epoch005.pt
        ├── speaker_encoder_epoch010.pt
        ├── backbone_final.pt          ← Backbone only (no AAM head) for VC pipeline
        └── history.json               ← Full training log (loss, EER per epoch)
```

---

## Training Pipeline (Phase by Phase)

### Phase 0: Environment Setup

```bash
# GPU check
nvidia-smi

# Install dependencies
pip install torch==2.6.0+cu121 torchaudio==2.6.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
pip install datasets soundfile transformers huggingface_hub \
    safetensors scikit-learn wandb umap-learn matplotlib

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Phase 1: Data Download

```bash
# Fast path — PharynxAI pre-sliced subsets (no token needed, ~15h/language)
python tools/download_data.py \
    --mode pharynx \
    --langs hi mr ta kn \
    --out_dir data

# Full dataset — requires HF token with AI4Bharat access approved
# Request access at: https://huggingface.co/datasets/ai4bharat/indicvoices_r
python tools/download_data.py \
    --mode full \
    --langs hi mr ta kn \
    --hf_token YOUR_TOKEN \
    --out_dir data
```

**Data requirements per language for speaker encoder:**

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Quick iteration | 5h / 50 speakers | PharynxAI subsets |
| Paper results | 50h / 200+ speakers | Full IndicVoices-R |

### Phase 2: Smoke Tests

```bash
# Content encoder — both backends
python tests/test_content_encoder.py --backend indicconformer --lang hi --device cuda
python tests/test_real_audio.py --audio YOUR.wav --backend both --lang hi --device cuda

# Speaker encoder
python tests/test_speaker_encoder.py --pretrained
python tests/test_speaker_real_audio.py --audio YOUR.wav --pretrained --device cuda
```

### Phase 3: Speaker Encoder Fine-tuning

```bash
# W&B login (once)
wandb login

# Train
python training/train_speaker_encoder.py \
    --train_manifest data/manifests/all_train.csv \
    --val_manifest   data/manifests/all_val.csv \
    --langs hi mr ta kn \
    --pretrained \
    --device cuda \
    --epochs 30 \
    --speakers_per_batch 32 \
    --utts_per_speaker 2 \
    --wandb \
    --wandb_project indicvc \
    --ckpt_dir checkpoints/speaker

# If it crashes or is interrupted — auto-resume picks up last.pt:
python training/train_speaker_encoder.py \
    --train_manifest data/manifests/all_train.csv \
    --val_manifest   data/manifests/all_val.csv \
    --auto_resume \
    --wandb \
    --device cuda
```

**What to watch on W&B:**
- `train/loss` — should decrease steadily, not oscillate wildly
- `val/EER` — target < 5%. If plateauing > 10% after epoch 10, reduce LR
- `val/sim_same_speaker` vs `val/sim_diff_speaker` histograms — the two distributions should separate over training
- `val/embedding_umap` — clusters should form by language/speaker (logged every 5 epochs)
- `train/grad_norm` — should stay in range 0.1–5.0; spikes indicate instability

**Expected training time on Quadro RTX 6000:**  
~8–12 hours for 30 epochs on 4 languages × ~15h audio each.

### Phase 4: Emotion Encoder *(upcoming)*

```bash
# Downloads AI4Bharat Rasa and fine-tunes emotion2vec
python training/train_emotion_encoder.py \
    --device cuda --wandb
```

### Phase 5: Joint Training *(upcoming)*

```bash
# MoE router + DiT decoder, all encoders frozen initially
python training/train_joint.py \
    --speaker_ckpt checkpoints/speaker/backbone_final.pt \
    --device cuda --wandb
```

---

## Checkpoint Recovery

The training script is designed for preemptible cluster environments:

| File | Written when | Contains |
|------|-------------|---------|
| `last.pt` | After **every epoch** (atomic write) | Full state — model, optimizer, schedulers, history, W&B run ID |
| `best.pt` | When val EER improves | Same as last.pt |
| `speaker_encoder_epochNNN.pt` | Every `--save_every` epochs | Same as last.pt |
| `backbone_final.pt` | End of training | Backbone weights only (for inference) |

**Atomic write:** `last.pt` is written via a temp file + `os.replace()`, so a mid-write crash leaves the previous `last.pt` intact.

**Signal handling:** `SIGINT` (Ctrl-C) and `SIGTERM` (cluster pre-emption) are caught. The current epoch completes, `last.pt` is saved, then the process exits cleanly. On resume, training continues from the next epoch.

**W&B run continuity:** The W&B run ID is stored in every checkpoint. On resume, the same run is continued — all metrics appear in one continuous timeline on the W&B dashboard.

---

## Evaluation Metrics (for paper)

| Metric | Measures | Target |
|--------|----------|--------|
| Speaker Similarity (SIM) | Cosine sim of ECAPA embeddings | > 0.85 |
| Word Error Rate (WER) | ASR on converted audio (IndicWhisper) | < 15% |
| Emotion Accuracy (EA) | emotion2vec on converted audio | > 70% |
| MOS | Naturalness (human eval, native speakers) | > 3.5 |
| Streaming Latency | Time to first output chunk | < 300ms |

---

## Baselines

| System | Type | Languages |
|--------|------|-----------|
| FreeVC | VC, English-only | En |
| MulliVC | Cross-lingual VC | En + limited |
| Seed-VC | Zero-shot VC (SOTA) | En + Zh |
| **IndicVC (ours)** | Zero-shot streaming emotional VC | 10 Indic |

Ablations: w/o MoE router, w/o emotion conditioning, w/o IndicVoices-R fine-tuning, IndicConformer vs IndicWhisper backend.

---

## Key References

- **Seed-VC**: arXiv:2411.09943 — base architecture for DiT decoder
- **IndicVoices-R**: NeurIPS 2024, AI4Bharat — primary training dataset
- **ECAPA-TDNN**: Desplanques et al., Interspeech 2020 — speaker encoder architecture
- **emotion2vec**: Ma et al., ACL 2024 — emotion encoder base
- **AAM-Softmax (ArcFace)**: Deng et al., CVPR 2019 — speaker training objective
- **BigVGAN**: Lee et al., ICLR 2023 — neural vocoder