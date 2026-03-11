"""
inspect_ckpt_keys.py
====================
Prints SpeechBrain checkpoint keys alongside our backbone keys
so we can write the correct key mapping.

Run:
    python tools/inspect_ckpt_keys.py
"""
import sys
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from modules.speaker_encoder import SpeakerEncoderConfig, ECAPATDNNBackbone

# Load checkpoint
print("Loading checkpoint...")
ckpt_path = hf_hub_download(
    repo_id="speechbrain/spkrec-ecapa-voxceleb",
    filename="embedding_model.ckpt",
)
raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "model" in raw:
    sb_state = raw["model"]
elif isinstance(raw, dict) and "embedding_model" in raw:
    sb_state = raw["embedding_model"]
else:
    sb_state = raw

# Build our backbone
cfg = SpeakerEncoderConfig(pretrained_model="")
backbone = ECAPATDNNBackbone(cfg)
our_state = backbone.state_dict()

print(f"\nCheckpoint has {len(sb_state)} keys")
print(f"Our backbone has {len(our_state)} keys")

print("\n" + "="*70)
print("CHECKPOINT KEYS (first 60):")
print("="*70)
for i, (k, v) in enumerate(sb_state.items()):
    print(f"  {k:60s}  {tuple(v.shape)}")
    if i >= 59:
        print(f"  ... and {len(sb_state)-60} more")
        break

print("\n" + "="*70)
print("OUR BACKBONE KEYS (first 60):")
print("="*70)
for i, (k, v) in enumerate(our_state.items()):
    print(f"  {k:60s}  {tuple(v.shape)}")
    if i >= 59:
        print(f"  ... and {len(our_state)-60} more")
        break

# Find shape matches regardless of key name
print("\n" + "="*70)
print("SHAPE-BASED MATCHES (our_key → sb_key):")
print("="*70)
sb_by_shape = {}
for k, v in sb_state.items():
    shape = tuple(v.shape)
    sb_by_shape.setdefault(shape, []).append(k)

matched = 0
for our_key, our_v in our_state.items():
    shape = tuple(our_v.shape)
    if shape in sb_by_shape:
        candidates = sb_by_shape[shape]
        print(f"  {our_key:50s} ← {candidates[0]}  {shape}")
        matched += 1

print(f"\nTotal shape matches: {matched}/{len(our_state)}")
