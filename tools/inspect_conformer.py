"""
tools/inspect_conformer.py
==========================
Diagnostic tool — run this ONCE to understand the internal structure
of the IndicConformer-600M model so we can hook into it correctly.

This script does NOT modify any files. It just prints information.

Usage
-----
    python tools/inspect_conformer.py

Output tells us:
  1. What attributes the AutoModel wrapper has
  2. Whether the NeMo model is accessible
  3. Where the encoder hidden states can be extracted from
  4. What the correct hook path should be
"""

import sys
import torch
from transformers import AutoModel

print("=" * 60)
print("IndicConformer Architecture Inspector")
print("=" * 60)

print("\n[1/5] Loading model (uses cached download)...")
try:
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    print("      ✅ Model loaded")
except Exception as e:
    print(f"      ❌ Failed to load: {e}")
    sys.exit(1)

print(f"\n[2/5] Top-level model type: {type(model)}")
print(f"      Module class: {model.__class__.__name__}")
print(f"      Module MRO: {[c.__name__ for c in type(model).__mro__[:5]]}")

print("\n[3/5] Top-level attributes (non-dunder):")
attrs = [a for a in dir(model) if not a.startswith("__")]
# Filter to likely-useful ones
interesting = [a for a in attrs if any(k in a.lower() for k in
    ["model", "encoder", "nemo", "conformer", "asr", "net", "backbone",
     "feature", "forward", "audio", "wav", "speech"])]
print(f"      Interesting attrs: {interesting}")

print("\n[4/5] Named children (nn.Module children):")
children = list(model.named_children())
if children:
    for name, mod in children:
        n_params = sum(p.numel() for p in mod.parameters())
        print(f"      {name:30s} | {type(mod).__name__:30s} | {n_params:>12,} params")
else:
    print("      ⚠️  No named_children() found — model is not a standard nn.Module container")
    print("      This is expected for NeMo-backed AutoModel wrappers.")

print("\n[5/5] Searching for NeMo model inside attributes...")
nemo_model = None
for attr_name in ["model", "_model", "nemo_model", "asr_model", "net",
                   "_nemo_model", "conformer", "encoder"]:
    if hasattr(model, attr_name):
        attr = getattr(model, attr_name)
        print(f"      Found attr '{attr_name}': {type(attr).__name__}")
        # Check if it has NeMo-style sub-attributes
        sub_attrs = [a for a in dir(attr) if not a.startswith("__")]
        nemo_interesting = [a for a in sub_attrs if any(k in a.lower() for k in
            ["encoder", "decoder", "conformer", "preprocessor", "ctc", "rnnt"])]
        if nemo_interesting:
            print(f"        NeMo-style sub-attrs: {nemo_interesting}")
            nemo_model = attr
            break

if nemo_model is None:
    print("      No standard NeMo model attribute found.")
    print("      Trying to inspect __dict__ directly...")
    for k, v in model.__dict__.items():
        if not k.startswith("__") and hasattr(v, "parameters"):
            try:
                n = sum(p.numel() for p in v.parameters())
                print(f"      __dict__['{k}']: {type(v).__name__} — {n:,} params")
            except Exception:
                pass

if nemo_model is not None:
    print(f"\n  Found NeMo model: {type(nemo_model).__name__}")
    print("  NeMo model named_children:")
    for name, mod in nemo_model.named_children():
        n_params = sum(p.numel() for p in mod.parameters())
        print(f"    {name:30s} | {type(mod).__name__:30s} | {n_params:>12,} params")

print("\n" + "=" * 60)
print("  RECOMMENDATION")
print("=" * 60)
print("""
Based on this output, update INDICCONFORMER_MODEL_ID in content_encoder.py
and update _ENCODER_SUBMODULE_CANDIDATES with the correct attribute path.

If you see a NeMo model with an 'encoder' child — use: 'model.encoder'
If you see a NeMo model as '_model' with 'encoder' child — use: '_model.encoder'

Paste the full output of this script back to continue.
""")
