"""
inspect_conformer_deep.py
=========================
Deep inspector for IndicConformer-600M to find exactly how to extract
encoder hidden states.

Probes:
  1. model.get_encoder() — what does it return?
  2. model.models — what's inside?
  3. Forward pass with dummy audio — what does the model actually output?
  4. Register hooks on every child and run forward — which ones fire?

Run:
    python tools/inspect_conformer_deep.py

Paste the full output back to continue building the encoder.
"""

import sys
import torch
import numpy as np

INDICCONFORMER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("=" * 60)
print("IndicConformer Deep Inspector")
print("=" * 60)

# ── 1. Load model ──────────────────────────────────────────────
print("\n[1/6] Loading model...")
from transformers import AutoModel
model = AutoModel.from_pretrained(
    INDICCONFORMER_MODEL_ID,
    trust_remote_code=True,
)
print("      ✅ Loaded")

# ── 2. Probe get_encoder() ─────────────────────────────────────
print("\n[2/6] Calling model.get_encoder()...")
try:
    enc = model.get_encoder()
    print(f"      Type         : {type(enc)}")
    print(f"      Is nn.Module : {isinstance(enc, torch.nn.Module)}")
    if isinstance(enc, torch.nn.Module):
        n_params = sum(p.numel() for p in enc.parameters())
        print(f"      Param count  : {n_params:,}")
        print(f"      Children     :")
        for name, child in enc.named_children():
            cp = sum(p.numel() for p in child.parameters())
            print(f"        {name:30s}  {cp:>12,} params")
    else:
        print(f"      Value: {enc}")
except Exception as e:
    print(f"      ❌ get_encoder() failed: {e}")
    enc = None

# ── 3. Probe model.models ──────────────────────────────────────
print("\n[3/6] Probing model.models attribute...")
try:
    models_attr = model.models
    print(f"      Type  : {type(models_attr)}")
    if isinstance(models_attr, (list, dict)):
        print(f"      Length: {len(models_attr)}")
        items = models_attr.items() if isinstance(models_attr, dict) else enumerate(models_attr)
        for k, v in items:
            print(f"      [{k}] {type(v).__name__}")
            if isinstance(v, torch.nn.Module):
                for cname, child in v.named_children():
                    cp = sum(p.numel() for p in child.parameters())
                    print(f"          {cname:30s}  {cp:>12,} params")
    elif isinstance(models_attr, torch.nn.Module):
        for cname, child in models_attr.named_children():
            cp = sum(p.numel() for p in child.parameters())
            print(f"      {cname:30s}  {cp:>12,} params")
    else:
        print(f"      Value: {models_attr}")
except Exception as e:
    print(f"      ❌ model.models failed: {e}")

# ── 4. Probe model.__dict__ for hidden nn.Modules ──────────────
print("\n[4/6] Scanning model.__dict__ for nn.Module attributes...")
for attr_name, attr_val in model.__dict__.items():
    if attr_name.startswith("_"):
        continue
    if isinstance(attr_val, torch.nn.Module):
        n = sum(p.numel() for p in attr_val.parameters())
        print(f"      model.{attr_name:30s} nn.Module  {n:>12,} params")
        for cname, child in attr_val.named_children():
            cp = sum(p.numel() for p in child.parameters())
            print(f"          .{cname:30s}  {cp:>12,} params")
    elif isinstance(attr_val, (list, dict)) and len(str(attr_val)) < 200:
        print(f"      model.{attr_name:30s} {type(attr_val).__name__:10s}  {attr_val}")

# ── 5. Register hooks on ALL sub-modules, run forward ──────────
print("\n[5/6] Registering hooks on all sub-modules and running forward pass...")
print("      (This reveals which modules actually fire during inference)")

hook_outputs = {}
hooks = []

def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            hook_outputs[name] = out.shape
        elif isinstance(out, (tuple, list)):
            shapes = [o.shape if isinstance(o, torch.Tensor) else type(o).__name__
                      for o in out]
            hook_outputs[name] = shapes
        else:
            hook_outputs[name] = type(out).__name__
    return hook

# Register on every named module
try:
    for name, mod in model.named_modules():
        if name == "":
            continue
        h = mod.register_forward_hook(make_hook(name))
        hooks.append(h)
    print(f"      Registered {len(hooks)} hooks")
except Exception as e:
    print(f"      Hook registration warning: {e}")

# Create 1-second dummy waveform
dummy_wav = torch.randn(1, 16000)

print("      Running forward pass with 1s dummy audio (lang=hi, mode=ctc)...")
try:
    with torch.no_grad():
        result = model(dummy_wav, "hi", "ctc")
    print(f"      Forward pass result type: {type(result)}")
    print(f"      Forward pass result: {str(result)[:200]}")
except Exception as e:
    print(f"      ❌ Forward pass failed: {e}")
    print(f"         Error type: {type(e).__name__}")

# Remove hooks
for h in hooks:
    h.remove()

print(f"\n      Modules that fired during forward pass:")
if hook_outputs:
    for name, shape in hook_outputs.items():
        print(f"        {name:50s}  output: {shape}")
else:
    print("      ⚠️  No hooks fired — model may use ONNX runtime internally")
    print("         This means hidden states are NOT accessible via PyTorch hooks.")
    print("         We need to use the model differently.")

# ── 6. Check if model uses ONNX runtime ───────────────────────
print("\n[6/6] Checking for ONNX runtime usage...")
model_source = ""
try:
    import inspect
    source_file = inspect.getfile(type(model))
    print(f"      Model source file: {source_file}")
    with open(source_file) as f:
        model_source = f.read()
    print(f"      Source file size: {len(model_source)} chars")
    # Check for ONNX indicators
    onnx_indicators = ["onnxruntime", "InferenceSession", ".onnx", "ort."]
    for indicator in onnx_indicators:
        if indicator in model_source:
            print(f"      ⚠️  Found ONNX indicator: '{indicator}'")
    # Print the forward() method source
    print("\n      model.forward() source:")
    print("      " + "-"*50)
    in_forward = False
    indent_level = 0
    for line in model_source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def forward("):
            in_forward = True
            indent_level = len(line) - len(line.lstrip())
        if in_forward:
            print(f"      {line}")
            # Stop after the function ends (next def at same indent)
            if stripped.startswith("def ") and not stripped.startswith("def forward") and \
               len(line) - len(line.lstrip()) <= indent_level and stripped != "":
                break
except Exception as e:
    print(f"      Could not inspect source: {e}")

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print("  Paste this complete output back to get the correct")
print("  IndicConformerEncoder implementation.")
print("="*60)
