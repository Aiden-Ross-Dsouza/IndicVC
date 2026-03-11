"""
read_conformer_source.py
========================
Reads the cached IndicConformer ONNX model source with correct encoding
and prints the parts we need to understand the encoder call API.

Run:
    python tools/read_conformer_source.py
"""
import os
import glob

cache_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
matches = glob.glob(
    os.path.join(cache_base, "**", "model_onnx.py"), recursive=True
)

if not matches:
    print("❌ Could not find model_onnx.py in HuggingFace cache.")
    print(f"   Searched: {cache_base}")
    exit(1)

source_file = matches[0]
print(f"Found: {source_file}\n")

# Read with latin-1 which never fails (every byte is valid)
with open(source_file, encoding="latin-1") as f:
    source = f.read()

print(f"Total source length: {len(source)} chars\n")
print("=" * 60)

# Print full source — we need to see the forward() and encoder call
# Truncate to first 15000 chars to avoid overwhelming output
print(source[:15000])
if len(source) > 15000:
    print(f"\n... [truncated — {len(source)-15000} more chars] ...")
    print("\nLast 3000 chars:")
    print(source[-3000:])
