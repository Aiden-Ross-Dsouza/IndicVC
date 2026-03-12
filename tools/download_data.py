"""
download_data.py
================
Downloads IndicVoices-R data for the 4-language pilot
(Hindi, Marathi, Tamil, Kannada) from both PharynxAI fast
subsets and optionally the full AI4Bharat gated dataset.

Outputs a manifest CSV per language:
    data/manifests/{lang}_train.csv
    data/manifests/{lang}_val.csv

Each row: utt_id, speaker_id, language, audio_path, duration_s, text

Usage
-----
    # Fast path — PharynxAI pre-sliced subsets (no HF token needed)
    python tools/download_data.py --mode pharynx --langs hi mr ta kn

    # Full gated dataset (requires HF token with ai4bharat/indicvoices_r access)
    python tools/download_data.py --mode full --langs hi mr ta kn --hf_token YOUR_TOKEN

    # Dry run — just print what would be downloaded
    python tools/download_data.py --mode pharynx --langs hi mr ta kn --dry_run
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHARYNX_REPOS = {
    "hi": "PharynxAI/IndicVoices-hindi-2000",
    "mr": "PharynxAI/IndicVoices-marathi-2000",
    "ta": "PharynxAI/IndicVoices-tamil-2000",
    "kn": "PharynxAI/IndicVoices-kannada-2000",
    # extras for later scaling
    "te": "PharynxAI/IndicVoices-telugu-2000",     # if available
    "bn": "PharynxAI/IndicVoices-bengali-2000",
    "gu": "PharynxAI/IndicVoices-gujarati-2000",
}

# Full AI4Bharat repo language codes (used in splits/config)
FULL_REPO = "ai4bharat/indicvoices_r"
FULL_LANG_NAMES = {
    "hi": "Hindi",    "mr": "Marathi",  "ta": "Tamil",    "kn": "Kannada",
    "te": "Telugu",   "bn": "Bengali",  "gu": "Gujarati", "ml": "Malayalam",
    "or": "Odia",     "pa": "Punjabi",  "as": "Assamese", "ur": "Urdu",
    "ks": "Kashmiri", "sa": "Sanskrit", "ne": "Nepali",
}

LANG_FAMILY = {
    "hi": "indo_aryan", "mr": "indo_aryan", "bn": "indo_aryan",
    "gu": "indo_aryan", "pa": "indo_aryan", "or": "indo_aryan",
    "ta": "dravidian",  "kn": "dravidian",  "te": "dravidian", "ml": "dravidian",
}

MIN_DURATION_S  = 1.0    # discard utterances shorter than this
MAX_DURATION_S  = 15.0   # discard very long utterances (VRAM)
VAL_SPEAKERS    = 50     # held-out speakers per language (200 total for 4 langs)
MIN_VAL_UTTS    = 3      # val speakers must have at least this many utterances

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_deps():
    """Install missing packages silently."""
    pkgs = []
    try: import datasets
    except ImportError: pkgs.append("datasets")
    try: import soundfile
    except ImportError: pkgs.append("soundfile")
    try: import librosa
    except ImportError: pkgs.append("librosa")
    if pkgs:
        import subprocess
        print(f"Installing: {pkgs}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def get_duration(audio_array, sampling_rate) -> float:
    return len(audio_array) / sampling_rate


def save_audio(arr, sr: int, path: Path):
    import soundfile as sf
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.array(arr, dtype="float32"), sr)


def write_manifest(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utt_id", "speaker_id", "language", "family",
                  "audio_path", "duration_s", "text", "emotion"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Audio decoding helper — bypasses torchcodec/FFmpeg entirely
# ---------------------------------------------------------------------------

def decode_audio_sample(sample: dict) -> tuple:
    """
    Extract (numpy_array, sample_rate) from a HuggingFace dataset sample,
    bypassing torchcodec/FFmpeg on Windows.

    HF datasets store audio in one of three forms depending on version:
      A) Already decoded: sample["audio"] = {"array": np.ndarray, "sampling_rate": int}
      B) Raw bytes:       sample["audio"] = {"bytes": bytes, "path": str}
      C) Path only:       sample["audio"] = {"path": str}

    We try A first (fast path), then fall back to soundfile for B/C.
    """
    import io
    import soundfile as sf
    import numpy as np

    audio = sample.get("audio") or sample.get("Audio")
    if audio is None:
        return None, None

    # Fast path — already decoded array
    if isinstance(audio.get("array"), np.ndarray):
        return audio["array"], audio["sampling_rate"]

    # Raw bytes path — decode with soundfile (no FFmpeg needed)
    raw_bytes = audio.get("bytes")
    if raw_bytes is not None:
        try:
            arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
            return arr.mean(axis=1), sr   # mono
        except Exception as e:
            print(f"    [warn] soundfile decode failed: {e}")
            return None, None

    # File path fallback
    path = audio.get("path")
    if path and os.path.exists(path):
        try:
            arr, sr = sf.read(path, dtype="float32", always_2d=True)
            return arr.mean(axis=1), sr
        except Exception as e:
            print(f"    [warn] soundfile read failed for {path}: {e}")
            return None, None

    return None, None


# ---------------------------------------------------------------------------
# PharynxAI downloader
# ---------------------------------------------------------------------------

def download_pharynx(lang: str, out_dir: Path, dry_run: bool = False) -> list[dict]:
    """
    Stream PharynxAI/IndicVoices-{lang}-2000 from HuggingFace,
    decode audio with soundfile (no FFmpeg/torchcodec required),
    save as .wav files, return manifest rows.
    """
    from datasets import load_dataset

    repo = PHARYNX_REPOS.get(lang)
    if repo is None:
        print(f"  ⚠  No PharynxAI repo known for lang={lang}, skipping.")
        return []

    print(f"\n[{lang}] Loading {repo}...")
    if dry_run:
        print(f"  DRY RUN — would download {repo}")
        return []

    try:
        # trust_remote_code removed in datasets >= 3.x — don't pass it
        ds = load_dataset(repo, split="train")
    except Exception as e:
        print(f"  ❌ Could not load {repo}: {e}")
        return []

    # Disable HF audio decoding so we get raw bytes, not torchcodec-decoded arrays.
    # This avoids the FFmpeg/torchcodec DLL requirement on Windows entirely.
    try:
        ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    except Exception:
        pass  # older datasets version — will attempt decode anyway, may still work

    print(f"  Dataset size: {len(ds)} utterances")

    audio_dir = out_dir / "audio" / lang
    rows = []
    skipped = 0

    for i, sample in enumerate(ds):
        # PharynxAI schema (actual):
        #   audio_filepath : dict  {"path": "xxx.flac", "array": np.ndarray, "sampling_rate": int}
        #   samples        : int   (sample count — not the waveform)
        #   duration       : float
        #   speaker_id     : str
        #   lang           : str
        #   text / verbatim / normalized : str
        audio_info = sample.get("audio_filepath")
        if audio_info is None or not isinstance(audio_info, dict):
            skipped += 1
            continue

        import numpy as np
        arr = audio_info.get("array")
        sr  = audio_info.get("sampling_rate", 16000)

        if arr is None:
            skipped += 1
            continue

        arr = np.asarray(arr, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=0)          # to mono

        dur = float(sample.get("duration") or (len(arr) / sr))

        if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
            skipped += 1
            continue

        spk_id = str(sample.get("speaker_id") or f"spk_{i:05d}")
        text   = (sample.get("normalized") or sample.get("verbatim")
                  or sample.get("text") or "")
        utt_id = f"{lang}_{i:06d}"

        audio_path = audio_dir / f"{utt_id}.wav"
        save_audio(arr, sr, audio_path)

        rows.append({
            "utt_id":     utt_id,
            "speaker_id": spk_id,          # raw ID — no lang prefix, enables cross-lang pooling
            "language":   lang,
            "family":     LANG_FAMILY.get(lang, "unknown"),
            "audio_path": str(audio_path),
            "duration_s": round(dur, 3),
            "text":       text,
            "emotion":    sample.get("emotion", "neutral"),
        })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(ds)} | saved {len(rows)} | skipped {skipped}")

    print(f"  Done: {len(rows)} kept, {skipped} skipped")
    return rows


# ---------------------------------------------------------------------------
# Full AI4Bharat downloader (gated)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full AI4Bharat downloader (gated) — with crash recovery
# ---------------------------------------------------------------------------

def _ckpt_path(out_dir: Path, lang: str) -> Path:
    """Path to the per-language download checkpoint (JSON)."""
    return out_dir / "manifests" / f".{lang}_download_ckpt.json"


def _load_ckpt(ckpt_path: Path) -> dict:
    """Load checkpoint: {rows: [...], next_stream_idx: int, skipped: int}"""
    import json
    if ckpt_path.exists():
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  ♻  Resuming from checkpoint: {len(data['rows'])} rows already saved "
                  f"(stream position ~{data['next_stream_idx']})")
            return data
        except Exception as e:
            print(f"  ⚠  Checkpoint corrupt ({e}), starting fresh.")
    return {"rows": [], "next_stream_idx": 0, "skipped": 0}


def _save_ckpt(ckpt_path: Path, rows: list, next_idx: int, skipped: int):
    """Atomically save checkpoint."""
    import json
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "next_stream_idx": next_idx, "skipped": skipped},
                  f, ensure_ascii=False)
    os.replace(tmp, ckpt_path)


def download_full(lang: str, out_dir: Path, hf_token: str,
                  dry_run: bool = False,
                  max_utts: int = None) -> list[dict]:
    """
    Stream ai4bharat/indicvoices_r for a single language.

    Crash recovery
    --------------
    A checkpoint is saved every CKPT_EVERY utterances processed.
    If interrupted, re-running the same command resumes from the checkpoint —
    no audio files are re-downloaded, streaming just skips ahead.

    Manifests are written after each language completes (not at the very end)
    so a crash on language 3 doesn't lose languages 1 and 2.
    """
    import json
    from datasets import load_dataset

    CKPT_EVERY = 500   # save checkpoint every N processed utterances

    # Set token as env var so HF hub uses it for all requests (suppresses warnings)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    lang_name = FULL_LANG_NAMES.get(lang, lang)
    ckpt_file = _ckpt_path(out_dir, lang)

    print(f"\n[{lang}] Streaming {FULL_REPO} config={lang_name}...")

    if dry_run:
        print(f"  DRY RUN — would stream {FULL_REPO}/{lang_name}")
        return []

    # ── Load checkpoint (resume if exists) ───────────────────────────────────
    ckpt      = _load_ckpt(ckpt_file)
    rows      = ckpt["rows"]
    skip_n    = ckpt["next_stream_idx"]   # number of stream items already processed
    skipped   = ckpt["skipped"]
    already   = len(rows)

    if max_utts and already >= max_utts:
        print(f"  Already have {already} rows (max_utts={max_utts}). Skipping download.")
        return rows

    # ── Open stream ──────────────────────────────────────────────────────────
    try:
        ds = load_dataset(
            FULL_REPO,
            name=lang_name,
            split="train",
            token=hf_token,
            streaming=True,
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"  ❌ Could not load {FULL_REPO}/{lang_name}: {e}")
        print(f"     Ensure your HF token has access at:")
        print(f"     https://huggingface.co/datasets/ai4bharat/indicvoices_r")
        return rows  # return whatever we have from checkpoint

    import numpy as np

    audio_dir = out_dir / "audio" / lang
    audio_dir.mkdir(parents=True, exist_ok=True)

    schema_key   = None
    schema_style = None
    i_global     = 0   # position in the stream (including skipped)

    print(f"  Fast-forwarding stream by {skip_n} items..." if skip_n > 0 else "  Starting fresh.")

    for sample in ds:
        # ── Fast-forward past already-processed items ─────────────────────────
        if i_global < skip_n:
            i_global += 1
            if i_global % 5000 == 0:
                print(f"  Fast-forwarding... {i_global}/{skip_n}")
            continue

        if max_utts and len(rows) >= max_utts:
            break

        # ── Schema detection (once) ───────────────────────────────────────────
        if schema_key is None:
            for candidate in ("audio", "audio_filepath", "Audio"):
                val = sample.get(candidate)
                if val is not None:
                    schema_key = candidate
                    if isinstance(val, dict):
                        schema_style = "hf_audio_dict" if "array" in val else \
                                       "bytes" if "bytes" in val else "hf_audio_dict"
                    else:
                        schema_style = "unknown"
                    break
            if schema_key is None:
                schema_key   = list(sample.keys())[0]
                schema_style = "unknown"
            print(f"  Schema detected: key='{schema_key}' style='{schema_style}'")

        # ── Audio extraction ──────────────────────────────────────────────────
        arr, sr = None, 16000
        raw = sample.get(schema_key)

        if schema_style == "hf_audio_dict" and isinstance(raw, dict):
            arr_val = raw.get("array")
            sr      = raw.get("sampling_rate", 16000)
            if arr_val is not None:
                arr = np.asarray(arr_val, dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)

        elif schema_style == "bytes" and isinstance(raw, dict):
            raw_bytes = raw.get("bytes")
            if raw_bytes:
                try:
                    import io, soundfile as sf
                    arr_nd, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
                    arr = arr_nd.mean(axis=0)
                except Exception:
                    pass

        else:
            path_val = raw.get("path", "") if isinstance(raw, dict) else str(raw or "")
            if path_val and os.path.exists(path_val):
                try:
                    import soundfile as sf
                    arr_nd, sr = sf.read(path_val, dtype="float32", always_2d=True)
                    arr = arr_nd.mean(axis=0)
                except Exception:
                    pass

        i_global += 1

        if arr is None or len(arr) == 0:
            skipped += 1
            continue

        dur = len(arr) / sr
        if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
            skipped += 1
            continue

        if sr != 16000:
            try:
                import resampy
                arr = resampy.resample(arr, sr, 16000)
            except ImportError:
                pass
            sr = 16000

        spk_id = str(sample.get("speaker_id") or sample.get("Speaker_ID") or f"spk_{i_global:06d}")
        text   = (sample.get("normalized") or sample.get("verbatim") or
                  sample.get("text") or sample.get("transcription") or "")
        utt_id = f"{lang}_{i_global:06d}"

        audio_path = audio_dir / f"{utt_id}.wav"
        save_audio(arr, sr, audio_path)

        rows.append({
            "utt_id":     utt_id,
            "speaker_id": f"{lang}_{spk_id}",
            "language":   lang,
            "family":     LANG_FAMILY.get(lang, "unknown"),
            "audio_path": str(audio_path),
            "duration_s": round(dur, 3),
            "text":       text,
            "emotion":    sample.get("emotion", "neutral"),
        })

        # ── Checkpoint every CKPT_EVERY rows saved ────────────────────────────
        if len(rows) % CKPT_EVERY == 0:
            _save_ckpt(ckpt_file, rows, i_global, skipped)
            print(f"  [{lang}] {i_global} streamed | {len(rows)} kept | {skipped} skipped  ✓ ckpt")

    # ── Final checkpoint + per-language manifest ──────────────────────────────
    _save_ckpt(ckpt_file, rows, i_global, skipped)

    # Write per-language manifests immediately (don't wait for all langs to finish)
    tr_rows, va_rows = split_manifest(rows, lang, out_dir)

    newly = len(rows) - already
    print(f"  [{lang}] Done: {len(rows)} total ({newly} new) | {skipped} skipped")
    return rows


# ---------------------------------------------------------------------------
# Manifest split
# ---------------------------------------------------------------------------

def split_manifest(rows: list[dict], lang: str, out_dir: Path):
    """
    Split into train/val by speaker ID.
    Val speakers are selected from those with >= MIN_VAL_UTTS utterances
    to ensure meaningful EER evaluation.
    """
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)

    # Separate speakers by utterance count
    rich_speakers  = sorted(s for s, v in by_speaker.items() if len(v) >= MIN_VAL_UTTS)
    poor_speakers  = sorted(s for s, v in by_speaker.items() if len(v) < MIN_VAL_UTTS)

    n_val = min(VAL_SPEAKERS, len(rich_speakers) // 5)  # don't starve train set
    val_spks   = set(rich_speakers[-n_val:])
    train_spks = set(rich_speakers[:-n_val]) | set(poor_speakers)

    train_rows = [r for s in train_spks for r in by_speaker[s]]
    val_rows   = [r for s in val_spks   for r in by_speaker[s]]

    total_dur_train = sum(r["duration_s"] for r in train_rows) / 3600
    total_dur_val   = sum(r["duration_s"] for r in val_rows)   / 3600

    print(f"  Total speakers  : {len(by_speaker)}")
    print(f"  Rich (>={MIN_VAL_UTTS} utts): {len(rich_speakers)} speakers")
    print(f"  Train : {len(train_rows):5d} utts, {len(train_spks):4d} speakers, {total_dur_train:.1f}h")
    print(f"  Val   : {len(val_rows):5d} utts, {len(val_spks):4d} speakers, {total_dur_val:.1f}h "
          f"(avg {len(val_rows)/max(len(val_spks),1):.1f} utts/spk)")

    write_manifest(train_rows, out_dir / "manifests" / f"{lang}_train.csv")
    write_manifest(val_rows,   out_dir / "manifests" / f"{lang}_val.csv")

    return train_rows, val_rows


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def print_stats(all_train: list[dict], all_val: list[dict]):
    print("\n" + "="*60)
    print("  Download Summary")
    print("="*60)

    by_lang_train = defaultdict(list)
    by_lang_val   = defaultdict(list)
    for r in all_train: by_lang_train[r["language"]].append(r)
    for r in all_val:   by_lang_val[r["language"]].append(r)

    total_h = 0
    for lang in sorted(set(r["language"] for r in all_train + all_val)):
        tr = by_lang_train[lang]
        va = by_lang_val[lang]
        tr_h = sum(float(r["duration_s"]) for r in tr) / 3600
        va_h = sum(float(r["duration_s"]) for r in va) / 3600
        n_spk_tr = len(set(r["speaker_id"] for r in tr))
        n_spk_va = len(set(r["speaker_id"] for r in va))
        avg_val  = len(va) / max(n_spk_va, 1)
        family = LANG_FAMILY.get(lang, "?")
        total_h += tr_h + va_h
        print(f"  {lang} ({family:12s}): "
              f"train {tr_h:.1f}h/{n_spk_tr}spk | "
              f"val {va_h:.1f}h/{n_spk_va}spk ({avg_val:.1f} utts/spk)")

    print(f"\n  Total data : {total_h:.1f}h across {len(by_lang_train)} languages")
    print(f"  Train rows : {len(all_train)}")
    print(f"  Val rows   : {len(all_val)}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pharynx", "full"], default="pharynx",
                        help="pharynx=fast PharynxAI subsets | full=gated AI4Bharat")
    parser.add_argument("--langs", nargs="+", default=["hi", "mr", "ta", "kn"],
                        help="Language codes to download")
    parser.add_argument("--out_dir", default="data",
                        help="Root output directory")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (required for --mode full)")
    parser.add_argument("--max_utts_per_lang", type=int, default=None,
                        help="Cap utterances per language (useful for testing or balancing). "
                             "Default: download everything available.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without downloading")
    args = parser.parse_args()

    if args.mode == "full" and not args.hf_token:
        print("ERROR: --mode full requires --hf_token")
        print("Get token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print("IndicVC — Data Downloader")
    print(f"Mode     : {args.mode}")
    print(f"Languages: {args.langs}")
    print(f"Output   : {args.out_dir}")

    ensure_deps()

    out_dir = Path(args.out_dir)
    completed_langs = []

    for lang in args.langs:
        if lang not in PHARYNX_REPOS and lang not in FULL_LANG_NAMES:
            print(f"  ⚠  Unknown language code: {lang}")
            continue

        if args.mode == "pharynx":
            rows = download_pharynx(lang, out_dir, args.dry_run)
            if rows and not args.dry_run:
                split_manifest(rows, lang, out_dir)
                completed_langs.append(lang)
        else:
            rows = download_full(lang, out_dir, args.hf_token, args.dry_run,
                                 max_utts=args.max_utts_per_lang)
            # download_full calls split_manifest internally after each language
            if rows and not args.dry_run:
                completed_langs.append(lang)

    if not args.dry_run and completed_langs:
        # Rebuild combined manifests from per-language files
        # This works even if only some languages completed
        all_train, all_val = [], []
        for lang in completed_langs:
            tr_path = out_dir / "manifests" / f"{lang}_train.csv"
            va_path = out_dir / "manifests" / f"{lang}_val.csv"
            if tr_path.exists():
                with open(tr_path, encoding="utf-8") as f:
                    all_train.extend(list(csv.DictReader(f)))
            if va_path.exists():
                with open(va_path, encoding="utf-8") as f:
                    all_val.extend(list(csv.DictReader(f)))

        if all_train:
            write_manifest(all_train, out_dir / "manifests" / "all_train.csv")
            write_manifest(all_val,   out_dir / "manifests" / "all_val.csv")
            print_stats(all_train, all_val)
            print(f"\n  ✅ Completed languages: {completed_langs}")
            missing = [l for l in args.langs if l not in completed_langs]
            if missing:
                print(f"  ⚠  Incomplete languages: {missing}")
                print(f"     Re-run with --langs {' '.join(missing)} to resume them.")


if __name__ == "__main__":
    main()