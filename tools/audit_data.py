"""
audit_data.py
=============
Thorough data quality audit before training.
Run this before every training run.

Usage
-----
    python tools/audit_data.py \
        --train data_full/manifests/all_train.csv \
        --val   data_full/manifests/all_val.csv

    # Also check that audio files actually exist on disk:
    python tools/audit_data.py \
        --train data_full/manifests/all_train.csv \
        --val   data_full/manifests/all_val.csv \
        --check_audio
"""

import argparse, csv, os, sys
from collections import defaultdict, Counter
from pathlib import Path


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["duration_s"] = float(r["duration_s"])
            rows.append(r)
    return rows


def audit(rows, label, check_audio=False):
    by_spk  = defaultdict(list)
    by_lang = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
        by_lang[r["language"]].append(r)

    total_h  = sum(r["duration_s"] for r in rows) / 3600
    durs     = [r["duration_s"] for r in rows]
    all_upc  = sorted(len(v) for v in by_spk.values())

    print(f"\n{'='*62}")
    print(f"  {label}: {len(rows):,} utts | {len(by_spk):,} speakers | {total_h:.1f}h")
    print(f"{'='*62}")

    # ── Per-language ─────────────────────────────────────────────
    print(f"\n  Per-language breakdown:")
    print(f"  {'lang':4}  {'utts':>7}  {'spk':>5}  {'hours':>6}  "
          f"{'avg':>5}  {'min':>4}  {'med':>4}  {'max':>5}  utts/spk")
    print(f"  {'-'*58}")
    for lg in sorted(by_lang):
        lg_by_spk = defaultdict(list)
        for r in by_lang[lg]:
            lg_by_spk[r["speaker_id"]].append(r)
        upc = sorted(len(v) for v in lg_by_spk.values())
        h   = sum(r["duration_s"] for r in by_lang[lg]) / 3600
        med = upc[len(upc)//2]
        print(f"  {lg:4}  {len(by_lang[lg]):>7,}  {len(lg_by_spk):>5}  {h:>6.1f}"
              f"  {sum(upc)/len(upc):>5.1f}  {upc[0]:>4}  {med:>4}  {upc[-1]:>5}")

    # ── Utts/speaker distribution ────────────────────────────────
    print(f"\n  Utterances-per-speaker distribution:")
    buckets = [(1,1),(2,2),(3,4),(5,9),(10,19),(20,49),(50,99),(100,499),(500,9999)]
    for lo, hi in buckets:
        n   = sum(1 for c in all_upc if lo <= c <= hi)
        pct = 100 * n / len(all_upc)
        bar = "█" * min(int(pct / 2), 35)
        print(f"    {lo:4}-{hi:<4}: {n:5} speakers ({pct:5.1f}%)  {bar}")
    print(f"\n  ≥ 2 utts/spk : {sum(1 for c in all_upc if c>=2):,}  "
          f"({100*sum(1 for c in all_upc if c>=2)/len(all_upc):.0f}%)")
    print(f"  ≥ 3 utts/spk : {sum(1 for c in all_upc if c>=3):,}  "
          f"({100*sum(1 for c in all_upc if c>=3)/len(all_upc):.0f}%)")
    print(f"  ≥ 5 utts/spk : {sum(1 for c in all_upc if c>=5):,}  "
          f"({100*sum(1 for c in all_upc if c>=5)/len(all_upc):.0f}%)")
    print(f"  ≥10 utts/spk : {sum(1 for c in all_upc if c>=10):,}  "
          f"({100*sum(1 for c in all_upc if c>=10)/len(all_upc):.0f}%)")

    # ── Duration distribution ────────────────────────────────────
    print(f"\n  Duration distribution:")
    print(f"    min={min(durs):.2f}s  max={max(durs):.2f}s  "
          f"mean={sum(durs)/len(durs):.2f}s  "
          f"median={sorted(durs)[len(durs)//2]:.2f}s")
    for lo, hi, tag in [
        (0,   1,   "< 1s  ⚠ very short"),
        (1,   2,   "1–2s  marginal   "),
        (2,   5,   "2–5s  ✓ ideal    "),
        (5,  10,   "5–10s ✓ good     "),
        (10, 15,   "10–15s ok        "),
        (15, 9999, ">15s  ⚠ very long"),
    ]:
        n = sum(1 for d in durs if lo <= d < hi)
        print(f"    {tag}: {n:6,}  ({100*n/len(durs):.1f}%)")

    # ── Duplicates ───────────────────────────────────────────────
    print(f"\n  Duplicate checks:")
    utt_ids   = [r["utt_id"]     for r in rows]
    aud_paths = [r["audio_path"] for r in rows]
    texts     = [r["text"].strip() for r in rows if r["text"].strip()]

    dup_ids   = len(utt_ids)   - len(set(utt_ids))
    dup_paths = len(aud_paths) - len(set(aud_paths))
    text_cnt  = Counter(texts)
    dup_texts = sum(v-1 for v in text_cnt.values() if v > 1)
    top_dups  = text_cnt.most_common(5)

    status = lambda n: "✅" if n == 0 else "⚠ "
    print(f"    {status(dup_ids)}  Duplicate utt_ids    : {dup_ids}")
    print(f"    {status(dup_paths)}  Duplicate audio_paths: {dup_paths}")
    print(f"    {'✅' if dup_texts < len(rows)*0.01 else '⚠ '}  "
          f"Exact-duplicate texts: {dup_texts}  "
          f"({100*dup_texts/max(len(texts),1):.1f}% of non-empty)")
    if dup_texts > 0:
        print(f"      Top repeated texts: {[(t[:40], c) for t, c in top_dups if c > 1]}")

    # ── Train/val speaker leak ───────────────────────────────────
    # (only printed when auditing both together — skipped here)

    # ── Speaker-language consistency ────────────────────────────
    spk_langs = defaultdict(set)
    for r in rows:
        spk_langs[r["speaker_id"]].add(r["language"])
    cross = {s: ls for s, ls in spk_langs.items() if len(ls) > 1}
    print(f"\n  Speaker-language consistency:")
    if cross:
        print(f"    ⚠  {len(cross)} speakers appear in multiple languages "
              f"(expected 0 — may indicate speaker_id collision):")
        for s, ls in list(cross.items())[:5]:
            print(f"      {s}: {sorted(ls)}")
    else:
        print(f"    ✅ All speakers appear in exactly one language")

    # ── Family column ────────────────────────────────────────────
    families  = Counter(r["family"] for r in rows)
    unknown   = families.get("unknown", 0)
    print(f"\n  Language families: {dict(families)}")
    if unknown:
        print(f"    ⚠  {unknown} rows with family='unknown'")
    else:
        print(f"    ✅ No unknown families")

    # ── Emotion column ───────────────────────────────────────────
    emotions = Counter(r["emotion"] for r in rows)
    print(f"\n  Emotion distribution: {dict(emotions)}")
    if len(emotions) == 1 and "neutral" in emotions:
        print(f"    ⚠  All rows have emotion='neutral' — emotion encoder will need "
              f"AI4Bharat Rasa dataset separately (this is expected)")

    # ── Missing audio files ──────────────────────────────────────
    if check_audio:
        print(f"\n  Audio file existence check (may take a minute)...")
        missing = [r["audio_path"] for r in rows if not os.path.exists(r["audio_path"])]
        if missing:
            print(f"  ⚠  {len(missing)} missing audio files!")
            for p in missing[:10]:
                print(f"      {p}")
            if len(missing) > 10:
                print(f"      ... and {len(missing)-10} more")
        else:
            print(f"  ✅ All {len(rows):,} audio files exist on disk")
    else:
        print(f"\n  (Run with --check_audio to verify all WAV files exist on disk)")

    return {
        "n_utts": len(rows), "n_speakers": len(by_spk),
        "hours": total_h, "dup_ids": dup_ids, "dup_paths": dup_paths,
        "pct_ge3": 100*sum(1 for c in all_upc if c>=3)/len(all_upc),
        "pct_ge5": 100*sum(1 for c in all_upc if c>=5)/len(all_upc),
    }


def check_train_val_leak(train_rows, val_rows):
    train_spks = set(r["speaker_id"] for r in train_rows)
    val_spks   = set(r["speaker_id"] for r in val_rows)
    leak = train_spks & val_spks
    print(f"\n  Train/Val speaker leak check:")
    if leak:
        print(f"  ⚠  {len(leak)} speakers appear in BOTH train and val!")
        print(f"     This would inflate val metrics. Examples: {list(leak)[:5]}")
    else:
        print(f"  ✅ Zero speaker overlap between train and val")


def training_readiness(tr_stats, va_stats):
    print(f"\n{'='*62}")
    print(f"  Training Readiness Assessment")
    print(f"{'='*62}")

    checks = [
        (tr_stats["n_speakers"] >= 500,
         f"Train speakers {tr_stats['n_speakers']} >= 500"),
        (tr_stats["pct_ge3"] >= 50,
         f"Train speakers w/ >=3 utts: {tr_stats['pct_ge3']:.0f}% (need >=50%)"),
        (tr_stats["pct_ge5"] >= 30,
         f"Train speakers w/ >=5 utts: {tr_stats['pct_ge5']:.0f}% (need >=30%)"),
        (va_stats["n_speakers"] >= 100,
         f"Val speakers {va_stats['n_speakers']} >= 100"),
        (va_stats["n_utts"] / max(va_stats["n_speakers"],1) >= 5,
         f"Val avg utts/spk {va_stats['n_utts']/max(va_stats['n_speakers'],1):.1f} >= 5"),
        (tr_stats["dup_ids"] == 0,   f"No duplicate utt_ids in train"),
        (va_stats["dup_ids"] == 0,   f"No duplicate utt_ids in val"),
        (tr_stats["hours"] >= 20,    f"Train data {tr_stats['hours']:.1f}h >= 20h"),
    ]

    all_pass = True
    for passed, msg in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {msg}")
        if not passed:
            all_pass = False

    print(f"\n  {'🟢 READY TO TRAIN' if all_pass else '🔴 FIX ISSUES ABOVE BEFORE TRAINING'}")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       required=True)
    parser.add_argument("--val",         required=True)
    parser.add_argument("--check_audio", action="store_true",
                        help="Verify every audio file exists on disk (slow)")
    args = parser.parse_args()

    print("IndicVC — Data Audit")
    print(f"Train: {args.train}")
    print(f"Val  : {args.val}")

    train_rows = load(args.train)
    val_rows   = load(args.val)

    tr_stats = audit(train_rows, "TRAIN", args.check_audio)
    va_stats = audit(val_rows,   "VAL",   args.check_audio)
    check_train_val_leak(train_rows, val_rows)
    training_readiness(tr_stats, va_stats)


if __name__ == "__main__":
    main()