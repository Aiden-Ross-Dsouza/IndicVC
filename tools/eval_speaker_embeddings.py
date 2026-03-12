"""
eval_speaker_embeddings.py
==========================
Standalone evaluation of a trained speaker encoder checkpoint.
Answers: "Has the model learned meaningful speaker embeddings?"

Produces:
  1. Same vs different speaker cosine similarity distributions + stats
  2. EER (Equal Error Rate) overall and per language
  3. Similarity histogram plot saved to disk
  4. UMAP scatter plots (by language and by speaker) saved to disk

Interpretation guide is printed inline — no need to interpret numbers yourself.

Usage
-----
    # Evaluate best.pt on val set
    python tools/eval_speaker_embeddings.py \
        --checkpoint checkpoints/speaker/best.pt \
        --manifest   data/manifests/all_val.csv \
        --device cuda

    # Evaluate last.pt (current training state) on train set for more speakers
    python tools/eval_speaker_embeddings.py \
        --checkpoint checkpoints/speaker/last.pt \
        --manifest   data/manifests/all_train.csv \
        --max_utts 500 --device cuda

    # Compare two checkpoints
    python tools/eval_speaker_embeddings.py \
        --checkpoint  checkpoints/speaker/best.pt \
        --checkpoint2 checkpoints/speaker/last_safe_backup.pt \
        --manifest data/manifests/all_val.csv --device cuda
"""

import argparse, csv, os, random, sys
from collections import defaultdict
from pathlib import Path

try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modules.speaker_encoder import SpeakerEncoderConfig, SpeakerEncoder

SR = 16_000


def load_audio(path):
    import soundfile as sf
    arr, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(arr.T).mean(0)
    if sr != SR:
        try:
            import torchaudio.functional as TAF
            wav = TAF.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
        except Exception:
            pass
    return wav


def load_manifest(path, max_utts=None, langs=None, min_utts_per_speaker=2):
    """Load manifest, filtering to speakers with >= min_utts_per_speaker utterances."""
    by_spk = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if langs and row["language"] not in langs:
                continue
            if os.path.exists(row["audio_path"]):
                by_spk[row["speaker_id"]].append(row)

    # Keep only speakers with enough utterances for same-speaker pairs
    rows = []
    skipped_spk = 0
    for spk, spk_rows in by_spk.items():
        if len(spk_rows) >= min_utts_per_speaker:
            rows.extend(spk_rows)
        else:
            skipped_spk += 1

    if skipped_spk > 0:
        print(f"  (Skipped {skipped_spk} speakers with < {min_utts_per_speaker} utterances)")

    if not rows:
        print(f"  ⚠  No speakers with >= {min_utts_per_speaker} utterances found.")
        print(f"     Use --min_utts_per_speaker 1 to include all, but EER will be meaningless.")
        # Fall back to all rows anyway
        rows = []
        for spk_rows in by_spk.values():
            rows.extend(spk_rows)

    if max_utts and len(rows) > max_utts:
        # Stratified — keep balance, but always include >= min_utts per speaker
        by_spk2 = defaultdict(list)
        for r in rows:
            by_spk2[r["speaker_id"]].append(r)
        sampled = []
        per_spk = max(min_utts_per_speaker, max_utts // len(by_spk2))
        for spk_rows in by_spk2.values():
            sampled.extend(random.sample(spk_rows, min(per_spk, len(spk_rows))))
        rows = sampled[:max_utts]

    return rows


def extract_embeddings(model, rows, device, max_dur_s=6.0, batch_size=32):
    model.eval()
    all_emb, all_spk, all_lang = [], [], []
    max_samples = int(max_dur_s * SR)
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        wavs = []
        for r in batch:
            try:
                wav = load_audio(r["audio_path"])
            except Exception:
                wav = torch.zeros(SR * 2)
            if wav.shape[-1] > max_samples:
                s = random.randint(0, wav.shape[-1] - max_samples)
                wav = wav[s:s+max_samples]
            wavs.append(wav)
        max_len = max(w.shape[-1] for w in wavs)
        padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs]).to(device)
        with torch.no_grad():
            emb = model(padded)
        all_emb.append(emb.cpu())
        all_spk.extend(r["speaker_id"] for r in batch)
        all_lang.extend(r["language"] for r in batch)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Embedded {min(i+batch_size, len(rows))}/{len(rows)}...")
    return torch.cat(all_emb), all_spk, all_lang


def compute_pairs(emb, spk_ids, max_pairs=100_000):
    n = len(emb)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    scores  = np.array([float(emb[i] @ emb[j]) for i, j in pairs], dtype=np.float32)
    targets = np.array([1 if spk_ids[i] == spk_ids[j] else 0 for i, j in pairs])
    return scores, targets


def compute_eer(scores, targets):
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(targets, scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        return float((fpr[idx] + fnr[idx]) / 2 * 100)
    except Exception:
        return 99.0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    n_spk = (ckpt.get("config") or {}).get("n_speakers", 3589) or 3589
    cfg = SpeakerEncoderConfig(pretrained_model="", embedding_dim=192, n_speakers=n_spk)
    model = SpeakerEncoder(cfg)
    missing, _ = model.load_state_dict(state, strict=False)
    non_head = [k for k in missing if "aam" not in k.lower()]
    if non_head:
        print(f"  ⚠  Missing keys (non-head): {non_head[:3]}")
    return model.to(device)


def run_evaluation(ckpt_path, rows, device, label="", out_dir=Path("eval_output"), save_plots=True):
    print(f"\n{'='*60}")
    print(f"  {Path(ckpt_path).name}  {label}")
    print(f"  Utterances: {len(rows)}  |  Speakers: {len(set(r['speaker_id'] for r in rows))}")
    print(f"{'='*60}")

    model = load_model(ckpt_path, device)
    emb_t, spk_ids, lang_ids = extract_embeddings(model, rows, device)
    emb = F.normalize(emb_t, dim=-1).numpy()

    scores, targets = compute_pairs(emb, spk_ids)
    same = scores[targets == 1]
    diff = scores[targets == 0]
    gap  = float(same.mean() - diff.mean())
    eer  = compute_eer(scores, targets)

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Cosine similarity statistics                       │
  ├─────────────────────────────────────────────────────┤
  │  Same-speaker :  mean={same.mean():+.4f}   std={same.std():.4f}    │
  │  Diff-speaker :  mean={diff.mean():+.4f}   std={diff.std():.4f}    │
  │  Separation Δ :  {gap:+.4f}                            │
  │  EER          :  {eer:.2f}%                              │
  └─────────────────────────────────────────────────────┘""")

    if gap > 0.30:   print("  ✅ GOOD — clear speaker separation")
    elif gap > 0.15: print("  🟡 MODERATE — some separation, more training may help")
    elif gap > 0.05: print("  🟠 WEAK — minimal separation")
    else:            print("  ❌ POOR — essentially random embeddings")

    if eer < 5:    print(f"  EER {eer:.1f}% → Excellent")
    elif eer < 10: print(f"  EER {eer:.1f}% → Good")
    elif eer < 20: print(f"  EER {eer:.1f}% → Fair (still learning)")
    elif eer < 35: print(f"  EER {eer:.1f}% → Weak (early training / small data)")
    else:          print(f"  EER {eer:.1f}% → Poor (not learning speaker identity)")

    # Per-language
    print(f"\n  Per-language EER:")
    lang_arr = np.array(lang_ids)
    for lg in sorted(set(lang_ids)):
        mask = lang_arr == lg
        if mask.sum() < 10: continue
        lg_spks = [s for s, l in zip(spk_ids, lang_ids) if l == lg]
        sc, tg = compute_pairs(emb[mask], lg_spks, max_pairs=20_000)
        lg_eer = compute_eer(sc, tg)
        lg_gap = sc[tg==1].mean() - sc[tg==0].mean() if (tg==1).any() else 0
        print(f"    {lg:4s}  EER={lg_eer:5.1f}%  Δsim={lg_gap:+.3f}  "
              f"n={mask.sum()}  spk={len(set(lg_spks))}")

    # Plots
    if save_plots:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = Path(ckpt_path).stem + (f"_{label}" if label else "")

            # Similarity histogram
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(diff, bins=100, alpha=0.55, color="steelblue",
                    label=f"Diff-speaker  μ={diff.mean():.3f}", density=True)
            ax.hist(same, bins=100, alpha=0.55, color="tomato",
                    label=f"Same-speaker  μ={same.mean():.3f}", density=True)
            ax.axvline(diff.mean(), color="steelblue", ls="--", lw=1.5)
            ax.axvline(same.mean(), color="tomato",    ls="--", lw=1.5)
            ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
            ax.set_title(f"Speaker embedding similarity  |  EER={eer:.2f}%  Δ={gap:.3f}")
            ax.legend(fontsize=9); plt.tight_layout()
            p = out_dir / f"{tag}_hist.png"
            fig.savefig(p, dpi=150); plt.close(fig)
            print(f"\n  📊 Saved histogram → {p}")

            # UMAP
            try:
                import umap as _umap
                n_max = min(600, len(emb))
                idx = np.random.choice(len(emb), n_max, replace=False)
                e2d = _umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(emb[idx])
                lang_s = np.array(lang_ids)[idx]
                spk_s  = np.array(spk_ids)[idx]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                for ax, arr, title in [(axes[0], lang_s, "Language"), (axes[1], spk_s, "Speaker")]:
                    uniq = sorted(set(arr))[:20]
                    cmap = plt.cm.get_cmap("tab20", len(uniq))
                    for ci, v in enumerate(uniq):
                        m = arr == v
                        ax.scatter(e2d[m,0], e2d[m,1], color=cmap(ci),
                                   label=v if title=="Language" else "", alpha=0.65, s=12)
                    if title == "Language": ax.legend(fontsize=9, markerscale=2)
                    ax.set_title(f"UMAP — by {title}"); ax.set_xlabel("U1"); ax.set_ylabel("U2")
                plt.suptitle(Path(ckpt_path).name, fontsize=9); plt.tight_layout()
                p2 = out_dir / f"{tag}_umap.png"
                fig.savefig(p2, dpi=150); plt.close(fig)
                print(f"  📊 Saved UMAP     → {p2}")
            except ImportError:
                print("  (UMAP skipped — pip install umap-learn)")
            except Exception as e:
                print(f"  (UMAP failed: {e})")
        except ImportError:
            print("  (Plots skipped — pip install matplotlib)")

    return {"eer": eer, "gap": gap, "same_mean": float(same.mean()), "diff_mean": float(diff.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--checkpoint2", default=None, help="Second checkpoint to compare")
    parser.add_argument("--manifest",    required=True)
    parser.add_argument("--langs",       nargs="+", default=None)
    parser.add_argument("--max_utts",    type=int, default=1000,
                        help="Max total utterances to evaluate (stratified sample)")
    parser.add_argument("--min_utts_per_speaker", type=int, default=2,
                        help="Min utterances per speaker (speakers below this are excluded). "
                             "Must be >=2 for meaningful EER. Use 1 only for embedding visualisation.")
    parser.add_argument("--out_dir",     default="eval_output")
    parser.add_argument("--no_plots",    action="store_true")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)
    rows = load_manifest(args.manifest, args.max_utts, args.langs,
                         min_utts_per_speaker=args.min_utts_per_speaker)
    print(f"Loaded {len(rows)} utterances")

    r1 = run_evaluation(args.checkpoint, rows, args.device,
                        out_dir=Path(args.out_dir), save_plots=not args.no_plots)

    if args.checkpoint2:
        r2 = run_evaluation(args.checkpoint2, rows, args.device, label="cmp2",
                            out_dir=Path(args.out_dir), save_plots=not args.no_plots)
        print(f"\n{'='*60}\n  Comparison\n{'='*60}")
        print(f"  {'Metric':15s}  {'ckpt1':>10}  {'ckpt2':>10}")
        for k in ["eer", "gap", "same_mean", "diff_mean"]:
            print(f"  {k:15s}  {r1[k]:>10.4f}  {r2[k]:>10.4f}")
        print(f"  → Better (lower EER): {'ckpt1' if r1['eer'] <= r2['eer'] else 'ckpt2'}")

if __name__ == "__main__":
    main()