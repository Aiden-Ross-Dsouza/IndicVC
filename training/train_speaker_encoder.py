"""
train_speaker_encoder.py
========================
Fine-tunes the IndicVC ECAPA-TDNN speaker encoder on IndicVoices-R.

Crash recovery
--------------
A `last.pt` checkpoint is written atomically after EVERY epoch (temp file
then os.replace). If the process is killed mid-epoch, the last completed
epoch is safe. On the next run, pass --auto_resume to pick up from last.pt
automatically, or --resume PATH for a specific checkpoint.

SIGINT / SIGTERM (Ctrl-C or cluster pre-emption) is caught: the current
epoch finishes, last.pt is saved, then the process exits cleanly.

W&B logging
-----------
Pass --wandb to enable. Logs every step (loss, lr, grad_norm) and every
epoch (train/val loss, acc, EER, similarity histograms, embedding scatter).
Run ID is saved to last.pt so resuming continues the same W&B run.

Training phases
---------------
Phase A (epochs 1 .. phase_b_epoch-1):
  First 2 TDNN layers frozen. LR = lr. Linear warmup over first 1000 steps.
  Rationale: VoxCeleb pretrained low-level filters already good — let upper
  layers adapt to Indic phoneme distribution first.

Phase B (epochs phase_b_epoch .. end):
  All layers unfrozen. LR = lr * 0.5, cosine decay to 1e-6.

Usage
-----
    # First run
    python training/train_speaker_encoder.py \
        --train_manifest data/manifests/all_train.csv \
        --val_manifest   data/manifests/all_val.csv \
        --pretrained --device cuda --epochs 30 --wandb

    # Auto-resume after crash (picks up last.pt in ckpt_dir)
    python training/train_speaker_encoder.py \
        --train_manifest data/manifests/all_train.csv \
        --val_manifest   data/manifests/all_val.csv \
        --auto_resume --device cuda --wandb

    # Resume from specific checkpoint
    python training/train_speaker_encoder.py \
        --resume checkpoints/speaker/speaker_encoder_epoch010.pt \
        --train_manifest data/manifests/all_train.csv \
        --val_manifest   data/manifests/all_val.csv
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

# ── torchaudio compat patch ──────────────────────────────────────────────────
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

sys.path.insert(0, ".")
from modules.speaker_encoder import SpeakerEncoderConfig, SpeakerEncoder
from training.speaker_dataset import build_speaker_loaders


# ============================================================================
# Global flag — set by signal handler so the epoch loop can exit gracefully
# ============================================================================
_STOP_REQUESTED = False

def _handle_signal(signum, frame):
    global _STOP_REQUESTED
    print(f"\n[Signal {signum}] Stop requested — finishing current epoch then saving.")
    _STOP_REQUESTED = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ============================================================================
# W&B wrapper — all wandb calls go through this so the rest of the code
# never needs to check `if wandb_enabled` inline.
# ============================================================================
class WandbLogger:
    def __init__(self, enabled: bool, project: str, run_name: str,
                 config: dict, run_id: str = None):
        self.enabled = enabled
        self.run = None
        self.run_id = run_id
        if not enabled:
            return
        try:
            import wandb
            import os as _os
            self.wandb = wandb
            init_kwargs = dict(
                project=project,
                name=run_name,
                config=config,
            )
            if run_id:
                # Set env var for reliability on Windows + thread start method
                _os.environ["WANDB_RUN_ID"] = run_id
                init_kwargs["id"]     = run_id
                init_kwargs["resume"] = "must"   # fail loudly if run not found
            else:
                init_kwargs["resume"] = "allow"
            self.run = wandb.init(**init_kwargs)
            self.run_id = self.run.id
            print(f"  W&B run: {self.run.url}")
        except ImportError:
            print("  ⚠  wandb not installed. Run: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"  ⚠  W&B init failed: {e}. Continuing without W&B.")
            self.enabled = False

    def log(self, metrics: dict, step: int = None):
        if not self.enabled or self.run is None:
            return
        # Always use wandb's internal step counter — passing explicit step
        # causes silent drops when it goes non-monotonic across calls.
        self.wandb.log(metrics)

    def log_histogram(self, name: str, data, step: int = None):
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.log({name: self.wandb.Histogram(data)})
        except Exception:
            pass

    def log_table(self, name: str, columns: list, data: list, step: int = None):
        if not self.enabled or self.run is None:
            return
        try:
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({name: table})
        except Exception:
            pass

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()


# ============================================================================
# EER (Equal Error Rate)
# ============================================================================

def compute_eer(embeddings: torch.Tensor, labels: torch.Tensor,
                max_pairs: int = 50_000) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (eer_pct, same_scores, diff_scores).

    NOTE: EER is unreliable with < 200 utterances / < 50 speakers.
    With small val sets, use val_loss as the primary training signal
    and treat EER as directional only.
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return 99.0, np.array([]), np.array([])

    n = len(embeddings)
    if n < 20:
        return 99.0, np.array([]), np.array([])

    emb = F.normalize(embeddings, dim=-1).cpu().numpy()
    lab = labels.cpu().numpy()

    n_speakers = len(set(lab.tolist()))
    if n_speakers < 10:
        print(f"  [EER] ⚠  Only {n_speakers} val speakers — EER will be noisy. "
              f"Use val_loss as primary metric.")

    # Build balanced pairs: equal number of same/diff speaker pairs
    same_pairs, diff_pairs = [], []
    indices_by_spk = {}
    for idx, l in enumerate(lab):
        indices_by_spk.setdefault(int(l), []).append(idx)

    # Same-speaker pairs
    for spk, idxs in indices_by_spk.items():
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                same_pairs.append((idxs[a], idxs[b]))

    # Different-speaker pairs (sample to balance with same)
    all_spks = list(indices_by_spk.keys())
    n_diff_target = min(max(len(same_pairs) * 10, 1000), max_pairs)
    for _ in range(n_diff_target):
        sa, sb = random.sample(all_spks, 2)
        diff_pairs.append((
            random.choice(indices_by_spk[sa]),
            random.choice(indices_by_spk[sb]),
        ))

    all_pairs  = same_pairs  + diff_pairs
    tgt        = [1] * len(same_pairs) + [0] * len(diff_pairs)
    scores     = np.array([float(emb[i] @ emb[j]) for i, j in all_pairs])
    targets    = np.array(tgt)

    same_scores = scores[targets == 1]
    diff_scores  = scores[targets == 0]

    try:
        fpr, tpr, _ = roc_curve(targets, scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2 * 100)
    except Exception:
        eer = 99.0

    return eer, same_scores, diff_scores


# ============================================================================
# Model utilities
# ============================================================================

def freeze_layers(model: SpeakerEncoder, n_freeze: int):
    """Freeze the first n_freeze bottom layers (input_block + Res2Net blocks)."""
    for p in model.backbone.input_block.parameters():
        p.requires_grad_(False)
    for i, (r, s) in enumerate(zip(
        model.backbone.res2net_blocks, model.backbone.se_blocks
    )):
        if i < n_freeze - 1:
            for p in r.parameters(): p.requires_grad_(False)
            for p in s.parameters(): p.requires_grad_(False)


def unfreeze_all(model: SpeakerEncoder):
    for p in model.parameters():
        p.requires_grad_(True)


def count_trainable(model: SpeakerEncoder) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(model: SpeakerEncoder) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ============================================================================
# Checkpoint helpers — atomic write via temp file + os.replace
# ============================================================================

def save_checkpoint(state: dict, path: str):
    """Atomically save checkpoint. Safe against mid-write crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on POSIX; near-atomic on Windows


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def find_auto_resume(ckpt_dir: Path) -> str | None:
    """Return path to last.pt if it exists, else None."""
    p = ckpt_dir / "last.pt"
    return str(p) if p.exists() else None


def build_checkpoint_state(
    epoch: int,
    model: SpeakerEncoder,
    optimizer,
    warmup_sched,
    cosine_sched,
    global_step: int,
    best_eer: float,
    best_val_loss: float,
    history: list,
    args,
    wandb_run_id: str | None,
    phase: str,
) -> dict:
    return {
        "epoch":          epoch,
        "global_step":    global_step,
        "phase":          phase,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "warmup":         warmup_sched.state_dict() if warmup_sched else None,
        "cosine":         cosine_sched.state_dict() if cosine_sched else None,
        "best_eer":       best_eer,
        "best_val_loss":  best_val_loss,
        "history":        history,
        "config":         vars(args),
        "n_speakers":     model.cfg.n_speakers,   # saved top-level for reliable resume
        "wandb_run_id":   wandb_run_id,
    }


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(
    model: SpeakerEncoder,
    loader,
    device: str,
    logger: WandbLogger,
    global_step: int,
) -> dict:
    model.eval()
    all_emb, all_lab, all_lang = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        wav = batch["waveform"].to(device)
        lab = batch["label"].to(device)
        emb = model(wav)
        loss = model.aam_softmax_loss(emb, lab)
        total_loss += loss.item()
        n          += 1
        all_emb.append(emb.cpu())
        all_lab.append(lab.cpu())
        all_lang.extend(batch["language"])

    all_emb = torch.cat(all_emb, dim=0)
    all_lab = torch.cat(all_lab, dim=0)

    eer, same_scores, diff_scores = compute_eer(all_emb, all_lab)

    # ── W&B: similarity distribution histograms ──────────────────────────────
    if len(same_scores) > 0:
        logger.log_histogram("val/sim_same_speaker", same_scores)
    if len(diff_scores) > 0:
        logger.log_histogram("val/sim_diff_speaker", diff_scores)

    # ── W&B: per-language EER breakdown ──────────────────────────────────────
    langs = sorted(set(all_lang))
    if len(langs) > 1:
        lang_rows = []
        for lg in langs:
            mask = torch.tensor([l == lg for l in all_lang])
            if mask.sum() < 20:
                continue
            lg_emb = all_emb[mask]
            lg_lab = all_lab[mask]
            lg_eer, _, _ = compute_eer(lg_emb, lg_lab, max_pairs=10_000)
            lang_rows.append([lg, round(lg_eer, 2), int(mask.sum())])
            logger.log({f"val/EER_{lg}": lg_eer})
        if lang_rows:
            logger.log_table(
                "val/per_language_EER",
                columns=["language", "EER_%", "n_utts"],
                data=lang_rows,
            )

    # ── W&B: 2-D UMAP scatter of embeddings (every 5 epochs, max 500 pts) ───
    # (logged separately when called from train loop with epoch info)

    model.train()
    return {
        "val_loss":    total_loss / max(n, 1),
        "val_eer":     eer,
        "same_scores": same_scores,
        "diff_scores": diff_scores,
        "embeddings":  all_emb,
        "labels":      all_lab,
        "languages":   all_lang,
    }


def log_umap(embeddings, labels, languages, logger: WandbLogger,
             global_step: int, n_max: int = 500):
    """Log a 2-D UMAP scatter coloured by language to W&B."""
    if not logger.enabled:
        return
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emb = F.normalize(embeddings, dim=-1).numpy()
        lang_arr = np.array(languages)

        # Subsample
        idx = np.random.choice(len(emb), min(n_max, len(emb)), replace=False)
        emb2 = umap.UMAP(n_components=2, random_state=42).fit_transform(emb[idx])
        langs_sub = lang_arr[idx]

        fig, ax = plt.subplots(figsize=(7, 6))
        for lg in sorted(set(langs_sub)):
            mask = langs_sub == lg
            ax.scatter(emb2[mask, 0], emb2[mask, 1], label=lg, alpha=0.6, s=12)
        ax.legend(fontsize=8)
        ax.set_title("Speaker embedding UMAP (coloured by language)")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        plt.tight_layout()

        logger.log({"val/embedding_umap": logger.wandb.Image(fig)})
        plt.close(fig)
    except ImportError:
        pass   # umap-learn or matplotlib not installed — skip silently
    except Exception as e:
        print(f"  [UMAP] skipped: {e}")


# ============================================================================
# Main training loop
# ============================================================================

def train(args):
    global _STOP_REQUESTED

    device   = args.device
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve resume path ──────────────────────────────────────────────────
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        resume_path = find_auto_resume(ckpt_dir)
        if resume_path:
            print(f"  [auto_resume] Found {resume_path}")
        else:
            print(f"  [auto_resume] No last.pt found in {ckpt_dir} — starting fresh.")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[1/5] Building data loaders...")
    train_loader, val_loader, n_speakers = build_speaker_loaders(
        train_manifest     = args.train_manifest,
        val_manifest       = args.val_manifest,
        speakers_per_batch = args.speakers_per_batch,
        utts_per_speaker   = args.utts_per_speaker,
        val_batch_size     = args.val_batch_size,
        num_workers        = args.num_workers,
        noise_dir          = args.noise_dir,
        langs              = args.langs,
        min_dur_s          = args.min_dur_s,
        max_dur_s          = args.max_dur_s,
    )
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  N training speakers : {n_speakers}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[2/5] Building SpeakerEncoder...")
    cfg = SpeakerEncoderConfig(
        pretrained_model = "speechbrain/spkrec-ecapa-voxceleb" if args.pretrained else "",
        embedding_dim    = 192,
        n_speakers       = n_speakers,
        aam_margin       = args.aam_margin,
        aam_scale        = args.aam_scale,
    )
    model = SpeakerEncoder(cfg).to(device)

    # ── Optimiser + schedulers ───────────────────────────────────────────────
    freeze_layers(model, n_freeze=2)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2,
    )
    warmup_steps  = min(1000, len(train_loader) * 2)
    phase_b_steps = max(1, (args.epochs - args.phase_b_epoch) * len(train_loader))
    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_steps)
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=phase_b_steps, eta_min=1e-6)

    # ── State from checkpoint ────────────────────────────────────────────────
    start_epoch  = 1
    best_eer     = 100.0
    best_val_loss = float("inf")
    no_improve_epochs = 0
    history      = []
    global_step  = 0
    wandb_run_id = None
    current_phase = "A"

    if resume_path:
        print(f"\n  Loading checkpoint: {resume_path}")
        ckpt = load_checkpoint(resume_path)

        # ── Model weights ────────────────────────────────────────────────────
        # Load with strict=False so AAM head size mismatch (different n_speakers)
        # doesn't crash — backbone weights transfer, head reinitialises fresh.
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        head_keys   = [k for k in missing + unexpected if "aam" in k.lower()]
        other_keys  = [k for k in missing if "aam" not in k.lower()]
        if head_keys:
            print(f"  ℹ  AAM head keys skipped (n_speakers changed): {len(head_keys)}")
        if other_keys:
            print(f"  ⚠  Other missing keys: {other_keys[:5]}")

        # ── Optimizer state ─────────────────────────────────────────────────
        # CRITICAL: must detect phase and rebuild optimizer BEFORE loading
        # state dict, because Phase B optimizer covers all params while
        # Phase A optimizer only covers unfrozen params.
        # n_speakers saved top-level since v2 — fall back to config for old ckpts
        ckpt_n_spk    = ckpt.get("n_speakers") or (ckpt.get("config") or {}).get("n_speakers", -1)
        ckpt_phase    = ckpt.get("phase", "A")
        ckpt_epoch    = ckpt.get("epoch", 0)
        resume_phase  = "B" if (ckpt_phase == "B" or ckpt_epoch + 1 >= args.phase_b_epoch) else "A"

        if resume_phase == "B":
            # Unfreeze model and rebuild optimizer with ALL params before loading state
            unfreeze_all(model)
            current_phase = "B"
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-2)
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=max(1, (args.epochs - ckpt_epoch) * len(train_loader)),
                eta_min=1e-6,
            )
            print(f"  ℹ  Resuming in Phase B — rebuilt optimizer with all params")

        if ckpt_n_spk == n_speakers:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("warmup"): warmup_sched.load_state_dict(ckpt["warmup"])
                if ckpt.get("cosine") and resume_phase == "B":
                    cosine_sched.load_state_dict(ckpt["cosine"])
                print(f"  ✅ Optimizer state restored (n_speakers={n_speakers})")
            except Exception as e:
                print(f"  ⚠  Optimizer restore failed ({e}) — optimizer starts fresh")
        else:
            print(f"  ℹ  Optimizer state skipped: checkpoint had {ckpt_n_spk} speakers, "
                  f"current has {n_speakers}. Optimizer starts fresh.")

        # ── Training state ───────────────────────────────────────────────────
        if ckpt_n_spk == n_speakers:
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_eer      = ckpt.get("best_eer", 100.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history       = ckpt.get("history", [])
            global_step   = ckpt.get("global_step", 0)
            wandb_run_id  = ckpt.get("wandb_run_id")
            # current_phase already set correctly above during optimizer rebuild
            if history and best_val_loss < float("inf"):
                # Count consecutive epochs from the END of history where
                # EER did not improve — this is what the training loop tracks.
                no_improve_epochs = 0
                for r in reversed(history):
                    if r.get("val_eer", 100.0) <= best_eer + 0.01:
                        break   # found an improvement, stop counting
                    no_improve_epochs += 1
                no_improve_epochs = min(no_improve_epochs, args.patience - 1)  # don't fire immediately on resume
                print(f"  Patience counter restored: {no_improve_epochs}/{args.patience}")
            print(f"  Resumed: epoch={start_epoch-1}, step={global_step}, "
                  f"phase={current_phase}, best_val_loss={best_val_loss:.4f}, "
                  f"best_EER={best_eer:.2f}%")
        else:
            print(f"  Starting from epoch 1 with restored backbone weights.")

    # ── W&B ─────────────────────────────────────────────────────────────────
    run_name = args.wandb_run_name or f"speaker-enc-{time.strftime('%m%d-%H%M')}"
    if wandb_run_id:
        print(f"  W&B resuming run: {wandb_run_id}")
    logger = WandbLogger(
        enabled  = args.wandb,
        project  = args.wandb_project,
        run_name = run_name,
        config   = vars(args),
        run_id   = wandb_run_id if args.wandb else None,
    )
    if args.wandb and logger.run_id:
        wandb_run_id = logger.run_id

    # ── Print plan ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Training plan:")
    print(f"  Phase A (2 layers frozen) : epochs 1–{args.phase_b_epoch-1}, LR={args.lr:.1e}")
    print(f"  Phase B (all unfrozen)    : epochs {args.phase_b_epoch}–{args.epochs}, "
          f"LR={args.lr*0.1:.1e} → cosine")
    print(f"  Checkpoint dir : {ckpt_dir}")
    print(f"  Device         : {device}")
    print(f"  W&B            : {'enabled ('+run_name+')' if args.wandb else 'disabled'}")
    print(f"  Auto-save      : every epoch (last.pt) + every {args.save_every} epochs")
    print()

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        if _STOP_REQUESTED:
            print("  Stop flag set before epoch start — exiting.")
            break

        # Phase transition
        if epoch == args.phase_b_epoch and current_phase == "A":
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch}: *** Phase B — unfreezing all layers ***")
            print(f"{'='*60}")
            unfreeze_all(model)
            current_phase = "B"
            # Rebuild optimiser with Phase B LR for all params
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-2)
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=max(1, (args.epochs - epoch) * len(train_loader)),
                eta_min=1e-6,
            )
            print(f"  Trainable params: {count_trainable(model):,}")

        model.train()
        ep_loss = ep_acc = ep_gnorm = 0.0
        n_batches = 0
        t_epoch   = time.time()

        for batch in train_loader:
            if _STOP_REQUESTED:
                break  # exit inner loop → will save after epoch metrics

            wav = batch["waveform"].to(device)    # (B, samples)
            lab = batch["label"].to(device)        # (B,)

            optimizer.zero_grad()
            emb = model(wav)                       # (B, 192)
            loss = model.aam_softmax_loss(emb, lab)
            loss.backward()

            gnorm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Scheduler stepping
            if global_step < warmup_steps:
                warmup_sched.step()
            elif current_phase == "B":
                cosine_sched.step()

            with torch.no_grad():
                acc = 0.0   # logits not returned by aam_softmax_loss; tracked via EER instead

            ep_loss  += loss.item()
            ep_acc   += acc
            ep_gnorm += gnorm
            n_batches += 1
            global_step += 1

            # Per-step W&B logging
            lr = optimizer.param_groups[0]["lr"]
            logger.log({
                "train/loss":      loss.item(),
                "train/lr":        lr,
                "train/grad_norm": gnorm,
            })

            if global_step % args.log_every == 0:
                print(f"  step {global_step:6d} | loss {loss.item():.4f} "
                      f"| lr {lr:.2e} | gnorm {gnorm:.3f}")

        ep_loss  /= max(n_batches, 1)
        ep_acc   /= max(n_batches, 1)
        ep_gnorm /= max(n_batches, 1)
        ep_time   = time.time() - t_epoch

        # ── Validation ───────────────────────────────────────────────────────
        val = validate(model, val_loader, device, logger, global_step)
        eer = val["val_eer"]

        print(f"\nEpoch {epoch:3d}/{args.epochs} | phase={current_phase} | "
              f"train_loss={ep_loss:.4f} | "
              f"val_loss={val['val_loss']:.4f} | val_EER={eer:.2f}% | "
              f"time={ep_time:.0f}s")

        # Per-epoch W&B logging
        logger.log({
            "epoch":             epoch,
            "train/epoch_loss":  ep_loss,
            "train/epoch_gnorm": ep_gnorm,
            "val/loss":          val["val_loss"],
            "val/EER":           eer,
            "val/phase":         0 if current_phase == "A" else 1,
        })

        # UMAP every 5 epochs
        if epoch % 5 == 0 and logger.enabled:
            log_umap(val["embeddings"], val["labels"],
                     val["languages"], logger, global_step)

        # ── History ──────────────────────────────────────────────────────────
        record = {
            "epoch": epoch, "global_step": global_step,
            "train_loss": ep_loss, "train_acc": ep_acc,
            "val_loss": val["val_loss"], "val_eer": eer,
        }
        history.append(record)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpointing ────────────────────────────────────────────────────
        val_loss = val["val_loss"]
        state = build_checkpoint_state(
            epoch=epoch, model=model, optimizer=optimizer,
            warmup_sched=warmup_sched, cosine_sched=cosine_sched,
            global_step=global_step, best_eer=best_eer,
            best_val_loss=best_val_loss,
            history=history, args=args,
            wandb_run_id=wandb_run_id, phase=current_phase,
        )

        # Always write last.pt (atomic) — safe crash recovery
        save_checkpoint(state, str(ckpt_dir / "last.pt"))

        # Periodic epoch checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(state, str(ckpt_dir / f"speaker_encoder_epoch{epoch:03d}.pt"))

        # Best by val_loss (primary — stable with small val set)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, str(ckpt_dir / "best.pt"))
            print(f"  ✨ New best val_loss: {best_val_loss:.4f}  → saved best.pt")

        # Best by EER (secondary — noisy with small val set, logged only)
        if eer < best_eer:
            best_eer = eer
            state["best_eer"] = best_eer
            save_checkpoint(state, str(ckpt_dir / "best_eer.pt"))
            print(f"  ✨ New best EER: {best_eer:.2f}%  → saved best_eer.pt")

        # Early stopping on EER (more meaningful than val_loss at this stage)
        if eer < best_eer + 0.01:   # improved or within noise floor
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= args.patience:
            print(f"  ⏹  EER no improvement for {args.patience} epochs. Stopping.")
            break

        # EER target (secondary early stop)
        if eer < args.target_eer:
            print(f"  ✅ Target EER {args.target_eer}% reached at epoch {epoch}. Stopping.")
            break

        if _STOP_REQUESTED:
            print("  Stop flag set — last.pt saved. Exiting cleanly.")
            break

    # ── Save final backbone (no classification head) ─────────────────────────
    print("\n[4/5] Saving backbone weights (no AAM-Softmax head)...")
    backbone_path = ckpt_dir / "backbone_final.pt"
    torch.save(model.backbone.state_dict(), backbone_path)
    print(f"  Backbone → {backbone_path}")
    print(f"\n  Training complete.")
    print(f"  Best val_loss : {best_val_loss:.4f}  (weights in best.pt)")
    print(f"  Best EER      : {best_eer:.2f}%       (weights in best_eer.pt)")
    print(f"  To use in VC pipeline:")
    print(f"    cfg = SpeakerEncoderConfig(pretrained_model='{backbone_path}', "
          f"embedding_dim=192, n_speakers=0)")

    logger.finish()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--train_manifest",    required=True,
                   help="Path to training manifest CSV")
    g.add_argument("--val_manifest",      required=True,
                   help="Path to validation manifest CSV")
    g.add_argument("--langs",             nargs="+", default=None,
                   help="Language codes to include (default: all in manifest)")
    g.add_argument("--noise_dir",         default=None,
                   help="Directory of noise .wav files for augmentation (e.g. MUSAN/noise)")
    g.add_argument("--min_dur_s",         type=float, default=2.0)
    g.add_argument("--max_dur_s",         type=float, default=6.0)
    g.add_argument("--num_workers",       type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--pretrained",        action="store_true",
                   help="Load VoxCeleb2 pretrained weights from speechbrain/spkrec-ecapa-voxceleb")
    g.add_argument("--aam_margin",        type=float, default=0.2)
    g.add_argument("--aam_scale",         type=float, default=30.0)

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs",             type=int,   default=30)
    g.add_argument("--phase_b_epoch",      type=int,   default=6,
                   help="Epoch number at which to unfreeze all layers (Phase B)")
    g.add_argument("--lr",                 type=float, default=1e-4)
    g.add_argument("--speakers_per_batch", type=int,   default=32,
                   help="Distinct speakers per batch")
    g.add_argument("--utts_per_speaker",   type=int,   default=2,
                   help="Utterances per speaker per batch (≥2 required for AAM-Softmax)")
    g.add_argument("--val_batch_size",     type=int,   default=64)
    g.add_argument("--target_eer",         type=float, default=5.0,
                   help="Secondary early stop when val EER drops below this (%)")
    g.add_argument("--patience",           type=int,   default=12,
                   help="Stop if val_loss does not improve for this many epochs")
    g.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    g = parser.add_argument_group("Checkpointing")
    g.add_argument("--ckpt_dir",    default="checkpoints/speaker",
                   help="Directory for all checkpoints")
    g.add_argument("--save_every",  type=int, default=5,
                   help="Save a named epoch checkpoint every N epochs "
                        "(last.pt is always saved every epoch)")
    g.add_argument("--resume",      default=None,
                   help="Path to a specific checkpoint to resume from")
    g.add_argument("--auto_resume", action="store_true",
                   help="Automatically resume from last.pt in ckpt_dir if it exists")

    # ── Logging ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Logging")
    g.add_argument("--log_every",       type=int,   default=50,
                   help="Print + W&B log every N steps")
    g.add_argument("--wandb",           action="store_true",
                   help="Enable Weights & Biases logging")
    g.add_argument("--wandb_project",   default="indicvc",
                   help="W&B project name")
    g.add_argument("--wandb_run_name",  default=None,
                   help="W&B run name (auto-generated if not set)")

    train(parser.parse_args())


if __name__ == "__main__":
    main()