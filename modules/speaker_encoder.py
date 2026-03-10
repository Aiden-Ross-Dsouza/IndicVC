"""
IndicVC — Speaker Encoder Module
=================================
Indic ECAPA-TDNN: a speaker encoder fine-tuned on IndicVoices-R.

Architecture
------------
ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation —
Time Delay Neural Network) is the current standard for speaker verification.
We start from SpeechBrain's VoxCeleb2-pretrained checkpoint and fine-tune
on IndicVoices-R for Indic speaker identity.

Pipeline
--------
    waveform (B, samples) @ 16kHz
        ↓
    Mel filterbank (80 bins, 25ms window, 10ms hop)
        ↓
    ECAPA-TDNN backbone (SpeechBrain pretrained → Indic fine-tuned)
        ↓
    Attentive Statistics Pooling  ← variable-length → fixed vector
        ↓
    Speaker embedding (192-dim, L2-normalised)
        ↓
    Used by DiT decoder for timbre conditioning

Design decisions
----------------
1.  Pretrained base: speechbrain/spkrec-ecapa-voxceleb
    Standard choice. VoxCeleb2 pretraining gives robust English speaker
    representations. Fine-tuning on IndicVoices-R adapts to Indic phonemes
    and vocal characteristics.

2.  Embedding dim: 192
    Standard for ECAPA-TDNN. Compact enough for conditioning but rich
    enough to capture timbre differences across Indic speakers.

3.  Training objective: AAM-Softmax (Additive Angular Margin Softmax)
    Loss implemented here but applied by the training loop, not this module.
    AAM-Softmax (m=0.2, s=30) is standard for speaker verification.

4.  L2 normalisation: always applied at inference.
    Speaker similarity in the DiT decoder is cosine distance — L2 norm
    ensures the embedding lives on the unit hypersphere for stable cosine
    similarity computation.

5.  Minimum audio duration: 1 second.
    ECAPA-TDNN with attentive pooling degrades significantly below ~0.5s.
    We enforce 1s minimum and pad shorter inputs.

6.  Language-agnostic by design:
    Speaker identity is language-independent (same person, different language
    → same embedding). This is the zero-shot cross-lingual VC requirement.
    We do NOT condition the speaker encoder on language ID.

Training phases
---------------
Phase 2 (Months 2–4):
    - Freeze ECAPA-TDNN backbone
    - Train only classifier head for speaker ID on IndicVoices-R
    - Verify clustering in UMAP: Indic speakers should separate

Phase 2b:
    - Unfreeze backbone, fine-tune end-to-end with lower LR (1e-5)
    - Verify cross-lingual speaker invariance (same speaker, hi vs ta)

Usage
-----
    from modules.speaker_encoder import SpeakerEncoderConfig, SpeakerEncoder

    cfg = SpeakerEncoderConfig()
    spk_enc = SpeakerEncoder(cfg).to(device)

    # Get speaker embedding from reference audio
    emb = spk_enc(ref_wav)           # (B, 192)

    # AAM-Softmax loss for training
    loss = spk_enc.aam_softmax_loss(emb, speaker_ids)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# torchaudio compatibility patch — MUST run before any speechbrain import.
# SpeechBrain 1.x calls torchaudio.list_audio_backends() at import time,
# but this function was removed in torchaudio 2.x. We patch it here at
# module level so it's in place before _load_pretrained() imports speechbrain.
# ---------------------------------------------------------------------------
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass  # torchaudio not installed — speechbrain will fail later with a clear error

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeakerEncoderConfig:
    """All hyperparameters for the Indic speaker encoder."""

    # Pretrained base checkpoint (SpeechBrain HuggingFace hub ID)
    pretrained_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # Output embedding dimension
    embedding_dim: int = 192

    # Mel filterbank config (must match pretrained model's preprocessing)
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: int = 25       # 25ms window
    hop_length_ms: int = 10       # 10ms hop → 100 frames/sec

    # ECAPA-TDNN architecture (must match pretrained weights)
    channels: int = 1024          # ECAPA channel width
    kernel_sizes: tuple = (5, 3, 3, 3, 1)
    dilations: tuple = (1, 2, 3, 4, 1)
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    global_context: bool = True

    # AAM-Softmax loss params (for training phase 2)
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    n_speakers: int = 1000        # Set to actual IndicVoices-R speaker count

    # Minimum input duration (pad shorter audio to this)
    min_duration_s: float = 1.0

    # Freeze backbone (True during phase 2 warmup, False for full fine-tune)
    freeze_backbone: bool = False


# ---------------------------------------------------------------------------
# ECAPA-TDNN building blocks
# ---------------------------------------------------------------------------

class TDNNBlock(nn.Module):
    """
    1D Temporal Dilated Convolution block (TDNN layer).
    Each TDNN block applies a dilated conv → BatchNorm → ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Res2NetBlock(nn.Module):
    """
    Res2Net block: multi-scale hierarchical residual connections.
    Splits channels into `scale` groups and processes them hierarchically.
    This is the core novelty of ECAPA-TDNN over standard TDNN.

    Reference: Gao et al. "Res2Net: A New Multi-scale Backbone Architecture"
    Applied to speaker verification in ECAPA-TDNN (Desplanques et al. 2020)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, scale: int = 8):
        super().__init__()
        assert out_channels % scale == 0, \
            f"out_channels ({out_channels}) must be divisible by scale ({scale})"
        self.scale = scale
        self.width = out_channels // scale
        padding = dilation * (kernel_size - 1) // 2

        # Input projection
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        # Per-scale convolutions (scale-1 because first slice is passed through)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.width, self.width, kernel_size,
                          dilation=dilation, padding=padding),
                nn.BatchNorm1d(self.width),
                nn.ReLU(),
            )
            for _ in range(scale - 1)
        ])
        # Output projection
        self.conv_out = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        # Residual if dims match
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv_in(x)

        # Split into `scale` chunks along channel dim
        chunks = torch.chunk(out, self.scale, dim=1)
        y = []
        prev = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                y.append(chunk)          # First slice: pass through
            elif i == 1:
                s = self.convs[i - 1](chunk)
                prev = s
                y.append(s)
            else:
                s = self.convs[i - 1](chunk + prev)   # Hierarchical residual
                prev = s
                y.append(s)

        out = self.conv_out(torch.cat(y, dim=1))
        return out + residual


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel responses by modelling inter-channel dependencies.
    Applied after each Res2Net block in ECAPA-TDNN.
    """
    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=-1)              # Global average pool → (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, C)
        return x * s.unsqueeze(-1)      # Recalibrate → (B, C, T)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) — matches SpeechBrain's implementation.

    SpeechBrain's ECAPA-TDNN concatenates four statistics:
        [weighted_mean, weighted_std, global_mean, global_std] → (B, 4*C)

    The attention network uses global context (mean+std of full sequence)
    concatenated with the frame features, then projects to per-channel weights.

    Output: (B, 4*C)  — e.g. 4*1536 = 6144 for the standard ECAPA-TDNN
    """
    def __init__(self, channels: int, attention_channels: int,
                 global_context: bool = True):
        super().__init__()
        self.global_context = global_context
        attn_in = channels * 3 if global_context else channels
        self.attn = nn.Sequential(
            nn.Conv1d(attn_in, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, 4*C) — [weighted_mean, weighted_std, global_mean, global_std]
        """
        # Global statistics (unweighted)
        global_mean = x.mean(dim=-1)      # (B, C)
        global_std  = x.std(dim=-1)       # (B, C)

        if self.global_context:
            g_mean_exp = global_mean.unsqueeze(-1).expand_as(x)
            g_std_exp  = global_std.unsqueeze(-1).expand_as(x)
            attn_input = torch.cat([x, g_mean_exp, g_std_exp], dim=1)  # (B, 3C, T)
        else:
            attn_input = x

        attn_weights = F.softmax(self.attn(attn_input), dim=-1)  # (B, C, T)

        # Weighted statistics
        weighted_mean = (attn_weights * x).sum(dim=-1)            # (B, C)
        weighted_var  = (attn_weights * x ** 2).sum(dim=-1) \
                        - weighted_mean ** 2
        weighted_std  = weighted_var.clamp(min=1e-9).sqrt()       # (B, C)

        # Concatenate all four statistics — matches SpeechBrain output shape
        return torch.cat([weighted_mean, weighted_std,
                          global_mean,   global_std], dim=1)      # (B, 4C)


# ---------------------------------------------------------------------------
# Main ECAPA-TDNN backbone
# ---------------------------------------------------------------------------

class ECAPATDNNBackbone(nn.Module):
    """
    Full ECAPA-TDNN backbone.

    Architecture (matching SpeechBrain's spkrec-ecapa-voxceleb):
        Input: mel filterbank (B, 80, T)
        ↓  TDNNBlock:  80 → C
        ↓  Res2Net + SE blocks × 4 (with skip connections)
        ↓  TDNNBlock:  4C → 1536  (aggregation)
        ↓  AttentiveStatisticsPooling: 1536 → 3072
        ↓  Linear: 3072 → embedding_dim

    The 4 Res2Net+SE blocks use different kernel sizes and dilations for
    multi-scale temporal context, a key design choice of ECAPA-TDNN.
    """

    def __init__(self, cfg: SpeakerEncoderConfig):
        super().__init__()
        C = cfg.channels
        E = cfg.embedding_dim

        # Input TDNN
        self.input_block = TDNNBlock(cfg.n_mels, C, kernel_size=5, dilation=1)

        # Multi-scale Res2Net + SE blocks
        self.res2net_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        in_ch = C
        for i in range(4):
            k = cfg.kernel_sizes[i + 1] if i + 1 < len(cfg.kernel_sizes) else 3
            d = cfg.dilations[i + 1] if i + 1 < len(cfg.dilations) else i + 1
            self.res2net_blocks.append(
                Res2NetBlock(in_ch, C, kernel_size=k, dilation=d,
                             scale=cfg.res2net_scale)
            )
            self.se_blocks.append(SEBlock(C, cfg.se_channels))
            in_ch = C

        # Aggregation layer: concatenate all Res2Net outputs
        # 4 blocks × C channels + input block output
        self.aggregation = TDNNBlock(C * 5, 1536, kernel_size=1)

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(
            channels=1536,
            attention_channels=cfg.attention_channels,
            global_context=cfg.global_context,
        )
        self.asp_bn = nn.BatchNorm1d(6144)  # 4 × 1536  (matches SpeechBrain)

        # Embedding projection
        self.fc = nn.Linear(6144, E)
        self.bn = nn.BatchNorm1d(E)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 80, T) mel filterbank features
        Returns: (B, embedding_dim) L2-normalised speaker embedding
        """
        # Input TDNN
        x = self.input_block(mel)       # (B, C, T)
        block_outputs = [x]

        # Res2Net + SE blocks with skip connection accumulation
        for res2net, se in zip(self.res2net_blocks, self.se_blocks):
            x = se(res2net(x))          # (B, C, T)
            block_outputs.append(x)

        # Aggregate: cat all block outputs along channel dim
        x = torch.cat(block_outputs, dim=1)   # (B, 5C, T)
        x = self.aggregation(x)               # (B, 1536, T)

        # Attentive statistics pooling → utterance embedding
        x = self.asp(x)                       # (B, 6144)
        x = self.asp_bn(x)

        # Project to embedding dim
        x = self.fc(x)                        # (B, embedding_dim)
        x = self.bn(x)

        # L2 normalise — embeddings live on unit hypersphere
        x = F.normalize(x, p=2, dim=-1)

        return x


# ---------------------------------------------------------------------------
# AAM-Softmax loss (for Phase 2 training)
# ---------------------------------------------------------------------------

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace / AAM-Softmax).

    Standard loss function for speaker verification training.
    Adds an angular margin m to the target class angle, which pushes
    embeddings of the same speaker closer together on the hypersphere
    and increases inter-class angular distance.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for
    Deep Face Recognition" CVPR 2019. Applied to SV in many papers.

    Parameters
    ----------
    embedding_dim : int
        Speaker embedding dimension (192).
    n_speakers : int
        Number of training speakers. Set to actual count in IndicVoices-R.
    margin : float
        Angular margin m (default 0.2). Larger = harder training.
    scale : float
        Logit scale s (default 30.0). Standard value for 192-dim embeddings.
    """

    def __init__(self, embedding_dim: int, n_speakers: int,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(
            torch.FloatTensor(n_speakers, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(π - m) and sin(π - m)
        self.cos_m = math.cos(math.pi - margin)
        self.sin_m = math.sin(math.pi - margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings : (B, embedding_dim) — L2-normalised speaker embeddings
        labels     : (B,) — integer speaker IDs

        Returns scalar cross-entropy loss with angular margin.
        """
        # Normalise weight matrix rows to unit hypersphere
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity between embeddings and all speaker centres
        cos_theta = F.linear(embeddings, weight)              # (B, n_speakers)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        sin_theta = (1.0 - cos_theta ** 2).clamp(min=1e-9).sqrt()

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        # For numerical stability: if cos(theta) < threshold, use linear approx
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Apply margin only to target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits *= self.scale

        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Main SpeakerEncoder module
# ---------------------------------------------------------------------------

class SpeakerEncoder(nn.Module):
    """
    Indic ECAPA-TDNN Speaker Encoder.

    Two operating modes:
    1.  from_pretrained=True (default, Phase 2 start):
        Loads SpeechBrain's VoxCeleb-pretrained ECAPA-TDNN weights.
        The backbone weights are directly compatible — SpeechBrain exports
        standard PyTorch state_dict that maps onto our ECAPATDNNBackbone.

    2.  from_scratch=False (for ablation):
        Initialises randomly. Useful for confirming pretraining benefit.

    The AAMSoftmax classifier head is always randomly initialised since
    IndicVoices-R speaker IDs differ from VoxCeleb speaker IDs.

    Forward pass returns L2-normalised speaker embeddings (B, 192).
    Call .aam_softmax_loss(emb, labels) separately in the training loop.
    """

    def __init__(self, cfg: SpeakerEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Build backbone
        self.backbone = ECAPATDNNBackbone(cfg)

        # Mel filterbank extractor (no learnable params)
        hop = cfg.hop_length_ms * cfg.sample_rate // 1000   # samples
        win = cfg.win_length_ms * cfg.sample_rate // 1000   # samples
        self.mel_extractor = torchaudio_compatible_mel(
            sample_rate=cfg.sample_rate,
            n_fft=512,
            win_length=win,
            hop_length=hop,
            n_mels=cfg.n_mels,
            f_min=20.0,
            f_max=7600.0,
        )

        # AAM-Softmax head (only used during Phase 2 training)
        self.classifier = AAMSoftmax(
            embedding_dim=cfg.embedding_dim,
            n_speakers=cfg.n_speakers,
            margin=cfg.aam_margin,
            scale=cfg.aam_scale,
        )

        # Load pretrained weights
        self._pretrained_loaded = False
        if cfg.pretrained_model:
            self._load_pretrained(cfg.pretrained_model)

        # Freeze backbone if requested (Phase 2 warmup)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("[SpeakerEncoder] Backbone frozen. Only classifier head trains.")

    # ── Mel extraction ───────────────────────────────────────────────────────

    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, samples) @ 16kHz
        Returns:  (B, 80, T) log mel filterbank
        """
        # Pad to minimum duration
        min_samples = int(self.cfg.min_duration_s * self.cfg.sample_rate)
        if waveform.shape[-1] < min_samples:
            pad_len = min_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))

        mel = self.mel_extractor(waveform)           # (B, 80, T)
        mel = torch.log(mel.clamp(min=1e-9))         # Log mel
        return mel

    # ── Pretrained weight loading ────────────────────────────────────────────

    def _load_pretrained(self, model_id: str):
        """
        Load SpeechBrain ECAPA-TDNN pretrained weights directly from
        HuggingFace, bypassing SpeechBrain's import chain entirely.

        SpeechBrain 1.x uses a lazy module loader that breaks with
        torchaudio 2.x regardless of monkey-patching. Instead we:
          1. Download the checkpoint file directly via huggingface_hub
          2. Load it as a raw PyTorch state_dict
          3. Map keys onto our ECAPATDNNBackbone

        The checkpoint is ~80MB and cached at
        ~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/
        """
        try:
            from huggingface_hub import hf_hub_download
            print(f"[SpeakerEncoder] Loading pretrained weights from: {model_id}")
            print(f"  Downloading embedding_model.ckpt (~80MB, cached after first run)...")

            ckpt_path = hf_hub_download(
                repo_id=model_id,
                filename="embedding_model.ckpt",
            )
            print(f"  Downloaded: {ckpt_path}")

            # Load raw state dict — SpeechBrain saves standard PyTorch checkpoints
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # SpeechBrain checkpoint format: may be bare state_dict or
            # wrapped in {"model": state_dict} or {"embedding_model": state_dict}
            if isinstance(raw, dict):
                if "model" in raw:
                    sb_state = raw["model"]
                elif "embedding_model" in raw:
                    sb_state = raw["embedding_model"]
                else:
                    # Bare state dict
                    sb_state = raw
            else:
                raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

            print(f"  Checkpoint keys: {len(sb_state)}")
            self._transfer_state_dict(sb_state)
            self._pretrained_loaded = True
            print(f"  ✅ Pretrained weights loaded successfully")

        except Exception as e:
            print(
                f"[SpeakerEncoder] WARNING: Could not load pretrained weights.\n"
                f"  Error: {e}\n"
                f"  Backbone initialised randomly. Training from scratch.\n"
                f"  This is fine for smoke tests — use --pretrained for real training."
            )

    def _transfer_state_dict(self, sb_state: dict):
        """
        Transfer SpeechBrain checkpoint weights to our ECAPATDNNBackbone.

        Exact key mapping derived from tools/inspect_ckpt_keys.py:

        SpeechBrain                              → Ours
        ────────────────────────────────────────────────────────────
        blocks.0.conv.conv.*                     → input_block.conv.*
        blocks.0.norm.norm.*                     → input_block.bn.*
        blocks.N.tdnn1.conv.conv.*               → res2net_blocks.N-1.conv_in.0.*
        blocks.N.tdnn1.norm.norm.*               → res2net_blocks.N-1.conv_in.1.*
        blocks.N.res2net_block.blocks.M.conv.*   → res2net_blocks.N-1.convs.M.0.*
        blocks.N.res2net_block.blocks.M.norm.*   → res2net_blocks.N-1.convs.M.1.*
        blocks.N.tdnn2.conv.conv.*               → res2net_blocks.N-1.conv_out.0.*
        blocks.N.tdnn2.norm.norm.*               → res2net_blocks.N-1.conv_out.1.*
        blocks.N.res2net_block_tdnn.conv.conv.*  → res2net_blocks.N-1.residual.*
        blocks.N.se_block.fc1.*                  → se_blocks.N-1.fc1.*
        blocks.N.se_block.fc2.*                  → se_blocks.N-1.fc2.*
        mfa.conv.conv.*                          → aggregation.conv.*
        mfa.norm.norm.*                          → aggregation.bn.*
        asp.tdnn.conv.conv.*                     → asp.attn.0.*
        asp.fc.conv.conv.*                       → asp.attn.2.*
        fc.conv.conv.*                           → fc.*  (squeeze kernel dim)
        fc.norm.norm.*                           → bn.*
        """

        def translate(sb_key: str):
            p = sb_key.split(".")

            # input_block
            if p[0] == "blocks" and p[1] == "0":
                if p[2] == "conv":   return f"input_block.conv.{p[-1]}"
                if p[2] == "norm":   return f"input_block.bn.{p[-1]}"

            # res2net_blocks + se_blocks  (blocks.1 .. blocks.4)
            if p[0] == "blocks" and p[1] in ("1","2","3","4"):
                bi = int(p[1]) - 1
                if p[2] == "tdnn1":
                    if p[3] == "conv": return f"res2net_blocks.{bi}.conv_in.0.{p[-1]}"
                    if p[3] == "norm": return f"res2net_blocks.{bi}.conv_in.1.{p[-1]}"
                if p[2] == "res2net_block" and p[3] == "blocks":
                    m = p[4]
                    if p[5] == "conv": return f"res2net_blocks.{bi}.convs.{m}.0.{p[-1]}"
                    if p[5] == "norm": return f"res2net_blocks.{bi}.convs.{m}.1.{p[-1]}"
                if p[2] == "tdnn2":
                    if p[3] == "conv": return f"res2net_blocks.{bi}.conv_out.0.{p[-1]}"
                    if p[3] == "norm": return f"res2net_blocks.{bi}.conv_out.1.{p[-1]}"
                if p[2] == "res2net_block_tdnn":
                    if p[3] == "conv": return f"res2net_blocks.{bi}.residual.{p[-1]}"
                if p[2] == "se_block":
                    # SpeechBrain uses conv1/conv2 (Conv1d), we use fc1/fc2 (Linear)
                    # conv1.conv.weight (128,1024,1) → fc1.weight (128,1024) — squeeze
                    # conv2.conv.weight (1024,128,1) → fc2.weight (1024,128) — squeeze
                    conv = p[3]  # conv1 or conv2
                    fc   = "fc1" if conv == "conv1" else "fc2"
                    if p[4] == "conv" and p[-1] in ("weight", "bias"):
                        return f"se_blocks.{bi}.{fc}.{p[-1]}"

            # aggregation (mfa)
            if p[0] == "mfa":
                if p[1] == "conv": return f"aggregation.conv.{p[-1]}"
                if p[1] == "norm": return f"aggregation.bn.{p[-1]}"

            # ASP attention network
            if p[0] == "asp":
                if p[1] == "tdnn" and p[2] == "conv": return f"asp.attn.0.{p[-1]}"
                if p[1] == "tdnn" and p[2] == "norm": return None  # no BN in our attn
                if p[1] == "fc"   and p[2] == "conv": return f"asp.attn.2.{p[-1]}"
                # asp.conv.conv.* — SpeechBrain's final ASP projection (not in our design)
                # We skip this; our attn.2 already projects to channels
                if p[1] == "conv": return None

            # asp_bn — SpeechBrain uses asp_bn.norm.*, we use asp_bn.*
            if p[0] == "asp_bn":
                if p[1] == "norm": return f"asp_bn.{p[-1]}"

            # final FC + BN — SpeechBrain: fc.conv.conv.weight (192,6144,1)
            if p[0] == "fc":
                if p[1] == "conv" and p[2] == "conv": return f"fc.{p[-1]}"
                if p[1] == "conv" and len(p) == 3:    return f"fc.{p[-1]}"  # fc.conv.bias
                if p[1] == "norm":                     return f"bn.{p[-1]}"

            return None

        our_state = self.backbone.state_dict()
        transferred = skipped_no_map = skipped_shape = 0

        for sb_key, sb_val in sb_state.items():
            our_key = translate(sb_key)
            if our_key is None or our_key not in our_state:
                skipped_no_map += 1
                continue
            our_t = our_state[our_key]
            v = sb_val
            # SpeechBrain Conv1d-as-Linear: shape (out, in, 1) → squeeze for nn.Linear
            if v.dim() == 3 and v.shape[-1] == 1 and our_t.dim() == 2:
                v = v.squeeze(-1)
            if v.shape == our_t.shape:
                our_state[our_key].copy_(v)
                transferred += 1
            else:
                skipped_shape += 1

        self.backbone.load_state_dict(our_state)
        total = len(our_state)
        pct = transferred / total * 100
        print(f"  Transferred   : {transferred}/{total} ({pct:.0f}%)")
        print(f"  No mapping    : {skipped_no_map}")
        print(f"  Shape mismatch: {skipped_shape}")
        if pct < 70:
            print(f"  ⚠️  Transfer <70% — run tools/inspect_ckpt_keys.py to debug")

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised speaker embedding.

        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio at 16kHz. Shape (B, samples).
            Minimum recommended duration: 1 second (cfg.min_duration_s).
            Longer is better — 3–10s is typical for speaker verification.

        Returns
        -------
        torch.Tensor
            Speaker embedding. Shape (B, embedding_dim).
            L2-normalised — lives on unit hypersphere.
            Use cosine similarity for speaker comparison.
        """
        mel = self._extract_mel(waveform)      # (B, 80, T)
        emb = self.backbone(mel)               # (B, embedding_dim)
        return emb

    def aam_softmax_loss(self, embeddings: torch.Tensor,
                         speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute AAM-Softmax training loss.

        Call this in the training loop during Phase 2:
            emb = speaker_encoder(wav)
            loss = speaker_encoder.aam_softmax_loss(emb, speaker_ids)
            loss.backward()

        Parameters
        ----------
        embeddings : torch.Tensor
            (B, embedding_dim) — output of forward()
        speaker_ids : torch.Tensor
            (B,) — integer speaker ID labels [0, n_speakers)

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return self.classifier(embeddings, speaker_ids)

    def similarity(self, emb_a: torch.Tensor,
                   emb_b: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between two speaker embeddings.

        Both inputs must be L2-normalised (which forward() guarantees).
        Returns value in [-1, 1] where 1 = same speaker, -1 = opposite.
        Typical same-speaker threshold: ~0.25 (dataset-dependent).

        Parameters
        ----------
        emb_a, emb_b : torch.Tensor
            (B, embedding_dim) or (embedding_dim,)

        Returns
        -------
        torch.Tensor
            (B,) cosine similarity scores.
        """
        if emb_a.dim() == 1:
            emb_a = emb_a.unsqueeze(0)
        if emb_b.dim() == 1:
            emb_b = emb_b.unsqueeze(0)
        return F.cosine_similarity(emb_a, emb_b, dim=-1)


# ---------------------------------------------------------------------------
# Mel extractor factory (avoids torchaudio.load dependency issues on Windows)
# ---------------------------------------------------------------------------

def torchaudio_compatible_mel(
    sample_rate: int, n_fft: int, win_length: int, hop_length: int,
    n_mels: int, f_min: float, f_max: float,
) -> nn.Module:
    """
    Create a mel filterbank extractor module.

    Tries torchaudio.transforms.MelSpectrogram first (preferred).
    Falls back to a pure-PyTorch implementation if torchaudio is broken
    (e.g. torchcodec/FFmpeg issues on Windows only affect audio loading,
    not transforms — but we handle both cases).
    """
    try:
        import torchaudio.transforms as T
        return T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
    except Exception:
        # Pure-PyTorch fallback using torch.stft + manual filterbank
        return _PureTorchMelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )


class _PureTorchMelSpectrogram(nn.Module):
    """
    Pure-PyTorch mel spectrogram fallback (no torchaudio dependency).
    Used only if torchaudio.transforms is unavailable.
    """
    def __init__(self, sample_rate, n_fft, win_length, hop_length,
                 n_mels, f_min, f_max):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer(
            "window", torch.hann_window(win_length)
        )
        self.register_buffer(
            "mel_fb", _build_mel_filterbank(
                sample_rate, n_fft, n_mels, f_min, f_max
            )
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B = waveform.shape[0]
        specs = []
        for i in range(B):
            stft = torch.stft(
                waveform[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
            power = stft.abs() ** 2        # (F, T)
            mel = self.mel_fb @ power      # (n_mels, T)
            specs.append(mel)
        return torch.stack(specs, dim=0)   # (B, n_mels, T)


def _build_mel_filterbank(sample_rate: int, n_fft: int,
                          n_mels: int, f_min: float,
                          f_max: float) -> torch.Tensor:
    """Build a triangular mel filterbank matrix (n_mels, n_fft//2+1)."""
    def hz_to_mel(f):  return 2595 * math.log10(1 + f / 700)
    def mel_to_hz(m):  return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
    bin_points = torch.floor(hz_points / (sample_rate / n_fft)).long()

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        f_left  = bin_points[m - 1]
        f_center= bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filterbank


# ---------------------------------------------------------------------------
# Factory function (YAML-driven)
# ---------------------------------------------------------------------------

def build_speaker_encoder(config: dict) -> SpeakerEncoder:
    """
    Build a SpeakerEncoder from a config dict (loaded from model.yaml).

    Example
    -------
        import yaml
        with open("config/model.yaml") as f:
            cfg = yaml.safe_load(f)
        encoder = build_speaker_encoder(cfg["speaker_encoder"])
    """
    se_cfg = config.get("speaker_encoder", config)
    spk_cfg = SpeakerEncoderConfig(
        pretrained_model=se_cfg.get("pretrained_model",
                                    "speechbrain/spkrec-ecapa-voxceleb"),
        embedding_dim=se_cfg.get("embedding_dim", 192),
        n_speakers=se_cfg.get("n_speakers", 1000),
        aam_margin=se_cfg.get("aam_margin", 0.2),
        aam_scale=se_cfg.get("aam_scale", 30.0),
        freeze_backbone=se_cfg.get("freeze_backbone", False),
    )
    return SpeakerEncoder(spk_cfg)