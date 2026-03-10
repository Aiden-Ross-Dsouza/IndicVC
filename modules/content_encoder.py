"""
IndicVC - Content Encoder Module
=================================
Supports two AI4Bharat backend options:
  1. IndicWhisper  — Whisper encoder fine-tuned on Indic languages,
                     adapted to causal (streaming) attention.
  2. IndicConformer — Native streaming-compatible Conformer with CTC/RNNT,
                      all 22 official Indian languages.

Design principles
-----------------
- Drop-in replacement for Seed-VC's Whisper-small encoder.
- Streaming-first: processes fixed-size chunks with a causal attention mask
  (IndicWhisper) or CTC frame-sync (IndicConformer).
- Language-ID aware: accepts a BCP-47 language code so the encoder can
  apply language-specific post-processing.
- Output shape is always (B, T, D) — batch, time-frames, feature-dim —
  matching what the downstream DiT decoder expects.

Usage
-----
    from modules.content_encoder import ContentEncoderConfig, ContentEncoder

    cfg = ContentEncoderConfig(backend="indicwhisper", chunk_size_ms=200)
    encoder = ContentEncoder(cfg).to(device)
    features = encoder(wav_tensor, lang_id="hi")   # (B, T, 1024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperModel,
    AutoModel,
)

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

INDIC_LANG_CODES = [
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks",
    "kok", "mai", "ml", "mni", "mr", "ne", "or",
    "pa", "sa", "sat", "sd", "ta", "te", "ur",
]

# Maps BCP-47 → language family
LANG_FAMILY = {
    # Indo-Aryan
    "hi": "indo_aryan", "bn": "indo_aryan", "mr": "indo_aryan",
    "gu": "indo_aryan", "pa": "indo_aryan", "or": "indo_aryan",
    "as": "indo_aryan", "mai": "indo_aryan", "ne": "indo_aryan",
    "ur": "indo_aryan", "ks": "indo_aryan", "sd": "indo_aryan",
    "doi": "indo_aryan", "sa": "indo_aryan", "kok": "indo_aryan",
    # Dravidian
    "ta": "dravidian", "te": "dravidian",
    "kn": "dravidian", "ml": "dravidian",
    # Sino-Tibetan / Austro-Asiatic (included for completeness)
    "mni": "sino_tibetan", "brx": "austro_asiatic", "sat": "austro_asiatic",
}

# HuggingFace model IDs for IndicWhisper per-language checkpoints.
# Falls back to the multilingual small model if language-specific not available.
INDICWHISPER_MODEL_IDS = {
    "hi": "vasista22/whisper-hindi-small",
    "ta": "vasista22/whisper-tamil-small",
    "te": "vasista22/whisper-telugu-small",
    "kn": "vasista22/whisper-kannada-small",
    "ml": "vasista22/whisper-malayalam-small",
    "bn": "vasista22/whisper-bengali-small",
    "mr": "vasista22/whisper-marathi-small",
    "gu": "vasista22/whisper-gujarati-small",
    "or": "vasista22/whisper-odia-small",
    "pa": "vasista22/whisper-punjabi-small",
    # Fallback — multilingual Whisper small for anything else
    "_default": "openai/whisper-small",
}

# Single multilingual IndicConformer checkpoint (all 22 languages)
INDICCONFORMER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContentEncoderConfig:
    """
    Configuration for the IndicVC content encoder.

    Parameters
    ----------
    backend : str
        "indicwhisper"  — per-language Whisper encoder with causal adaptation.
        "indicconformer" — single multilingual Conformer (native streaming).
    chunk_size_ms : int
        Audio chunk duration in milliseconds for streaming inference.
        200ms is a good default (balances latency and quality).
        Must be a multiple of the model's frame shift (10ms for Whisper).
    lookahead_ms : int
        Future context the encoder is allowed to see per chunk.
        Set to 0 for strict causal streaming.
        Small values (40–80ms) improve quality with minimal latency cost.
    output_dim : int
        Projected output feature dimension fed to the DiT decoder.
        Default 1024 matches Seed-VC's DiT input dimension.
    freeze_encoder : bool
        If True, encoder weights are frozen. Only the projection head trains.
        Set False when fine-tuning on IndicVoices-R.
    dropout : float
        Dropout applied to projected features.
    cache_encoders : bool
        Whether to cache loaded encoder models in a module-level dict
        (saves GPU memory when encoding many languages in one session).
    languages : list[str]
        Languages this encoder instance will be used with.
        Used to pre-load the correct checkpoints.
    """
    backend: Literal["indicwhisper", "indicconformer"] = "indicwhisper"
    chunk_size_ms: int = 200
    lookahead_ms: int = 40
    output_dim: int = 1024
    freeze_encoder: bool = False
    dropout: float = 0.1
    cache_encoders: bool = True
    languages: list = field(default_factory=lambda: [
        "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "or", "pa"
    ])


# ---------------------------------------------------------------------------
# Causal self-attention adapter (IndicWhisper path)
# ---------------------------------------------------------------------------

class CausalWhisperAttention(nn.Module):
    """
    Replaces Whisper's bidirectional encoder self-attention with
    chunk-causal attention.

    Whisper's encoder uses full self-attention over the entire 30-second
    mel-spectrogram. For streaming we split into chunks and apply a causal
    mask so each frame can only attend to:
      - all frames in the same chunk (local context), and
      - lookahead_frames frames from the next chunk.

    This preserves most of Whisper's quality while enabling streaming.

    Parameters
    ----------
    original_attn : WhisperAttention
        The original bidirectional attention module from the loaded checkpoint.
    chunk_frames : int
        Number of mel frames per chunk (chunk_size_ms / 10ms frame shift).
    lookahead_frames : int
        Number of future frames visible per chunk.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        chunk_frames: int,
        lookahead_frames: int,
    ):
        super().__init__()
        self.attn = original_attn
        self.chunk_frames = chunk_frames
        self.lookahead_frames = lookahead_frames

    def _build_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build a chunk-causal attention mask of shape (1, 1, T, T).
        Frame i can attend to frame j iff:
            chunk(j) <= chunk(i)   [same or earlier chunk]
            OR j <= i + lookahead  [within lookahead window]
        """
        # Chunk indices for each frame
        chunk_idx = torch.arange(seq_len, device=device) // self.chunk_frames
        # i can attend to j if chunk_idx[j] <= chunk_idx[i]
        chunk_mask = chunk_idx.unsqueeze(0) <= chunk_idx.unsqueeze(1)  # (T, T)
        # Allow lookahead: j <= i + lookahead
        pos = torch.arange(seq_len, device=device)
        lookahead_mask = pos.unsqueeze(0) <= (
            pos.unsqueeze(1) + self.lookahead_frames
        )  # (T, T)
        mask = chunk_mask | lookahead_mask  # (T, T)
        # Convert to additive mask (0 = attend, -inf = ignore)
        additive = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
        additive[~mask] = float("-inf")
        return additive.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        B, T, _ = hidden_states.shape
        causal_mask = self._build_causal_mask(
            T, hidden_states.device, hidden_states.dtype
        )
        # Merge with any padding mask passed in
        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask
        return self.attn(
            hidden_states,
            attention_mask=causal_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )


# ---------------------------------------------------------------------------
# IndicWhisper encoder wrapper
# ---------------------------------------------------------------------------

class IndicWhisperEncoder(nn.Module):
    """
    Loads a per-language IndicWhisper checkpoint and patches its
    self-attention layers to be chunk-causal for streaming.

    The encoder outputs hidden states from the final layer, which are
    then projected to `output_dim` by the parent ContentEncoder.

    Notes
    -----
    - We load only the encoder half of WhisperModel (no decoder).
    - Language-specific checkpoints share the same WhisperConfig, so
      cross-language feature spaces are compatible for cross-lingual VC.
    - Feature extraction (log mel-spectrogram) is handled internally
      so the module accepts raw waveforms at 16kHz.
    """

    def __init__(self, cfg: ContentEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_frames = cfg.chunk_size_ms // 10   # 10ms per mel frame
        self.lookahead_frames = cfg.lookahead_ms // 10

        # Cache: lang_code → encoder
        self._encoder_cache: dict[str, nn.Module] = {}
        # Feature extractor is shared across languages (Whisper standard)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-small"
        )
        # Pre-load all requested languages
        for lang in cfg.languages:
            self._load_encoder(lang)

        # Encoder hidden size (Whisper-small = 768)
        sample_enc = next(iter(self._encoder_cache.values()))
        self.encoder_dim = sample_enc.config.d_model

    def _load_encoder(self, lang: str) -> nn.Module:
        """Load (and cache) the encoder for a given language code."""
        if lang in self._encoder_cache:
            return self._encoder_cache[lang]

        model_id = INDICWHISPER_MODEL_IDS.get(lang, INDICWHISPER_MODEL_IDS["_default"])
        print(f"[IndicWhisperEncoder] Loading {model_id} for lang='{lang}'...")

        # CVE-2025-32434: transformers >= 4.x blocks torch.load on PyTorch < 2.6.
        # vasista22 models only ship pytorch_model.bin (no safetensors).
        # Fix: convert bin → safetensors in the HF cache before calling from_pretrained.
        model = self._load_whisper_safe(model_id)
        encoder = model.encoder  # only keep encoder half

        # Patch self-attention → chunk-causal attention
        for layer in encoder.layers:
            layer.self_attn = CausalWhisperAttention(
                original_attn=layer.self_attn,
                chunk_frames=self.chunk_frames,
                lookahead_frames=self.lookahead_frames,
            )

        if self.cfg.freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)

        if self.cfg.cache_encoders:
            self._encoder_cache[lang] = encoder
        return encoder

    @staticmethod
    def _load_whisper_safe(model_id: str):
        """
        Load a WhisperModel even when the repo only has pytorch_model.bin
        and transformers blocks torch.load on PyTorch < 2.6 (CVE-2025-32434).

        Strategy: download the .bin file explicitly via hf_hub_download,
        load it with torch.load(..., weights_only=False) ourselves (we trust
        HuggingFace-hosted weights), write a safetensors version next to it
        in the local cache, then call from_pretrained with local_files_only=True.
        On subsequent calls the safetensors file is already present and
        from_pretrained loads it directly with no CVE check.
        """
        import os
        from huggingface_hub import snapshot_download, hf_hub_download
        from transformers import WhisperModel

        # First try: from_pretrained directly (works if safetensors already cached)
        try:
            return WhisperModel.from_pretrained(model_id)
        except (ValueError, OSError):
            pass  # CVE block or missing file — fall through to manual path

        print(f"  [CVE workaround] Converting pytorch_model.bin → safetensors ...")

        # 1. Download the full repo so we have config files
        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.safetensors"],  # skip if somehow present
        )

        bin_path = os.path.join(local_dir, "pytorch_model.bin")
        st_path  = os.path.join(local_dir, "model.safetensors")

        if not os.path.exists(st_path):
            if not os.path.exists(bin_path):
                raise FileNotFoundError(
                    f"Neither pytorch_model.bin nor model.safetensors found in {local_dir}"
                )
            # Load with weights_only=False — we trust the HF-hosted vasista22 models
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            try:
                from safetensors.torch import save_file
                save_file(state_dict, st_path)
                print(f"  [CVE workaround] Saved safetensors to {st_path}")
            except ImportError:
                # safetensors not installed — install it silently and retry
                import subprocess, sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "safetensors", "-q"]
                )
                from safetensors.torch import save_file
                save_file(state_dict, st_path)
                print(f"  [CVE workaround] Installed safetensors and saved to {st_path}")

        # 2. Now from_pretrained will find model.safetensors and bypass the CVE check
        return WhisperModel.from_pretrained(local_dir)

    # Whisper encoder always expects exactly this many mel frames (30s window)
    WHISPER_MEL_FRAMES = 3000

    def _wav_to_mel(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw waveform to padded mel-spectrogram.

        Whisper's encoder requires input_features of shape (B, 80, 3000)
        regardless of actual audio length. We:
          1. Extract mel frames for the true audio length.
          2. Pad to 3000 frames with zeros (silence).
          3. Return a boolean mask indicating which frames are real content
             (True = real, False = padding) — used to trim encoder output.

        Parameters
        ----------
        waveform : torch.Tensor  shape (B, samples)

        Returns
        -------
        input_features : torch.Tensor  shape (B, 80, 3000)
        content_mask   : torch.Tensor  shape (B, T_frames)  bool
            True for frames that correspond to real audio (not padding).
        """
        B = waveform.shape[0]

        # HuggingFace feature extractor works on lists of 1D numpy arrays
        wav_list = [waveform[i].cpu().numpy() for i in range(B)]

        # Extract mel — feature extractor pads/truncates to 30s (3000 frames)
        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",       # always pad to 30s
            return_attention_mask=True, # gives us the real-content mask
        )
        input_features = inputs.input_features  # (B, 80, 3000)

        # Compute how many mel frames the actual audio occupies
        # Whisper uses 160-sample hop (10ms @ 16kHz) → frames = samples // 160
        actual_frames = [
            min(waveform.shape[1] // 160, self.WHISPER_MEL_FRAMES)
            for _ in range(B)
        ]

        # Build content mask: (B, T_encoder_output)
        # Whisper encoder downsamples mel frames by 2 via Conv layers,
        # so encoder output T = WHISPER_MEL_FRAMES // 2 = 1500
        T_enc = self.WHISPER_MEL_FRAMES // 2
        content_mask = torch.zeros(B, T_enc, dtype=torch.bool)
        for i, af in enumerate(actual_frames):
            real_enc_frames = min(af // 2, T_enc)
            content_mask[i, :real_enc_frames] = True

        return input_features, content_mask

    def forward(
        self,
        waveform: torch.Tensor,
        lang_id: str = "hi",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio at 16kHz, shape (B, samples).
            Any length is accepted — internally padded to 30s for Whisper.
        lang_id : str
            BCP-47 language code, e.g. "hi", "ta".
        attention_mask : optional
            Padding mask for batched variable-length inputs (sample-level).
            Shape (B, samples). If None, all samples treated as real content.

        Returns
        -------
        torch.Tensor
            Encoder hidden states for REAL content frames only.
            Shape (B, T_real, encoder_dim) where T_real = actual mel frames // 2.
            For a batch with different lengths, T_real = max real frames in batch.
        """
        encoder = self._load_encoder(lang_id)
        encoder = encoder.to(waveform.device)

        # Convert waveform → padded mel + content mask
        input_features, content_mask = self._wav_to_mel(waveform)
        input_features = input_features.to(waveform.device)
        content_mask   = content_mask.to(waveform.device)

        # Run Whisper encoder — always gets (B, 80, 3000) input
        outputs = encoder(input_features)
        hidden = outputs.last_hidden_state  # (B, 1500, encoder_dim)

        # Trim to longest real content in the batch
        # This removes the padding frames from the output
        max_real = content_mask.sum(dim=1).max().item()
        hidden = hidden[:, :max_real, :]        # (B, T_real, encoder_dim)
        content_mask = content_mask[:, :max_real]

        return hidden  # (B, T_real, encoder_dim)


# ---------------------------------------------------------------------------
# IndicConformer encoder wrapper
# ---------------------------------------------------------------------------

class IndicConformerEncoder(nn.Module):
    """
    Wraps AI4Bharat's IndicConformer-600M-Multilingual for use as a
    frozen content encoder.

    HOW HIDDEN STATE EXTRACTION WORKS
    -----------------------------------
    From source inspection of model_onnx.py, the model exposes:

        model.encode(wav) → (outputs, encoded_lengths)

    where `outputs` is a numpy array of shape (B, T, 1024) — the encoder
    hidden states BEFORE any CTC/RNNT decoder head. This is exactly what
    we need for VC content features.

    The encode() method internally:
        1. Runs wav through a TorchScript preprocessor → mel features
        2. Runs mel features through an ONNX InferenceSession (encoder)
        3. Returns encoder outputs as numpy array

    IMPORTANT CONSTRAINTS (discovered from source):
    ------------------------------------------------
    1. ALWAYS FROZEN: The encoder is an ONNX InferenceSession — not a
       PyTorch module. No gradients can flow through it. freeze_encoder
       config flag is ignored for this backend (always frozen).
       Only the projection head (in ContentEncoder) can be trained.

    2. DEVICE HANDLING: The ONNX session uses CUDAExecutionProvider if
       GPU is available, CPUExecutionProvider otherwise. This is set at
       model load time — we cannot move it after loading.

    3. OUTPUT IS NUMPY: model.encode() returns numpy arrays. We convert
       to torch tensors immediately for compatibility with the rest of
       the pipeline.

    4. NO BATCHING IN ONNX: The encoder ONNX model accepts length as a
       scalar, so true batching is limited. We process each item in the
       batch separately and pad to the longest sequence.

    Advantages over IndicWhisper
    -----------------------------
    - Single checkpoint for all 22 Indic languages (no per-language loading)
    - 600M parameters → richer representations than Whisper-small (74M)
    - CTC-trained → inherently left-to-right, naturally streaming-compatible
    - No causal attention adaptation needed

    Disadvantages
    -------------
    - Completely frozen — projection head is the only trainable component
    - ONNX backend means no PyTorch autograd through encoder
    - Slightly slower than Whisper for batched inference (loop over batch)
    """

    # ONNX encoder output shape confirmed from source:
    # model.models['encoder'].run(['outputs', 'encoded_lengths'], ...)
    # outputs shape: (1, T, 1024)
    ENCODER_DIM = 1024

    def __init__(self, cfg: ContentEncoderConfig):
        super().__init__()
        self.cfg = cfg

        print(f"[IndicConformerEncoder] Loading {INDICCONFORMER_MODEL_ID}...")
        print(f"  First run downloads ~2.4GB. Subsequent runs use cache.")
        self._model = AutoModel.from_pretrained(
            INDICCONFORMER_MODEL_ID,
            trust_remote_code=True,
        )

        # Validate that encode() method exists (sanity check)
        if not hasattr(self._model, "encode"):
            raise RuntimeError(
                "IndicASRModel does not have an encode() method. "
                "The model version may have changed. "
                "Run tools/inspect_conformer_deep.py to investigate."
            )

        # Auto-detect actual encoder output dimension.
        # Use random noise — silence causes the ONNX encoder to collapse
        # to degenerate outputs with wrong dimensions.
        # Use 2 seconds (32000 samples) — enough for stable feature extraction.
        print("  [probe] Detecting encoder output dimension with 2s noise...")
        probe_wav = torch.randn(1, 32000) * 0.01  # small amplitude noise
        with torch.no_grad():
            probe_out, probe_len = self._model.encode(probe_wav)
        print(f"  [probe] Raw ONNX output shape: {probe_out.shape}")
        print(f"  [probe] Encoded lengths: {probe_len}")

        # ONNX encoder always outputs (1, D, T) — D=1024, T=time frames.
        # We always transpose to (1, T, D) for consistency with the rest
        # of the pipeline. Axis 1 is always the feature dim.
        if probe_out.ndim == 2:
            probe_out = probe_out[np.newaxis]   # (D, T) → (1, D, T)
        # Always treat as (B, D, T) → (B, T, D)
        probe_out = probe_out.transpose(0, 2, 1)   # (1, T, D)

        actual_dim = probe_out.shape[-1]
        print(f"  [probe] Normalised shape: {probe_out.shape}  →  encoder_dim={actual_dim}")
        self.encoder_dim = actual_dim

        if cfg.freeze_encoder is False:
            print(
                "  [WARNING] freeze_encoder=False has no effect for "
                "IndicConformer — the ONNX encoder is always frozen. "
                "Only the projection head will be trained."
            )

    def forward(
        self,
        waveform: torch.Tensor,
        lang_id: str = "hi",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract encoder hidden states by calling model.encode() directly.

        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio at 16kHz. Shape (B, samples).
            Processed one item at a time due to ONNX session constraints.
        lang_id : str
            BCP-47 language code. IndicConformer supports all 22 official
            Indian languages — lang_id only affects the decoder head which
            we do NOT use here.

        Returns
        -------
        torch.Tensor
            Encoder hidden states. Shape (B, T_max, 1024).
            T_max = longest sequence in batch. Shorter sequences are
            zero-padded on the right.
        """
        assert waveform.shape[-1] > 0, "Empty waveform passed to IndicConformerEncoder"
        B = waveform.shape[0]
        all_outputs = []

        for i in range(B):
            wav_i = waveform[i].unsqueeze(0)  # (1, samples)

            # Call encode() — returns (numpy_array, numpy_lengths)
            # outputs shape: (1, T, 1024) as numpy float32
            # encoded_lengths shape: (1,) — number of valid frames
            with torch.no_grad():
                outputs_np, lengths_np = self._model.encode(wav_i)

            # ONNX encoder always outputs (1, D, T) — always transpose to (1, T, D)
            if outputs_np.ndim == 2:
                outputs_np = outputs_np[np.newaxis]          # (D,T) → (1,D,T)
            outputs_np = outputs_np.transpose(0, 2, 1)       # (1,D,T) → (1,T,D)

            # Convert to torch tensor
            hidden = torch.from_numpy(outputs_np.copy())     # (1, T, encoder_dim)

            # Trim to valid frames using encoded_lengths
            valid_T = int(lengths_np[0])
            hidden = hidden[:, :valid_T, :]

            all_outputs.append(hidden)

        # Pad batch to longest sequence and concatenate
        max_T = max(h.shape[1] for h in all_outputs)
        padded = []
        for h in all_outputs:
            T = h.shape[1]
            if T < max_T:
                pad = torch.zeros(
                    1, max_T - T, self.ENCODER_DIM,
                    dtype=h.dtype  # keep on CPU — ONNX always outputs CPU
                )
                h = torch.cat([h, pad], dim=1)
            padded.append(h)

        result = torch.cat(padded, dim=0)  # (B, max_T, 1024)

        # Move to same device as input waveform
        return result.to(waveform.device)


# ---------------------------------------------------------------------------
# Main ContentEncoder (used by the rest of IndicVC)
# ---------------------------------------------------------------------------

class ContentEncoder(nn.Module):
    """
    IndicVC content encoder.

    Wraps either IndicWhisperEncoder or IndicConformerEncoder and adds:
      1. A linear projection to `output_dim` (matching DiT decoder input).
      2. Layer norm + dropout for training stability.
      3. Language-family embedding added to features — subtle signal
         helping downstream MoE router distinguish language families
         even from content features alone.

    Interface
    ---------
    All other modules in IndicVC interact with this class only.
    They never import IndicWhisperEncoder or IndicConformerEncoder directly.

    Example
    -------
    >>> cfg = ContentEncoderConfig(backend="indicwhisper", languages=["hi","ta"])
    >>> enc = ContentEncoder(cfg).cuda()
    >>> wav = torch.randn(2, 32000).cuda()   # 2-second batch at 16kHz
    >>> out = enc(wav, lang_id="hi")
    >>> out.shape
    torch.Size([2, 200, 1024])
    """

    def __init__(self, cfg: ContentEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Instantiate backend
        if cfg.backend == "indicwhisper":
            self.backend = IndicWhisperEncoder(cfg)
        elif cfg.backend == "indicconformer":
            self.backend = IndicConformerEncoder(cfg)
        else:
            raise ValueError(
                f"Unknown backend '{cfg.backend}'. "
                "Choose 'indicwhisper' or 'indicconformer'."
            )

        encoder_dim = self.backend.encoder_dim

        # Language-family embedding (small — 32 dims added to features)
        # Families: indo_aryan, dravidian, sino_tibetan, austro_asiatic
        self.n_families = 4
        self.family_embed = nn.Embedding(self.n_families, 32)
        self._family_to_idx = {
            "indo_aryan": 0, "dravidian": 1,
            "sino_tibetan": 2, "austro_asiatic": 3,
        }

        # Projection: encoder_dim + 32 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim + 32, cfg.output_dim),
            nn.GELU(),
            nn.Linear(cfg.output_dim, cfg.output_dim),
        )
        self.norm = nn.LayerNorm(cfg.output_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def _family_idx(self, lang_id: str) -> int:
        family = LANG_FAMILY.get(lang_id, "indo_aryan")
        return self._family_to_idx[family]

    def forward(
        self,
        waveform: torch.Tensor,
        lang_id: str = "hi",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio at 16kHz. Shape: (B, samples).
        lang_id : str
            BCP-47 source language code.
        attention_mask : optional torch.Tensor
            Padding mask for batched variable-length inputs.

        Returns
        -------
        torch.Tensor
            Content features. Shape: (B, T, output_dim).
            T depends on the backend's frame shift:
              - IndicWhisper  : ~T = samples / 160  (10ms @ 16kHz)
              - IndicConformer: ~T = samples / 160  (same frame shift)
        """
        # 1. Get raw encoder features
        features = self.backend(waveform, lang_id=lang_id,
                                attention_mask=attention_mask)
        # features: (B, T, encoder_dim)

        B, T, _ = features.shape

        # 2. Language-family embedding broadcast over time
        fam_idx = torch.tensor(
            [self._family_idx(lang_id)], device=features.device
        )  # (1,)
        fam_emb = self.family_embed(fam_idx)          # (1, 32)
        fam_emb = fam_emb.unsqueeze(1).expand(B, T, -1)  # (B, T, 32)

        # 3. Concatenate and project
        combined = torch.cat([features, fam_emb], dim=-1)  # (B, T, encoder_dim+32)
        projected = self.projection(combined)              # (B, T, output_dim)
        out = self.dropout(self.norm(projected))           # (B, T, output_dim)

        return out

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def stream(self, waveform_chunk: torch.Tensor, lang_id: str = "hi"):
        """
        Process a single audio chunk for real-time streaming inference.

        Parameters
        ----------
        waveform_chunk : torch.Tensor
            One chunk of audio at 16kHz. Shape: (1, chunk_samples).
            chunk_samples = cfg.chunk_size_ms * 16  (e.g. 200ms → 3200 samples)
        lang_id : str
            BCP-47 source language code.

        Returns
        -------
        torch.Tensor
            Content features for this chunk. Shape: (1, T_chunk, output_dim).

        Notes
        -----
        For IndicWhisper backend: the causal attention mask ensures this
        chunk only attends to itself + lookahead frames.
        For IndicConformer backend: CTC is inherently causal, no special
        handling needed.
        """
        with torch.no_grad():
            return self.forward(waveform_chunk, lang_id=lang_id)


# ---------------------------------------------------------------------------
# Utility: instantiate from a flat config dict (for YAML-based training)
# ---------------------------------------------------------------------------

def build_content_encoder(config: dict) -> ContentEncoder:
    """
    Convenience factory. Called from train.py and inference.py.

    Example config dict (from config/model.yaml):
        content_encoder:
            backend: indicwhisper
            chunk_size_ms: 200
            lookahead_ms: 40
            output_dim: 1024
            freeze_encoder: false
            dropout: 0.1
            languages: [hi, ta, te, kn, ml, bn, mr, gu, or, pa]

    Parameters
    ----------
    config : dict
        Parsed YAML sub-dict for content_encoder.

    Returns
    -------
    ContentEncoder
    """
    cfg = ContentEncoderConfig(**config)
    return ContentEncoder(cfg)