"""
IndicVC — modules package
=========================
Import all public module interfaces here so the rest of the repo
can use clean imports:

    from modules import ContentEncoder, ContentEncoderConfig
    from modules import build_content_encoder

Do NOT import heavy dependencies at package level — keep this file
lightweight so `import modules` doesn't trigger model downloads.
"""

from .content_encoder import (
    ContentEncoder,
    ContentEncoderConfig,
    build_content_encoder,
    INDIC_LANG_CODES,
    LANG_FAMILY,
)

from .speaker_encoder import (
    SpeakerEncoder,
    SpeakerEncoderConfig,
    build_speaker_encoder,
)

__all__ = [
    "ContentEncoder",
    "ContentEncoderConfig",
    "build_content_encoder",
    "INDIC_LANG_CODES",
    "LANG_FAMILY",
    "SpeakerEncoder",
    "SpeakerEncoderConfig",
    "build_speaker_encoder",
]