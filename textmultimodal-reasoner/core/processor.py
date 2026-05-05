"""Raw multimodal input processing utilities for Gemma-style pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover
    AudioSegment = None

# Gemma 4 vision encoder input size (square resolution).
GEMMA4_IMAGE_SIZE = (896, 896)
# Common multimodal audio projector defaults.
GEMMA4_AUDIO_SAMPLE_RATE = 16_000
GEMMA4_AUDIO_N_MELS = 128
GEMMA4_AUDIO_HOP_LENGTH = 160
GEMMA4_AUDIO_FFT = 400


@dataclass(slots=True)
class ProcessedBundle:
    """Canonical data packet expected by llama-cpp multimodal projectors."""

    prompt: str
    pixel_values: np.ndarray | None
    audio_features: np.ndarray | None


def _to_rgb(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def prepare_image_tensor(
    image_path: str | Path,
    target_size: tuple[int, int] = GEMMA4_IMAGE_SIZE,
) -> np.ndarray:
    """Resize an image exactly to Gemma 4 expected resolution and return pixel_values.

    Output shape: (1, 3, H, W), float32 normalized to [0, 1].
    """
    image = _to_rgb(image_path)
    image = image.resize(target_size, resample=Image.Resampling.BICUBIC)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    # HWC -> CHW and add batch dimension.
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_audio_waveform(audio_path: str | Path, sample_rate: int) -> np.ndarray:
    if librosa is not None:
        waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        return waveform.astype(np.float32)

    if AudioSegment is None:
        raise ImportError(
            "Install either `librosa` or `pydub` to process audio inputs."
        )

    segment = AudioSegment.from_file(str(audio_path))
    segment = segment.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
    raw = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if raw.size == 0:
        return raw
    return raw / np.iinfo(np.int16).max


def prepare_audio_features(
    audio_path: str | Path,
    sample_rate: int = GEMMA4_AUDIO_SAMPLE_RATE,
    n_mels: int = GEMMA4_AUDIO_N_MELS,
) -> np.ndarray:
    """Convert audio file to projector-friendly audio_features tensor.

    Output shape: (1, n_mels, frames), float32 log-mel features.
    """
    if librosa is None:
        raise ImportError(
            "`librosa` is required for mel-spectrogram feature extraction."
        )

    waveform = _load_audio_waveform(audio_path, sample_rate=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=GEMMA4_AUDIO_FFT,
        hop_length=GEMMA4_AUDIO_HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    features = log_mel.astype(np.float32)[None, ...]
    return np.ascontiguousarray(features)


def build_media_first_prompt(items: list[dict[str, Any]]) -> str:
    """Build prompt with media tokens prepended before all text segments."""
    media_tokens: list[str] = []
    text_blocks: list[str] = []

    for item in items:
        item_type = item.get("type")
        data = item.get("data", "")
        if item_type == "image":
            media_tokens.append(f"<image>{Path(str(data)).name}</image>")
        elif item_type == "audio":
            media_tokens.append(f"<audio>{Path(str(data)).name}</audio>")
        elif item_type == "text":
            text_blocks.append(str(data))

    return "\n\n".join(part for part in ["\n".join(media_tokens), "\n".join(text_blocks)] if part).strip()


def process_interleaved_inputs(items: list[dict[str, Any]]) -> ProcessedBundle:
    """Process raw interleaved items into prompt + multimodal tensors.

    Returns tensors under names (`pixel_values`, `audio_features`) that llama-cpp
    multimodal projectors expect.
    """
    pixel_values: np.ndarray | None = None
    audio_features: np.ndarray | None = None

    for item in items:
        item_type = item.get("type")
        data = item.get("data")

        if item_type == "image" and data:
            pixel_values = prepare_image_tensor(data)
        elif item_type == "audio" and data:
            audio_features = prepare_audio_features(data)

    return ProcessedBundle(
        prompt=build_media_first_prompt(items),
        pixel_values=pixel_values,
        audio_features=audio_features,
    )
