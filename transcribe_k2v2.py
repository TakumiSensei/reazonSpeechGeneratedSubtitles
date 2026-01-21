from __future__ import annotations

import signal
import json
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import torch

def _ensure_signal_attrs():
    """Define POSIX-only signal names missing on Windows."""
    fallback = getattr(signal, "SIGTERM", None)
    if fallback is None:
        return

    for name in ("SIGKILL",):
        if not hasattr(signal, name):
            setattr(signal, name, fallback)


def _ensure_dll_paths():
    """Add nvidia/torch DLL paths to search path on Windows."""
    if os.name != "nt":
        return
        
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    paths = [
        venv_path / "Lib" / "site-packages" / "torch" / "lib",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "curand" / "bin",
        venv_path / "Lib" / "site-packages" / "nvidia" / "cufft" / "bin",
    ]

    for p in paths:
        if p.exists():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass

_ensure_dll_paths()


def _ensure_ml_dtypes():
    """Define Windows-only fallbacks for missing ml_dtypes types."""
    try:
        import ml_dtypes
    except ImportError:
        return

    fallbacks = {
        "float4_e2m1fn": np.float16,
        "float8_e8m0fnu": np.float16,
        "uint4": np.uint8,
        "int4": np.int8,
    }
    for name, alias in fallbacks.items():
        if not hasattr(ml_dtypes, name):
            setattr(ml_dtypes, name, alias)


_ensure_signal_attrs()
_ensure_ml_dtypes()

from reazonspeech.k2.asr import load_model, transcribe, TranscribeConfig, audio_from_numpy
# Note: K2 interface might not have AudioData exposed directly in top-level,
# usually it uses internal audio loading or just raw paths/arrays.
# For silence splitting we need librosa loading anyway.
from reazonspeech.k2.asr.interface import TranscribeResult, Subword

# Helper to load audio for splitting (reusing valid parts from nemo script or writing new one)
# We will use librosa directly for splitting logic to keep it independent of model specific loaders if possible,
# but we need to pass audio to model.
# K2 transcribe accepts: (model, audio, config=None)
# audio can be path or numpy array.

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_SUBWORD_DURATION = 0.3
SILENCE_TOP_DB = 45
SILENCE_FRAME_LENGTH = 2048
SILENCE_HOP_LENGTH = 512
SILENCE_PADDING_SECONDS = 0.15


@dataclass
class Segment:
    """Mock Segment class to unify interface with NeMo based logic."""
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class AppConfig:
    input_dir: str = "inputs"
    output_dir: str = "outputs"
    output_mode: str = "segment"  # or "subword"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    device_preference: str = "auto"  # auto/cuda/cpu
    extend_segment_end: bool = False
    extend_segment_end_seconds: float = 0.5
    supported_extensions: list[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    )

    def normalize(self) -> None:
        self.output_mode = self.output_mode.lower()
        if self.output_mode not in {"segment", "subword"}:
            self.output_mode = "segment"

        self.device_preference = self.device_preference.lower()
        if self.device_preference not in {"auto", "cuda", "cpu"}:
            self.device_preference = "auto"

        try:
            self.extend_segment_end_seconds = max(
                float(self.extend_segment_end_seconds), 0.0
            )
        except (TypeError, ValueError):
            self.extend_segment_end_seconds = 0.5

        self.supported_extensions = sorted(
            {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
             for ext in self.supported_extensions}
        )


def load_app_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config = AppConfig()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse config: {path}") from exc
        allowed = {f.name for f in fields(config)}
        for key, value in data.items():
            if key in allowed:
                setattr(config, key, value)
    config.normalize()
    return config


def format_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def normalize_text(text: str) -> str:
    return text.replace("\u2581", " ").strip()


def _extend_segment_end_times(
    segments: Sequence[Segment], pad_seconds: float
) -> list[Segment]:
    if not segments or pad_seconds <= 0:
        return list(segments)

    adjusted: list[Segment] = []
    last_idx = len(segments) - 1

    for idx, segment in enumerate(segments):
        candidate_end = segment.end_seconds + pad_seconds
        if idx < last_idx:
            next_start = segments[idx + 1].start_seconds
            end_seconds = candidate_end if candidate_end < next_start else segment.end_seconds
        else:
            end_seconds = candidate_end

        adjusted.append(
            Segment(
                start_seconds=segment.start_seconds,
                end_seconds=end_seconds,
                text=segment.text,
            )
        )

    return adjusted


def segments_to_srt(transcription: TranscribeResult, *, extend_end_seconds: float = 0.0) -> str:
    lines: list[str] = []
    # K2 TranscribeResult doesn't have segments, we attach our own constructed segments
    # to the result object or pass them directly.
    # Here we assume merge_transcriptions attaches 'segments' to the result object
    # dynamically or we change the signature to accept segments.
    # For now, let's assume transcription has attribute 'segments' injected by us.
    segments: Sequence[Segment] = getattr(transcription, 'segments', []) or []
    segments = _extend_segment_end_times(segments, extend_end_seconds)

    for idx, segment in enumerate(segments, 1):
        text = normalize_text(segment.text) or "(no speech)"
        lines.append(str(idx))
        lines.append(
            f"{format_timestamp(segment.start_seconds)} --> {format_timestamp(segment.end_seconds)}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def subwords_to_srt(subwords: Sequence[Subword]) -> str:
    if not subwords:
        return ""

    def end_time(idx: int, start: float) -> float:
        if idx + 1 < len(subwords):
            nxt = subwords[idx + 1].seconds
            if nxt > start:
                return nxt
        return start + DEFAULT_SUBWORD_DURATION

    lines: list[str] = []
    for idx, sw in enumerate(subwords, 1):
        start = sw.seconds
        end = end_time(idx, start)
        text = normalize_text(sw.token) or "(no speech)"
        lines.append(str(idx))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def ensure_directories(base_dir: Path, config: AppConfig) -> tuple[Path, Path]:
    input_dir = (base_dir / config.input_dir).resolve()
    output_dir = (base_dir / config.output_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def iter_audio_files(input_dir: Path, extensions: Sequence[str]) -> Iterable[Path]:
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def transcribe_file(model, audio_path: Path) -> tuple[TranscribeResult, int]:
    # Load audio for splitting using librosa
    y, sr = librosa.load(str(audio_path), sr=16000)
    
    intervals = split_audio_on_silence(y, sr)
    chunks = build_audio_chunks(y, sr, intervals)
    chunk_results: list[tuple[TranscribeResult, float, float]] = []

    # ReazonSpeech K2 transcribe config usually handles verbosity
    config = TranscribeConfig(verbose=False)

    for chunk_audio, offset, duration in chunks:
        # chunk_audio is numpy array float32
        # Wrap numpy array into AudioData compatible object
        audio_data = audio_from_numpy(chunk_audio, 16000)
        chunk_result = transcribe(model, audio_data, config=config)
        chunk_results.append((chunk_result, offset, duration))

    return merge_transcriptions(chunk_results), len(chunks)


def split_audio_on_silence(waveform: np.ndarray, samplerate: int) -> list[tuple[int, int]]:
    if waveform.size == 0:
        return [(0, 0)]

    intervals = librosa.effects.split(
        waveform,
        top_db=SILENCE_TOP_DB,
        frame_length=SILENCE_FRAME_LENGTH,
        hop_length=SILENCE_HOP_LENGTH,
    )

    if intervals.size == 0:
        return [(0, waveform.shape[0])]

    keep = int(SILENCE_PADDING_SECONDS * samplerate)
    merged: list[tuple[int, int]] = []

    for start, end in intervals.tolist():
        start = max(0, int(start) - keep)
        end = min(waveform.shape[0], int(end) + keep)
        if end <= start:
            continue

        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged or [(0, waveform.shape[0])]


def build_audio_chunks(
    waveform: np.ndarray, samplerate: int, intervals: Sequence[tuple[int, int]]
) -> list[tuple[np.ndarray, float, float]]:
    chunks: list[tuple[np.ndarray, float, float]] = []
    
    for start, end in intervals:
        if end - start <= 0:
            continue
        chunk_waveform = waveform[start:end].copy()
        offset_seconds = start / samplerate
        duration_seconds = (end - start) / samplerate
        chunks.append(
            (chunk_waveform, offset_seconds, duration_seconds)
        )

    if not chunks:
        chunks.append((waveform.copy(), 0.0, len(waveform)/samplerate))

    return chunks


def merge_transcriptions(
    chunk_results: Sequence[tuple[TranscribeResult, float, float]]
) -> TranscribeResult:
    if not chunk_results:
        res = TranscribeResult(text="", subwords=[])
        res.segments = []
        return res

    segments: list[Segment] = []
    subwords: list[Subword] = []
    texts: list[str] = []

    for chunk_result, offset, duration in chunk_results:
        chunk_text = chunk_result.text.strip()
        if chunk_text:
            texts.append(chunk_text)
            
            # Since K2 doesn't give segments, we treat the whole chunk as one segment
            segments.append(
                Segment(
                    start_seconds=offset,
                    end_seconds=offset + duration,
                    text=chunk_text,
                )
            )

        for sw in chunk_result.subwords or []:
            subwords.append(
                Subword(
                    seconds=sw.seconds + offset,
                    token=sw.token,
                )
            )

    combined_text = " ".join(texts).strip()
    result = TranscribeResult(text=combined_text, subwords=subwords)
    # Inject segments
    result.segments = segments
    return result


def build_output_path(audio_path: Path, output_dir: Path, timestamp_format: str) -> Path:
    timestamp = datetime.now().strftime(timestamp_format)
    return output_dir / f"{audio_path.stem}_{timestamp}.srt"


def select_device(config: AppConfig) -> tuple[str, str]:
    prefer = config.device_preference
    if prefer == "cpu":
        return "cpu", "Forced CPU execution."
    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda", "Using CUDA as requested."
        return "cpu", "Requested CUDA but unavailable, falling back to CPU."

    # auto
    if torch.cuda.is_available():
        return "cuda", "CUDA detected, using GPU."
    return "cpu", "CUDA unavailable, falling back to CPU."


def main():
    base_dir = Path(__file__).resolve().parent
    config = load_app_config()
    input_dir, output_dir = ensure_directories(base_dir, config)

    audio_files = list(iter_audio_files(input_dir, config.supported_extensions))
    if not audio_files:
        print(f"No audio files found in: {input_dir}")
        return

    device, message = select_device(config)
    print(message)
    print("Loading ReazonSpeech k2-v2 model...")
    model = load_model(device=device)

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        result, chunk_count = transcribe_file(model, audio_path)
        if config.output_mode == "subword":
            srt_content = subwords_to_srt(result.subwords)
        else:
            extend_seconds = (
                config.extend_segment_end_seconds if config.extend_segment_end else 0.0
            )
            srt_content = segments_to_srt(
                result, extend_end_seconds=extend_seconds
            )
        output_path = build_output_path(audio_path, output_dir, config.timestamp_format)
        output_path.write_text(srt_content, encoding="utf-8")
        print(f" -> {output_path} [{chunk_count} chunk(s)]")


if __name__ == '__main__':
    main()
