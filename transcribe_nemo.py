from __future__ import annotations

import signal
import json
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

from reazonspeech.nemo.asr import audio_from_path, audio_from_numpy, load_model, transcribe
from reazonspeech.nemo.asr.interface import AudioData, Segment, Subword, TranscribeResult

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_SUBWORD_DURATION = 0.3
SILENCE_TOP_DB = 45
SILENCE_FRAME_LENGTH = 2048
SILENCE_HOP_LENGTH = 512
SILENCE_PADDING_SECONDS = 0.15


@dataclass
class AppConfig:
    input_dir: str = "inputs"
    output_dir: str = "outputs"
    output_mode: str = "segment"  # or "subword"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    device_preference: str = "auto"  # auto/cuda/cpu
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


def segments_to_srt(transcription: TranscribeResult) -> str:
    lines: list[str] = []
    for idx, segment in enumerate(transcription.segments, 1):
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
    audio = audio_from_path(str(audio_path))
    intervals = split_audio_on_silence(audio)
    chunks = build_audio_chunks(audio, intervals)
    chunk_results: list[tuple[TranscribeResult, float]] = []

    for chunk_audio, offset in chunks:
        chunk_result = transcribe(model, chunk_audio)
        chunk_results.append((chunk_result, offset))

    return merge_transcriptions(chunk_results), len(chunks)


def split_audio_on_silence(audio: AudioData) -> list[tuple[int, int]]:
    waveform = np.asarray(audio.waveform, dtype=np.float32)
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

    keep = int(SILENCE_PADDING_SECONDS * audio.samplerate)
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
    audio: AudioData, intervals: Sequence[tuple[int, int]]
) -> list[tuple[AudioData, float]]:
    chunks: list[tuple[AudioData, float]] = []
    waveform = np.asarray(audio.waveform, dtype=np.float32)

    for start, end in intervals:
        if end - start <= 0:
            continue
        chunk_waveform = waveform[start:end].copy()
        offset_seconds = start / audio.samplerate
        chunks.append(
            (audio_from_numpy(chunk_waveform, audio.samplerate), offset_seconds)
        )

    if not chunks:
        chunks.append((audio_from_numpy(waveform.copy(), audio.samplerate), 0.0))

    return chunks


def merge_transcriptions(
    chunk_results: Sequence[tuple[TranscribeResult, float]]
) -> TranscribeResult:
    if not chunk_results:
        return TranscribeResult(text="", subwords=[], segments=[])

    segments: list[Segment] = []
    subwords: list[Subword] = []
    texts: list[str] = []

    for chunk_result, offset in chunk_results:
        chunk_text = chunk_result.text.strip()
        if chunk_text:
            texts.append(chunk_text)

        for seg in chunk_result.segments or []:
            segments.append(
                Segment(
                    start_seconds=seg.start_seconds + offset,
                    end_seconds=seg.end_seconds + offset,
                    text=seg.text,
                )
            )

        for sw in chunk_result.subwords or []:
            subwords.append(
                Subword(
                    seconds=sw.seconds + offset,
                    token_id=sw.token_id,
                    token=sw.token,
                )
            )

    combined_text = " ".join(texts).strip()
    return TranscribeResult(text=combined_text, subwords=subwords, segments=segments)


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
    model = load_model(device=device)

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        result, chunk_count = transcribe_file(model, audio_path)
        if config.output_mode == "subword":
            srt_content = subwords_to_srt(result.subwords)
        else:
            srt_content = segments_to_srt(result)
        output_path = build_output_path(audio_path, output_dir, config.timestamp_format)
        output_path.write_text(srt_content, encoding="utf-8")
        print(f" -> {output_path} [{chunk_count} chunk(s)]")


if __name__ == '__main__':
    main()
