import json
import os
import signal
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

# Try to ensure DLL paths for Windows
def _ensure_dll_paths():
    if os.name != "nt":
        return

    paths = []
    if hasattr(sys, "frozen"):
        # Running as compiled EXE
        base_path = Path(sys._MEIPASS)
        paths = [
            base_path / "torch" / "lib",
            base_path / "nvidia" / "cublas" / "bin",
            base_path / "nvidia" / "cudnn" / "bin",
            base_path / "nvidia" / "cuda_runtime" / "bin",
            base_path / "nvidia" / "curand" / "bin",
            base_path / "nvidia" / "cufft" / "bin",
        ]
    else:
        # Running as script
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

# Run environment setup immediately on import
_ensure_dll_paths()
_ensure_signal_attrs()
_ensure_ml_dtypes()


DEFAULT_SUBWORD_DURATION = 0.3

# Shared dataclasses
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
    remove_period: bool = False
    model_type: str = "k2v2" # k2v2/nemo
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
            
        self.model_type = self.model_type.lower()
        if self.model_type not in {"k2v2", "nemo"}:
            self.model_type = "k2v2"

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

def get_base_path() -> Path:
    """Return the base path for config and IO directories."""
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE
        # sys.executable is the path to the exe itself
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent

def load_app_config(base_path: Path = None) -> AppConfig:
    if base_path is None:
        base_path = get_base_path()
        
    config_path = base_path / "config.json"
    config = AppConfig()
    
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # We might want to warn here but for now just use defaults or raise
            raise RuntimeError(f"Failed to parse config: {config_path}") from exc
        allowed = {f.name for f in fields(config)}
        for key, value in data.items():
            if key in allowed:
                setattr(config, key, value)
    config.normalize()
    return config

def ensure_directories(base_dir: Path, config: AppConfig) -> tuple[Path, Path]:
    input_dir = (base_dir / config.input_dir).resolve()
    output_dir = (base_dir / config.output_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir

def format_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def normalize_text(text: str, remove_period: bool = False) -> str:
    # Basic normalization
    text = text.replace("\u2581", " ").strip()
    if remove_period:
        # Remove trailing periods (Japanese "。" and English ".")
        # You might want to be more sophisticated, but simple rstrip is usually requested
        if text.endswith("。"):
            text = text[:-1]
        elif text.endswith("."):
            text = text[:-1]
    return text

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

def segments_to_srt(segments: Sequence[Segment], *, extend_end_seconds: float = 0.0, remove_period: bool = False) -> str:
    lines: list[str] = []
    # Note: caller is responsible for extracting segments from TranscribeResult
    segments = _extend_segment_end_times(segments, extend_end_seconds)

    for idx, segment in enumerate(segments, 1):
        text = normalize_text(segment.text, remove_period) or "(no speech)"
        lines.append(str(idx))
        lines.append(
            f"{format_timestamp(segment.start_seconds)} --> {format_timestamp(segment.end_seconds)}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def subwords_to_srt(subwords: Sequence, remove_period: bool = False) -> str:
    if not subwords:
        return ""

    def end_time(idx: int, start: float) -> float:
        if idx + 1 < len(subwords):
            # Assumes subword object has .seconds attribute
            nxt = subwords[idx + 1].seconds
            if nxt > start:
                return nxt
        return start + DEFAULT_SUBWORD_DURATION

    lines: list[str] = []
    for idx, sw in enumerate(subwords, 1):
        start = sw.seconds
        end = end_time(idx, start)
        # Assumes subword object has .token attribute
        text = normalize_text(sw.token, remove_period) or "(no speech)"
        lines.append(str(idx))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"

def build_output_path(audio_path: Path, output_dir: Path, timestamp_format: str) -> Path:
    timestamp = datetime.now().strftime(timestamp_format)
    return output_dir / f"{audio_path.stem}_{timestamp}.srt"

def select_device(config: AppConfig):
    import torch # Import here to avoid early import if just checking config
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
