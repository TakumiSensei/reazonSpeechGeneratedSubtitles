from __future__ import annotations

import librosa
import numpy as np

# app_utils handles dll/signal setup on import
from app_utils import (
    AppConfig,
    Segment,
    load_app_config,
    ensure_directories,
    select_device,
    build_output_path,
    segments_to_srt,
    subwords_to_srt,
    get_base_path
)

from reazonspeech.k2.asr import load_model, transcribe, TranscribeConfig, audio_from_numpy
from reazonspeech.k2.asr.interface import TranscribeResult, Subword

# Constants
SILENCE_TOP_DB = 45
SILENCE_FRAME_LENGTH = 2048
SILENCE_HOP_LENGTH = 512
SILENCE_PADDING_SECONDS = 0.15


def iter_audio_files(input_dir, extensions):
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def transcribe_file(model, audio_path) -> tuple[TranscribeResult, int]:
    # Load audio for splitting using librosa
    y, sr = librosa.load(str(audio_path), sr=16000)
    
    intervals = split_audio_on_silence(y, sr)
    chunks = build_audio_chunks(y, sr, intervals)
    chunk_results = []

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
    waveform: np.ndarray, samplerate: int, intervals
) -> list[tuple[np.ndarray, float, float]]:
    chunks = []
    
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
    chunk_results
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


def run_batch(config: AppConfig, base_dir=None):
    if base_dir is None:
        base_dir = get_base_path()
        
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
            srt_content = subwords_to_srt(result.subwords, remove_period=config.remove_period)
        else:
            extend_seconds = (
                config.extend_segment_end_seconds if config.extend_segment_end else 0.0
            )
            # K2 uses our injected segments
            # segments_to_srt expects a list of Segments, not TranscribeResult directly unless we extract it
            # But the utility expects `segments`. 
            # `merge_transcriptions` attaches `.segments` to `result`.
            srt_content = segments_to_srt(
                result.segments, 
                extend_end_seconds=extend_seconds,
                remove_period=config.remove_period
            )
            
        output_path = build_output_path(audio_path, output_dir, config.timestamp_format)
        output_path.write_text(srt_content, encoding="utf-8")
        print(f" -> {output_path} [{chunk_count} chunk(s)]")


def main():
    config = load_app_config()
    run_batch(config)


if __name__ == '__main__':
    main()
