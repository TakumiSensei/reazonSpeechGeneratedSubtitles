from __future__ import annotations

import librosa
import numpy as np

# app_utils handles dll/signal setup on import
from app_utils import (
    AppConfig,
    Segment,
    Subword,
    load_app_config,
    ensure_directories,
    select_device,
    build_output_path,
    segments_to_srt,
    subwords_to_srt,
    get_base_path
)

from reazonspeech.nemo.asr import audio_from_path, audio_from_numpy, load_model, transcribe
from reazonspeech.nemo.asr.interface import AudioData, TranscribeResult

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
    audio = audio_from_path(str(audio_path))
    intervals = split_audio_on_silence(audio)
    chunks = build_audio_chunks(audio, intervals)
    chunk_results = []

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
    audio: AudioData, intervals
) -> list[tuple[AudioData, float]]:
    chunks = []
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
    chunk_results
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
            # Note: app_utils Subword vs reazon subword might essentially be same but distinct classes
            # app_utils doesn't export Subword class, it uses whatever is passed to subwords_to_srt
            # But here we are constructing ReazonSpeech TranscribeResult which expects Reazon Subword?
            # Or our own?
            # TranscribeResult in existing code came from reazonspeech.nemo.asr.interface
            # So we should use that.
            # But app_utils.subwords_to_srt expects object with .seconds and .token
            subwords.append(
                Subword(
                    seconds=sw.seconds + offset,
                    token_id=sw.token_id,
                    token=sw.token,
                )
            )

    combined_text = " ".join(texts).strip()
    return TranscribeResult(text=combined_text, subwords=subwords, segments=segments)


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
    print("Loading ReazonSpeech NeMo model...")
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
