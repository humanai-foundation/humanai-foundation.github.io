import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_and_normalize(audio_path, sample_rate=16000):
    """Load audio and normalize peak amplitude into [-1, 1]."""
    waveform, loaded_sample_rate = librosa.load(str(audio_path), sr=sample_rate)
    normalized_waveform = librosa.util.normalize(waveform)
    return normalized_waveform, loaded_sample_rate


def segment_audio(waveform, sample_rate, segment_seconds=2.0):
    """Split waveform into fixed-size segments; return one segment if disabled."""
    if segment_seconds is None or segment_seconds <= 0:
        return [(0.0, len(waveform) / float(sample_rate), waveform)]

    segment_size = int(segment_seconds * sample_rate)
    if segment_size <= 0 or len(waveform) <= segment_size:
        return [(0.0, len(waveform) / float(sample_rate), waveform)]

    segments = []
    for start in range(0, len(waveform), segment_size):
        end = min(start + segment_size, len(waveform))
        chunk = waveform[start:end]
        if len(chunk) == 0:
            continue
        start_sec = start / float(sample_rate)
        end_sec = end / float(sample_rate)
        segments.append((start_sec, end_sec, chunk))
    return segments


def extract_features(segment_waveform, sample_rate=16000, n_mfcc=13):
    """Extract MFCC + pitch + spectral centroid + energy + duration features."""
    mfcc_matrix = librosa.feature.mfcc(y=segment_waveform, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = mfcc_matrix.mean(axis=1)

    f0 = librosa.yin(segment_waveform, fmin=50, fmax=400, sr=sample_rate)
    voiced_f0 = f0[np.isfinite(f0)]
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size > 0 else 0.0

    spectral_centroid = librosa.feature.spectral_centroid(y=segment_waveform, sr=sample_rate)
    spectral_centroid_mean = float(np.mean(spectral_centroid))

    rms_energy = librosa.feature.rms(y=segment_waveform)
    energy_mean = float(np.mean(rms_energy))

    duration_seconds = len(segment_waveform) / float(sample_rate)

    feature_row = {
        "pitch_mean_hz": pitch_mean,
        "spectral_centroid_mean_hz": spectral_centroid_mean,
        "energy_rms_mean": energy_mean,
        "duration_seconds": float(duration_seconds),
    }
    for index, value in enumerate(mfcc_mean, start=1):
        feature_row[f"mfcc_{index}"] = float(value)

    return feature_row


def build_feature_dataset(
    input_dir,
    output_csv,
    normalized_dir=None,
    sample_rate=16000,
    n_mfcc=13,
    segment_seconds=2.0,
    max_files=None,
):
    """Process a folder of audio files and write a structured ML-ready CSV."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    normalized_dir = Path(normalized_dir) if normalized_dir else None

    if normalized_dir:
        normalized_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.glob("*.wav"))
    if max_files is not None:
        audio_files = audio_files[: max(0, int(max_files))]

    dataset_rows = []
    for audio_file in audio_files:
        normalized_waveform, loaded_sample_rate = load_and_normalize(audio_file, sample_rate=sample_rate)

        if normalized_dir:
            normalized_path = normalized_dir / audio_file.name
            sf.write(str(normalized_path), normalized_waveform, loaded_sample_rate)

        segments = segment_audio(
            waveform=normalized_waveform,
            sample_rate=loaded_sample_rate,
            segment_seconds=segment_seconds,
        )
        for segment_index, (start_sec, end_sec, segment_waveform) in enumerate(segments):
            row = {
                "filename": audio_file.name,
                "segment_index": segment_index,
                "segment_start_seconds": float(start_sec),
                "segment_end_seconds": float(end_sec),
            }
            row.update(extract_features(segment_waveform, sample_rate=loaded_sample_rate, n_mfcc=n_mfcc))
            dataset_rows.append(row)

    if not dataset_rows:
        raise ValueError(f"No .wav files found in {input_dir}")

    fieldnames = list(dataset_rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"Wrote {len(dataset_rows)} rows to {output_csv}")
    return dataset_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: audio processing pipeline")
    parser.add_argument("--input-dir", default="../examples", help="Directory containing .wav files")
    parser.add_argument(
        "--output-csv",
        default="../examples/task1_features_dataset.csv",
        help="Path to save extracted feature dataset CSV",
    )
    parser.add_argument(
        "--normalized-dir",
        default="../examples/normalized_audio",
        help="Directory to save normalized audio files",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick demo runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = (script_dir / args.input_dir).resolve()
    output_csv = (script_dir / args.output_csv).resolve()
    normalized_dir = (script_dir / args.normalized_dir).resolve() if args.normalized_dir else None

    build_feature_dataset(
        input_dir=input_dir,
        output_csv=output_csv,
        normalized_dir=normalized_dir,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        segment_seconds=args.segment_seconds,
        max_files=args.max_files,
    )
