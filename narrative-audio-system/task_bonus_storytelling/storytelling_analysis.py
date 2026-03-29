import argparse
import csv
import re
import sys
from pathlib import Path

import librosa
import numpy as np

# Reuse Task 3 Whisper helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task3_transcription.whisper_transcriber import load_model, transcribe_file


def _min_max_normalize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def add_storytelling_scores(rows):
    """Add a 0-100 storytelling score using weighted expressive features."""
    if not rows:
        return rows

    pitch_var = _min_max_normalize([float(r["pitch_std_hz"]) for r in rows])
    energy_dyn = _min_max_normalize([float(r["energy_dynamic_range"]) for r in rows])
    pause_dyn = _min_max_normalize([float(r["pause_ratio"]) for r in rows])
    sentence_len = _min_max_normalize([float(r["avg_sentence_words"]) for r in rows])

    for idx, row in enumerate(rows):
        # Heuristic: storytelling tends to have richer pitch/energy dynamics,
        # purposeful pauses, and somewhat longer phrasing.
        score_0_1 = (
            0.35 * pitch_var[idx]
            + 0.30 * energy_dyn[idx]
            + 0.20 * pause_dyn[idx]
            + 0.15 * sentence_len[idx]
        )
        row["storytelling_score"] = round(float(score_0_1 * 100.0), 2)

    return rows


def extract_storytelling_features(audio_path, sample_rate=16000):
    """Compute pacing/pauses, pitch variation, and energy dynamics for one file."""
    waveform, sr = librosa.load(str(audio_path), sr=sample_rate)

    # Duration and pacing proxy.
    duration_seconds = len(waveform) / float(sr)
    tempo_bpm = float(librosa.feature.tempo(y=waveform, sr=sr)[0])

    # Pause dynamics from low-energy frames.
    rms = librosa.feature.rms(y=waveform, frame_length=1024, hop_length=256)[0]
    silence_threshold = max(0.01, float(np.percentile(rms, 20)))
    silence_mask = rms < silence_threshold
    silence_ratio = float(np.mean(silence_mask)) if len(silence_mask) else 0.0
    pause_events = int(np.sum((~silence_mask[:-1]) & (silence_mask[1:]))) if len(silence_mask) > 1 else 0

    # Pitch variation from voiced F0 values.
    f0 = librosa.yin(waveform, fmin=50, fmax=400, sr=sr)
    voiced_f0 = f0[np.isfinite(f0)]
    pitch_mean_hz = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std_hz = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

    # Energy dynamics.
    energy_mean = float(np.mean(rms)) if len(rms) else 0.0
    energy_std = float(np.std(rms)) if len(rms) else 0.0
    energy_dynamic_range = float(np.percentile(rms, 90) - np.percentile(rms, 10)) if len(rms) else 0.0

    return {
        "duration_seconds": float(duration_seconds),
        "tempo_bpm": tempo_bpm,
        "pause_ratio": silence_ratio,
        "pause_events": pause_events,
        "pitch_mean_hz": pitch_mean_hz,
        "pitch_std_hz": pitch_std_hz,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_dynamic_range": energy_dynamic_range,
    }


def sentence_length_features(transcript_text):
    """Approximate sentence-length metrics from transcript punctuation."""
    normalized = transcript_text.strip()
    if not normalized:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_words": 0.0,
            "max_sentence_words": 0,
        }

    words = re.findall(r"\b[\w']+\b", normalized)
    sentence_chunks = [s.strip() for s in re.split(r"[.!?]+", normalized) if s.strip()]
    sentence_lengths = [len(re.findall(r"\b[\w']+\b", s)) for s in sentence_chunks]

    return {
        "word_count": len(words),
        "sentence_count": len(sentence_chunks),
        "avg_sentence_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "max_sentence_words": int(max(sentence_lengths)) if sentence_lengths else 0,
    }


def analyze_storytelling(input_dir, output_csv, max_files=12, model_size="tiny"):
    """Analyze a subset of recordings and save storytelling-oriented features to CSV."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.glob("*.wav"))[:max_files]
    if not audio_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    print(f"Bonus task: loading Whisper '{model_size}' for transcript-based sentence features...")
    whisper_model = load_model(model_size)

    rows = []
    for audio_file in audio_files:
        transcript = transcribe_file(whisper_model, audio_file)
        audio_feats = extract_storytelling_features(audio_file)
        text_feats = sentence_length_features(transcript)

        row = {
            "filename": audio_file.name,
            "transcript": transcript,
            **audio_feats,
            **text_feats,
        }
        rows.append(row)
        print(f"Bonus task: analyzed {audio_file.name}")

    rows = add_storytelling_scores(rows)

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Bonus task: wrote {len(rows)} rows to {output_csv}")
    return rows


def discuss_storytelling_signals(rows):
    """Print a short discussion of which features can signal storytelling narration."""
    pitch_std = np.array([float(r["pitch_std_hz"]) for r in rows], dtype=float)
    pause_ratio = np.array([float(r["pause_ratio"]) for r in rows], dtype=float)
    energy_range = np.array([float(r["energy_dynamic_range"]) for r in rows], dtype=float)
    sentence_len = np.array([float(r["avg_sentence_words"]) for r in rows], dtype=float)

    print("\nBonus discussion: storytelling vs conversational speech")
    print("- Pacing and pauses: higher pause ratio or more pause events can indicate narrative phrasing and dramatic timing.")
    print("- Pitch variation: larger pitch_std_hz usually suggests expressive storytelling rather than flat conversational delivery.")
    print("- Energy dynamics: larger energy_dynamic_range often reflects emphasis and emotional arcs in stories.")
    print("- Sentence length: longer average sentence length can indicate narration; shorter fragments can indicate dialogue exchanges.")
    print("\nObserved on this subset:")
    print(f"- Mean pitch variation (std Hz): {np.mean(pitch_std):.2f}")
    print(f"- Mean pause ratio: {np.mean(pause_ratio):.3f}")
    print(f"- Mean energy dynamic range: {np.mean(energy_range):.4f}")
    print(f"- Mean average sentence length: {np.mean(sentence_len):.2f} words")

    ranked = sorted(rows, key=lambda r: float(r.get("storytelling_score", 0.0)), reverse=True)
    print("\nTop storytelling-like clips (heuristic score):")
    for row in ranked[:3]:
        print(f"- {row['filename']}: score={float(row['storytelling_score']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bonus: Storytelling audio analysis")
    parser.add_argument("--input-dir", default="../examples", help="Directory containing .wav files")
    parser.add_argument(
        "--output-csv",
        default="../examples/storytelling_analysis.csv",
        help="Path to output storytelling analysis CSV",
    )
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    rows = analyze_storytelling(
        input_dir=(script_dir / args.input_dir).resolve(),
        output_csv=(script_dir / args.output_csv).resolve(),
        max_files=args.max_files,
        model_size=args.model_size,
    )
    discuss_storytelling_signals(rows)
