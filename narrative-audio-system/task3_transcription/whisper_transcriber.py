"""Task 3: batch Whisper transcription with simple WER evaluation."""

import argparse
import shutil
from pathlib import Path

import whisper


RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def _ravdess_reference(filename: str) -> str | None:
    """Return ground-truth text for a RAVDESS filename, or None if unknown."""
    parts = Path(filename).stem.split("-")
    if len(parts) >= 5:
        return RAVDESS_STATEMENTS.get(parts[4])
    return None


def _edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    """Standard dynamic-programming edit distance."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], (
                prev if ref_tokens[i - 1] == hyp_tokens[j - 1]
                else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def load_model(model_size: str = "tiny"):
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found — Whisper may fail to decode audio.")
    return whisper.load_model(model_size)


def transcribe_file(model, audio_path: str) -> str:
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def transcribe_directory(
    input_dir: str,
    output_txt: str,
    model_size: str = "tiny",
    max_files: int | None = None,
) -> dict[str, str]:
    """Transcribe `.wav` files and write `filename<TAB>transcript` lines."""
    input_path = Path(input_dir)
    audio_files = sorted(input_path.glob("*.wav"))
    if max_files:
        audio_files = audio_files[:max_files]

    if not audio_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    print(f"\nLoading Whisper '{model_size}' model ...")
    model = load_model(model_size)

    transcripts: dict[str, str] = {}
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_fp:
        for audio_file in audio_files:
            text = transcribe_file(model, audio_file)
            transcripts[audio_file.name] = text
            out_fp.write(f"{audio_file.name}\t{text}\n")
            print(f"  {audio_file.name}: {text}")

    print(f"\nTranscripts saved to {output_path}  ({len(transcripts)} files)")
    return transcripts


def measure_accuracy(
    transcripts: dict[str, str],
    max_samples: int = 20,
) -> None:
    """Compute WER on up to `max_samples` files with known references."""
    wer_scores: list[float] = []
    evaluated: list[tuple[str, str, str, float]] = []

    for filename, hypothesis in list(transcripts.items())[:max_samples]:
        reference = _ravdess_reference(filename)
        if reference is None:
            continue
        wer = word_error_rate(reference, hypothesis)
        wer_scores.append(wer)
        evaluated.append((filename, reference, hypothesis, wer))

    if not wer_scores:
        print("\nNo ground-truth references available for accuracy measurement.")
        return

    avg_wer = sum(wer_scores) / len(wer_scores)

    print(f"\n--- Transcription Accuracy ({len(wer_scores)} samples) ---")
    for filename, ref, hyp, wer in evaluated:
        print(f"  File : {filename}")
        print(f"  Ref  : {ref}")
        print(f"  Hyp  : {hyp}")
        print(f"  WER  : {wer:.2%}")
        print()
    print(f"  Average WER : {avg_wer:.2%}")

    _discuss_quality(avg_wer, len(wer_scores))


def _discuss_quality(avg_wer: float, n_samples: int) -> None:
    print("\n--- Discussion of Transcription Quality ---")
    print(
        f"Whisper (tiny) achieved an average WER of {avg_wer:.1%} on {n_samples} RAVDESS samples."
    )
    if avg_wer <= 0.05:
        print("This is near-perfect transcription — the sentences are short, clear, and in English.")
    elif avg_wer <= 0.20:
        print("Good accuracy. Minor errors (dropped/swapped words) occur under emotional prosody.")
    else:
        print("Moderate WER, likely caused by strong emotional expressiveness distorting phonemes.")
    print(
        "Using the 'base' or 'small' Whisper model instead of 'tiny' would further reduce WER."
    )
    print(
        "For production use, speaker diarisation and language-model rescoring would help further."
    )
    print("------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: Whisper batch transcription")
    parser.add_argument("--input-dir", default="../examples", help="Directory of .wav files")
    parser.add_argument(
        "--output-txt",
        default="../examples/transcripts.txt",
        help="Output file for transcripts (TSV: filename<TAB>transcript)",
    )
    parser.add_argument(
        "--model-size",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for quick runs")
    parser.add_argument(
        "--accuracy-samples",
        type=int,
        default=20,
        help="Number of files to use for WER evaluation",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = (script_dir / args.input_dir).resolve()
    output_txt = (script_dir / args.output_txt).resolve()

    transcripts = transcribe_directory(
        input_dir=str(input_dir),
        output_txt=str(output_txt),
        model_size=args.model_size,
        max_files=args.max_files,
    )

    measure_accuracy(transcripts, max_samples=args.accuracy_samples)