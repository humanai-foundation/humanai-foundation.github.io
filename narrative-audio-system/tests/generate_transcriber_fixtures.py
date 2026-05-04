"""
Generate test fixtures for Step 4 Streaming Transcription tests.

Three fixture categories
------------------------

1. unit_fixtures/  (JSON only, no audio — pure API/structural tests)
   Synthetic Utterance descriptors with known start/end/duration so
   TranscriptionResult metadata can be validated without running a model.

2. integration_fixtures/  (JSON manifests pointing to real WAV files)
   Selects a small subset of RAVDESS examples with known ground-truth text.
   RAVDESS filenames encode the spoken statement:
       position 5 (0-indexed 4) == "01"  ->  "Kids are talking by the door"
       position 5 (0-indexed 4) == "02"  ->  "Dogs are sitting by the door"
   Used by integration tests that actually invoke the transcriber model.

3. edge_fixtures/  (WAV files for boundary-condition tests)
   silence_3s.wav   — 3 s of pure silence
   noise_3s.wav     — 3 s of broadband noise (no speech)
   tiny_speech.wav  — 50 ms synthetic speech (very short)
   long_speech.wav  — 8 s continuous synthetic speech

Run:
    python tests/generate_transcriber_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUT_ROOT = Path(__file__).parent / "fixtures" / "transcriber"
UNIT_DIR = OUT_ROOT / "unit"
INTEG_DIR = OUT_ROOT / "integration"
EDGE_DIR = OUT_ROOT / "edge"

for d in (UNIT_DIR, INTEG_DIR, EDGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(seed=99)

STATEMENT_TEXT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(duration_s: float) -> np.ndarray:
    return RNG.normal(0, 1e-5, int(duration_s * SAMPLE_RATE)).astype(np.float32)


def _speech(duration_s: float, amplitude: float = 0.35) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float32)
    for k, gain in enumerate([1.0, 0.6, 0.4, 0.3, 0.2], start=1):
        sig += gain * np.sin(2 * np.pi * 160.0 * k * t).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-9
    sig *= amplitude
    fade = min(int(0.01 * SAMPLE_RATE), n // 4)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    sig[:fade] *= ramp
    sig[-fade:] *= ramp[::-1]
    return sig


def _write_wav(name: str, audio: np.ndarray, directory: Path) -> Path:
    path = directory / name
    sf.write(str(path), audio, SAMPLE_RATE)
    return path


def _ravdess_statement(filename: str) -> str:
    """Parse RAVDESS filename to get ground-truth spoken text."""
    parts = Path(filename).stem.split("-")
    if len(parts) >= 5:
        return STATEMENT_TEXT.get(parts[4], "")
    return ""


# ---------------------------------------------------------------------------
# 1. Unit fixtures — pure JSON, no audio files
# ---------------------------------------------------------------------------

def generate_unit_fixtures():
    fixtures = [
        {
            "name": "single_utterance",
            "utterances": [{"start": 1.2, "end": 3.84, "duration_s": 2.64}],
            "note": "1 utterance -> 1 TranscriptionResult",
            "expected_results": 1,
        },
        {
            "name": "three_utterances",
            "utterances": [
                {"start": 0.5,  "end": 2.5,  "duration_s": 2.0},
                {"start": 3.1,  "end": 5.2,  "duration_s": 2.1},
                {"start": 6.0,  "end": 7.8,  "duration_s": 1.8},
            ],
            "note": "3 utterances -> 3 results; full_transcript joins all",
            "expected_results": 3,
        },
        {
            "name": "empty_utterances",
            "utterances": [],
            "note": "no utterances -> no results, full_transcript is empty",
            "expected_results": 0,
        },
        {
            "name": "zero_duration_edge",
            "utterances": [{"start": 2.0, "end": 2.0, "duration_s": 0.0}],
            "note": "zero-duration utterance should not crash",
            "expected_results": 1,
        },
    ]
    for fx in fixtures:
        path = UNIT_DIR / f"{fx['name']}.json"
        path.write_text(json.dumps(fx, indent=2))
        print(f"  unit/{fx['name']}.json  ({fx['expected_results']} result(s)) [{fx['note']}]")


# ---------------------------------------------------------------------------
# 2. Integration fixtures — real RAVDESS audio with known text
# ---------------------------------------------------------------------------

def generate_integration_fixtures():
    if not EXAMPLES_DIR.exists():
        print("  [SKIP] examples/ not found — skipping integration fixtures")
        return

    # Pick 2 neutral files per statement (statement 01 and 02)
    selected = []
    for stmt_code, text in STATEMENT_TEXT.items():
        matches = sorted(EXAMPLES_DIR.glob(f"03-01-*-*-{stmt_code}-*-*.wav"))[:2]
        for wav in matches:
            selected.append({
                "filename": wav.name,
                "filepath": str(wav.resolve()),
                "statement_code": stmt_code,
                "ground_truth": text,
                # RAVDESS duration is typically 3-4 s
                "start_s": 0.0,
            })

    manifest = {
        "description": "RAVDESS files with known ground-truth text",
        "ground_truth_note": (
            "Whisper tiny may paraphrase. "
            "Tests check for keyword presence, not exact match."
        ),
        "keywords": {
            "01": ["kids", "talking", "door"],
            "02": ["dogs", "sitting", "door"],
        },
        "files": selected,
    }
    path = INTEG_DIR / "ravdess_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  integration/ravdess_manifest.json  ({len(selected)} file(s))")


# ---------------------------------------------------------------------------
# 3. Edge fixtures — WAV files for boundary conditions
# ---------------------------------------------------------------------------

def generate_edge_fixtures():
    edges = [
        ("silence_3s.wav",    _silence(3.0),
         "3 s pure silence -> transcriber returns empty/whitespace text"),
        ("noise_3s.wav",      RNG.normal(0, 0.15, int(3.0 * SAMPLE_RATE)).astype(np.float32),
         "broadband noise -> transcriber may hallucinate; result must still be str"),
        ("tiny_speech.wav",   np.concatenate([_silence(0.05), _speech(0.05), _silence(0.05)]),
         "50 ms speech clip -> result must be str without crashing"),
        ("long_speech.wav",   np.concatenate([_silence(0.3), _speech(8.0), _silence(0.3)]),
         "8 s speech -> tests longer utterance handling"),
    ]

    manifest_entries = []
    for name, audio, note in edges:
        path = _write_wav(name, audio, EDGE_DIR)
        duration = len(audio) / SAMPLE_RATE
        manifest_entries.append({
            "filename": name,
            "filepath": str(path.resolve()),
            "duration_s": round(duration, 3),
            "note": note,
        })
        print(f"  edge/{name:<25}  {duration:.2f} s  [{note[:55]}...]")

    manifest = {"files": manifest_entries}
    (EDGE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating transcriber fixtures in {OUT_ROOT}/\n")

    print("Unit fixtures:")
    generate_unit_fixtures()

    print("\nIntegration fixtures:")
    generate_integration_fixtures()

    print("\nEdge fixtures:")
    generate_edge_fixtures()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
