"""
Generate synthetic WAV test fixtures for Step 2 VAD tests.

Each file has a known ground-truth layout of speech and silence so that
test_vad.py can assert exactly which segments the VAD should detect.

Files produced in tests/fixtures/:
  silence_only.wav       — 3 s pure silence          → 0 segments
  speech_only.wav        — 3 s continuous speech      → 1 segment
  short_burst.wav        — 0.1 s speech + silence     → 0 segments (below min)
  speech_gap_speech.wav  — 1 s speech, 1 s silence, 1 s speech → 2 segments
  multi_segment.wav      — 4 speech islands in 10 s   → 4 segments
  noisy_speech.wav       — speech buried in background noise → >=1 segment
  quiet_speech.wav       — low-amplitude speech       → >=1 segment

Run:
    python tests/generate_test_audio.py
"""

from pathlib import Path
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

def silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Near-zero noise floor — amplitude well below VAD energy threshold."""
    n = int(duration_s * sr)
    return RNG.normal(0, 1e-5, n).astype(np.float32)


def speech(duration_s: float, sr: int = SAMPLE_RATE, amplitude: float = 0.4) -> np.ndarray:
    """
    Synthetic speech: fundamental + harmonics + formant-band noise.
    Mimics voiced speech well enough for webrtcvad to label as speech.
    """
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)

    f0 = 160.0          # fundamental (Hz) — typical male voice
    signal = np.zeros(n, dtype=np.float32)

    # First 6 harmonics with falling amplitude
    for k, gain in enumerate([1.0, 0.6, 0.4, 0.3, 0.2, 0.15], start=1):
        signal += gain * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)

    # Formant-band noise (500–3500 Hz) to simulate fricatives
    noise = RNG.normal(0, 0.05, n).astype(np.float32)
    # Simple bandpass: subtract low-pass from original
    from numpy.fft import rfft, irfft
    spectrum = rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum[(freqs < 500) | (freqs > 3500)] = 0
    noise = irfft(spectrum, n=n).astype(np.float32)
    signal += noise

    # Normalise then scale
    peak = np.abs(signal).max()
    if peak > 0:
        signal /= peak
    signal *= amplitude

    # Mild amplitude envelope (avoids hard clicks at edges)
    fade = int(0.01 * sr)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    signal[:fade] *= ramp
    signal[-fade:] *= ramp[::-1]

    return signal


def concat(*segments) -> np.ndarray:
    return np.concatenate(segments).astype(np.float32)


def save(name: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = OUT_DIR / name
    sf.write(str(path), audio, sr)
    duration = len(audio) / sr
    print(f"  Wrote {path.name:35s}  {duration:.2f} s  {len(audio)} samples")
    return path


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------

FIXTURES = {}   # name -> (audio, expected_segment_count, description)


def _define(name, audio, expected_segments, description):
    FIXTURES[name] = {
        "audio": audio,
        "expected_segments": expected_segments,
        "description": description,
        "duration_s": len(audio) / SAMPLE_RATE,
    }


# 1. Pure silence — nothing should be detected
_define(
    "silence_only.wav",
    concat(silence(3.0)),
    expected_segments=0,
    description="3 s pure silence — expect 0 segments",
)

# 2. Continuous speech — one long segment
_define(
    "speech_only.wav",
    concat(speech(3.0)),
    expected_segments=1,
    description="3 s continuous speech — expect 1 segment",
)

# 3. Very short burst (100 ms) — below min_speech_ms=250, should be discarded
_define(
    "short_burst.wav",
    concat(silence(0.5), speech(0.10), silence(1.0)),
    expected_segments=0,
    description="0.1 s speech burst (below min threshold) — expect 0 segments",
)

# 4. Speech / pause / speech — two clearly separated utterances
_define(
    "speech_gap_speech.wav",
    concat(silence(0.5), speech(1.0), silence(1.0), speech(1.0), silence(0.5)),
    expected_segments=2,
    description="1 s speech, 1 s silence, 1 s speech — expect 2 segments",
)

# 5. Four speech islands — robust multi-segment detection
_define(
    "multi_segment.wav",
    concat(
        silence(0.5),
        speech(0.8),    # segment 1
        silence(0.8),
        speech(1.2),    # segment 2
        silence(1.0),
        speech(0.6),    # segment 3
        silence(0.7),
        speech(1.0),    # segment 4
        silence(0.5),
    ),
    expected_segments=4,
    description="4 speech islands in ~7 s — expect 4 segments",
)

# 6. Speech with background noise — VAD must still detect it
_define(
    "noisy_speech.wav",
    concat(
        silence(0.5),
        (lambda: (
            s := speech(2.0, amplitude=0.3),
            n := (RNG.normal(0, 0.08, len(s)).astype(np.float32)),
            np.clip(s + n, -1.0, 1.0).astype(np.float32)
        )[-1])(),
        silence(0.5),
    ),
    expected_segments=1,
    description="Speech + background noise — expect >=1 segment",
)

# 7. Low-amplitude speech — soft voice
_define(
    "quiet_speech.wav",
    concat(silence(0.3), speech(2.0, amplitude=0.08), silence(0.3)),
    expected_segments=1,
    description="Low-amplitude speech (amplitude=0.08) — expect >=1 segment",
)


# ---------------------------------------------------------------------------
# Write all fixtures
# ---------------------------------------------------------------------------

def main():
    print(f"Generating VAD test fixtures in {OUT_DIR}/\n")
    manifest_lines = [
        "# VAD test fixtures manifest",
        "# name, expected_segments, duration_s, description",
    ]

    for name, info in FIXTURES.items():
        save(name, info["audio"])
        manifest_lines.append(
            f"{name},{info['expected_segments']},{info['duration_s']:.3f},"
            f"{info['description']}"
        )

    manifest_path = OUT_DIR / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"\n  Manifest written to {manifest_path}")
    print(f"\nDone — {len(FIXTURES)} fixture files ready.")


if __name__ == "__main__":
    main()
