"""
Generate long synthetic audio test files with distinct emotional tones
for testing the Step 5 EmotionClassifier.

Each file is 6–8 s long and is designed with acoustic properties that
match the MFCC fingerprint each emotion class is known to produce:

  calm       — steady low pitch (~120 Hz), soft energy, slow regular rhythm
  happy      — higher pitch (~200 Hz), upward inflections, brighter harmonics
  angry      — high energy, harsh upper harmonics, fast irregular bursts
  sad        — low pitch (~90 Hz), falling contour, sparse energy
  fearful    — breathy, mid-high pitch (~180 Hz), erratic amplitude
  neutral    — flat pitch (~150 Hz), constant moderate energy
  surprised  — sudden high-pitch onset (~260 Hz), short burst then decay
  disgust    — low-mid pitch, strong low-frequency noise, staccato bursts

Files written to tests/fixtures/emotion_audio/:
  calm_6s.wav        happy_7s.wav        angry_8s.wav
  sad_6s.wav         fearful_7s.wav      neutral_6s.wav
  surprised_5s.wav   disgust_7s.wav

A manifest (manifest.json) records the expected label for each file.

Run:
    python tests/generate_emotion_audio.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.fft import rfft, irfft

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).parent / "fixtures" / "emotion_audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(seed=777)


# ---------------------------------------------------------------------------
# Low-level audio primitives
# ---------------------------------------------------------------------------

def _silence(duration_s: float) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    return RNG.normal(0, 5e-5, n).astype(np.float32)


def _bandpass(signal: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase bandpass via FFT zeroing."""
    spectrum = rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    return irfft(spectrum, n=len(signal)).astype(np.float32)


def _voiced(
    duration_s: float,
    f0: float = 150.0,
    f0_vibrato_hz: float = 0.0,
    f0_vibrato_depth: float = 0.0,
    harmonic_gains: tuple = (1.0, 0.6, 0.4, 0.25, 0.15, 0.1),
    amplitude: float = 0.35,
    spectral_tilt: float = 0.0,   # dB/octave, negative = darker
    formant_noise_amp: float = 0.04,
    formant_low: float = 400.0,
    formant_high: float = 3500.0,
    breathiness: float = 0.0,     # additive breathiness (0-1)
) -> np.ndarray:
    """
    Synthesise voiced speech-like audio.

    Parameters control the acoustic features that distinguish emotions:
      f0              — fundamental frequency (pitch)
      f0_vibrato_*    — slow pitch modulation (expressiveness)
      harmonic_gains  — relative amplitude of harmonics
      spectral_tilt   — spectral slope (harsh = more high-freq energy)
      breathiness     — unvoiced noise mixed into the signal
    """
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, endpoint=False)

    # Pitch contour with optional vibrato
    if f0_vibrato_hz > 0:
        f0_contour = f0 + f0_vibrato_depth * np.sin(2 * np.pi * f0_vibrato_hz * t)
    else:
        f0_contour = np.full(n, f0)

    # Cumulative phase for each harmonic (handles variable f0)
    dt = 1.0 / SAMPLE_RATE
    phase = np.cumsum(2 * np.pi * f0_contour * dt)

    signal = np.zeros(n, dtype=np.float64)
    for k, gain in enumerate(harmonic_gains, start=1):
        # Apply spectral tilt: gain_k *= k^(tilt/20)
        tilt_factor = (k ** (spectral_tilt / 20.0)) if spectral_tilt != 0 else 1.0
        signal += gain * tilt_factor * np.sin(k * phase)

    # Formant noise
    formant_noise = RNG.normal(0, formant_noise_amp, n)
    formant_noise = _bandpass(formant_noise.astype(np.float32), formant_low, formant_high)
    signal += formant_noise

    # Breathiness (wideband noise, softer)
    if breathiness > 0:
        breath = RNG.normal(0, breathiness, n).astype(np.float32)
        breath = _bandpass(breath, 1000, 8000)
        signal += breath

    signal = signal.astype(np.float32)
    peak = np.abs(signal).max()
    if peak > 0:
        signal /= peak
    signal *= amplitude

    # Fade edges
    fade = min(int(0.015 * SAMPLE_RATE), n // 8)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    signal[:fade] *= ramp
    signal[-fade:] *= ramp[::-1]

    return signal


def _amplitude_envelope(signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Multiply signal by a pre-built amplitude envelope (same length)."""
    env = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, len(envelope)),
        envelope,
    ).astype(np.float32)
    return (signal * env).astype(np.float32)


def _phrase_rhythm(
    phrase_duration_s: float,
    pause_duration_s: float,
    num_phrases: int,
    voiced_fn,
) -> np.ndarray:
    """Produce speech-like phrases separated by short pauses."""
    parts = []
    for _ in range(num_phrases):
        parts.append(voiced_fn(phrase_duration_s))
        parts.append(_silence(pause_duration_s))
    return np.concatenate(parts).astype(np.float32)


def _save(name: str, audio: np.ndarray, label: str) -> dict:
    path = OUT_DIR / name
    sf.write(str(path), audio, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"  {name:<28}  {duration:.2f}s  label={label!r}")
    return {
        "filename": name,
        "filepath": str(path.resolve()),
        "label": label,
        "duration_s": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
    }


# ---------------------------------------------------------------------------
# Emotion-specific audio generators
# ---------------------------------------------------------------------------

def make_calm() -> np.ndarray:
    """
    Calm narration: steady low pitch (120 Hz), soft regular rhythm,
    minimal spectral energy above 2 kHz, gentle vibrato.
    Duration: ~6 s
    """
    def phrase(d):
        return _voiced(d, f0=120.0, f0_vibrato_hz=4.5, f0_vibrato_depth=3.0,
                       harmonic_gains=(1.0, 0.55, 0.3, 0.15, 0.08, 0.04),
                       amplitude=0.28, spectral_tilt=-2.0,
                       formant_noise_amp=0.02, formant_high=2500.0)

    return _phrase_rhythm(0.9, 0.35, 5, phrase)


def make_happy() -> np.ndarray:
    """
    Happy speech: elevated pitch (200 Hz), bright upper harmonics,
    upward-inflected phrases, bouncy rhythm.
    Duration: ~7 s
    """
    parts = []
    pitches = [195.0, 210.0, 200.0, 220.0, 205.0, 215.0]
    for i, f0 in enumerate(pitches):
        seg = _voiced(0.75, f0=f0, f0_vibrato_hz=5.0, f0_vibrato_depth=8.0,
                      harmonic_gains=(1.0, 0.7, 0.55, 0.4, 0.3, 0.2),
                      amplitude=0.38, spectral_tilt=1.5,
                      formant_noise_amp=0.05, formant_high=4000.0)
        # Rising envelope on each phrase
        env = np.linspace(0.6, 1.0, len(seg)).astype(np.float32)
        parts.append(seg * env)
        parts.append(_silence(0.22 if i % 2 == 0 else 0.15))
    return np.concatenate(parts).astype(np.float32)


def make_angry() -> np.ndarray:
    """
    Angry speech: high energy, harsh upper harmonics (spectral_tilt +4),
    fast irregular bursts, clipped amplitude shape.
    Duration: ~8 s
    """
    parts = []
    phrase_lens = [0.55, 0.4, 0.7, 0.45, 0.6, 0.5, 0.65, 0.4]
    pause_lens  = [0.1,  0.08, 0.12, 0.09, 0.1, 0.08, 0.11, 0.15]
    for plen, slen in zip(phrase_lens, pause_lens):
        f0 = float(RNG.uniform(165, 195))
        seg = _voiced(plen, f0=f0, f0_vibrato_hz=0.0,
                      harmonic_gains=(1.0, 0.8, 0.65, 0.55, 0.45, 0.4),
                      amplitude=0.55, spectral_tilt=4.0,
                      formant_noise_amp=0.09, formant_low=300.0, formant_high=5000.0)
        # Hard-clip to simulate vocal strain
        seg = np.clip(seg, -0.45, 0.45).astype(np.float32)
        parts.append(seg)
        parts.append(_silence(slen))
    return np.concatenate(parts).astype(np.float32)


def make_sad() -> np.ndarray:
    """
    Sad speech: low pitch (90 Hz), falling contour on each phrase,
    sparse energy, slow pace, dark spectral tilt.
    Duration: ~6.5 s
    """
    parts = []
    for i in range(4):
        dur = 0.95 + i * 0.1
        seg = _voiced(dur, f0=92.0, f0_vibrato_hz=2.5, f0_vibrato_depth=2.0,
                      harmonic_gains=(1.0, 0.45, 0.22, 0.1, 0.05, 0.02),
                      amplitude=0.22, spectral_tilt=-3.5,
                      formant_noise_amp=0.015, formant_high=2000.0)
        # Falling amplitude envelope
        env = np.linspace(1.0, 0.35, len(seg)).astype(np.float32)
        parts.append(seg * env)
        parts.append(_silence(0.5 + i * 0.05))
    return np.concatenate(parts).astype(np.float32)


def make_fearful() -> np.ndarray:
    """
    Fearful speech: high mid pitch (175 Hz), breathy, erratic amplitude
    (trembling), irregular phrase lengths, high breathiness.
    Duration: ~7 s
    """
    parts = []
    for i in range(7):
        dur = float(RNG.uniform(0.45, 0.85))
        f0  = float(RNG.uniform(160, 190))
        amp = float(RNG.uniform(0.18, 0.38))
        seg = _voiced(dur, f0=f0, f0_vibrato_hz=8.0, f0_vibrato_depth=12.0,
                      harmonic_gains=(1.0, 0.5, 0.35, 0.2, 0.15, 0.1),
                      amplitude=amp, spectral_tilt=0.5,
                      formant_noise_amp=0.04, formant_high=3500.0,
                      breathiness=0.12)
        # Trembling: fast amplitude modulation
        t = np.linspace(0, dur, len(seg))
        tremor = (1.0 + 0.25 * np.sin(2 * np.pi * 7.5 * t)).astype(np.float32)
        parts.append((seg * tremor).astype(np.float32))
        parts.append(_silence(float(RNG.uniform(0.08, 0.25))))
    return np.concatenate(parts).astype(np.float32)


def make_neutral() -> np.ndarray:
    """
    Neutral narration: flat pitch (150 Hz), constant moderate energy,
    regular phrase structure, balanced spectrum.
    Duration: ~6 s
    """
    def phrase(d):
        return _voiced(d, f0=150.0, f0_vibrato_hz=0.0,
                       harmonic_gains=(1.0, 0.55, 0.35, 0.2, 0.12, 0.07),
                       amplitude=0.32, spectral_tilt=0.0,
                       formant_noise_amp=0.03, formant_high=3000.0)

    return _phrase_rhythm(0.85, 0.30, 5, phrase)


def make_surprised() -> np.ndarray:
    """
    Surprised exclamation: sudden high-pitch burst (260 Hz) at onset,
    rapid decay in pitch and energy.
    Duration: ~6 s
    """
    parts = []
    # Opening burst — very high pitch
    burst = _voiced(0.5, f0=265.0,
                    harmonic_gains=(1.0, 0.75, 0.6, 0.5, 0.4, 0.35),
                    amplitude=0.55, spectral_tilt=2.5,
                    formant_noise_amp=0.07, formant_high=5000.0)
    parts.append(burst)
    parts.append(_silence(0.15))

    # Trailing phrases — pitch falls after the initial surprise
    falling_f0 = [240.0, 215.0, 195.0, 175.0, 162.0]
    for f0 in falling_f0:
        seg = _voiced(0.6, f0=f0,
                      harmonic_gains=(1.0, 0.65, 0.45, 0.3, 0.2, 0.12),
                      amplitude=0.38, spectral_tilt=1.0,
                      formant_high=3500.0)
        parts.append(seg)
        parts.append(_silence(0.22))
    return np.concatenate(parts).astype(np.float32)


def make_disgust() -> np.ndarray:
    """
    Disgust: low-mid pitch (105 Hz), strong low-frequency noise (growl),
    staccato bursts, heavy spectral tilt emphasising lower harmonics.
    Duration: ~7 s
    """
    parts = []
    for i in range(6):
        dur = float(RNG.uniform(0.55, 0.85))
        seg = _voiced(dur, f0=105.0, f0_vibrato_hz=3.0, f0_vibrato_depth=4.0,
                      harmonic_gains=(1.0, 0.5, 0.3, 0.18, 0.1, 0.06),
                      amplitude=0.42, spectral_tilt=-1.5,
                      formant_noise_amp=0.06, formant_low=100.0, formant_high=2200.0)
        # Low growl layer
        growl = RNG.normal(0, 0.08, len(seg)).astype(np.float32)
        growl = _bandpass(growl, 60.0, 350.0)
        seg = np.clip(seg + growl * 0.35, -0.8, 0.8).astype(np.float32)
        parts.append(seg)
        parts.append(_silence(float(RNG.uniform(0.18, 0.35))))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry and writer
# ---------------------------------------------------------------------------

# Per-emotion RMS targets preserve relative energy ordering.
# angry > neutral/happy > calm/fearful > sad (matching real RAVDESS levels).
EMOTION_GENERATORS = [
    ("calm_6s.wav",      "calm",      make_calm,      0.18),
    ("happy_7s.wav",     "happy",     make_happy,     0.26),
    ("angry_8s.wav",     "angry",     make_angry,     0.38),
    ("sad_6s.wav",       "sad",       make_sad,       0.12),
    ("fearful_7s.wav",   "fearful",   make_fearful,   0.20),
    ("neutral_6s.wav",   "neutral",   make_neutral,   0.24),
    ("surprised_5s.wav", "surprised", make_surprised, 0.30),
    ("disgust_7s.wav",   "disgust",   make_disgust,   0.28),
]


def _rms_normalise(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current = float(np.sqrt(np.mean(audio ** 2)))
    if current > 1e-8:
        audio = audio * (target_rms / current)
    return np.clip(audio, -0.95, 0.95).astype(np.float32)


def main():
    print(f"Generating emotion audio fixtures in {OUT_DIR}/\n")
    manifest_entries = []
    for filename, label, generator, target_rms in EMOTION_GENERATORS:
        audio = generator()
        audio = _rms_normalise(audio, target_rms)
        entry = _save(filename, audio, label)
        manifest_entries.append(entry)

    manifest = {
        "description": (
            "Synthetic emotion audio fixtures for Step 5 classifier tests. "
            "Each file is designed with acoustic properties matching the "
            "MFCC fingerprint of its target emotion class."
        ),
        "acoustic_design": {
            "calm":      "f0=120Hz, soft energy, dark spectrum, slow rhythm",
            "happy":     "f0=200Hz, bright harmonics, rising inflections",
            "angry":     "f0=175Hz, high energy, hard-clipped, fast bursts",
            "sad":       "f0=90Hz, falling contour, sparse energy, slow",
            "fearful":   "f0=175Hz, breathy, trembling amplitude, irregular",
            "neutral":   "f0=150Hz, flat, constant energy, balanced spectrum",
            "surprised": "f0=260Hz onset falling to 160Hz, short burst",
            "disgust":   "f0=105Hz, low-freq growl, staccato, dark tilt",
        },
        "note": (
            "Classifier predictions on synthetic audio may not always match "
            "the target label — the model was trained on RAVDESS human speech. "
            "Tests assert structural correctness and soft acoustic constraints, "
            "not exact label matching."
        ),
        "files": manifest_entries,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest -> {manifest_path}")
    print(f"\nDone -- {len(manifest_entries)} emotion audio files ready.")


if __name__ == "__main__":
    main()
