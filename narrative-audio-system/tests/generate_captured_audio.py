"""
Generate captured_audio.wav for pipeline testing by concatenating real
RAVDESS speech files — one per emotion — with 0.8 s silence between them.

RAVDESS filename convention (3rd field = emotion):
  01=neutral  02=calm  03=happy  04=sad
  05=angry    06=fearful  07=disgust  08=surprised

Selected clips (one speaker, both statements):
  calm    -> 03-01-02-01-02-01-01.wav
  happy   -> 03-01-03-01-01-01-01.wav
  angry   -> 03-01-05-01-02-01-01.wav   (tense/angry)
  fearful -> 03-01-06-01-01-01-01.wav

Each clip is resampled to 16 kHz mono and RMS-normalised to a consistent
loudness before concatenation so VAD fires evenly on all segments.

Output: examples/captured_audio.wav

Run:
    python tests/generate_captured_audio.py
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT     = Path(__file__).resolve().parent.parent
EX_DIR   = ROOT / "examples"
OUT_PATH = EX_DIR / "captured_audio.wav"
SR       = 16000
PAUSE_S  = 0.8   # silence between utterances
TARGET_RMS = 0.18


def _load_resample(path: Path) -> np.ndarray:
    """Load a wav file, convert to 16 kHz mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def _rms_normalise(audio: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return (audio * (target / rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pick one representative file per emotion from what exists in examples/
# Fallback: scan for any file matching the emotion code if preferred missing.
# ---------------------------------------------------------------------------

def _pick(emotion_code: str) -> Path:
    """Return first available file for this emotion code (field 3 = 1-indexed)."""
    pattern = f"03-01-{emotion_code:02d}-*.wav"
    matches = sorted(EX_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No RAVDESS file matching {pattern} in {EX_DIR}.\n"
            f"Download RAVDESS or run: python examples/download_ravdess.py"
        )
    # Prefer statement 2 ("children are talking") actor 1 if available
    preferred = [m for m in matches if m.name.endswith("-02-01-01.wav")]
    return preferred[0] if preferred else matches[0]


CLIPS = [
    ("calm",    _pick(2)),
    ("happy",   _pick(3)),
    ("angry",   _pick(5)),
    ("fearful", _pick(6)),
]

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

silence = np.zeros(int(PAUSE_S * SR), dtype=np.float32)
lead_in = np.zeros(int(0.3 * SR),    dtype=np.float32)

audio = lead_in
for i, (label, path) in enumerate(CLIPS):
    seg = _rms_normalise(_load_resample(path))
    audio = np.concatenate([audio, seg])
    if i < len(CLIPS) - 1:
        audio = np.concatenate([audio, silence])

audio = np.concatenate([audio, np.zeros(int(0.3 * SR), dtype=np.float32)])

sf.write(str(OUT_PATH), audio, SR, subtype="PCM_16")

total_s = len(audio) / SR
print(f"[generate_captured_audio] Wrote {OUT_PATH}")
print(f"  Duration : {total_s:.2f}s  |  {len(CLIPS)} utterances  "
      f"(+{PAUSE_S}s gaps)")
for label, path in CLIPS:
    seg = _rms_normalise(_load_resample(path))
    rms = np.sqrt(np.mean(seg ** 2))
    print(f"    [{label:<7}]  {path.name}  ({len(seg)/SR:.2f}s, RMS={rms:.4f})")
