"""
Step 2 VAD — test suite
========================
Runs VADProcessor against the synthetic fixtures produced by
generate_test_audio.py and asserts correctness of:
  - segment count
  - timestamp plausibility (start < end, within file bounds)
  - segment duration >= min_speech_ms
  - no overlap between segments
  - audio array shape and dtype

Run:
    # Generate fixtures first (once):
    python tests/generate_test_audio.py

    # Then run tests:
    python tests/test_vad.py

    # Verbose output (shows each detected segment):
    python tests/test_vad.py --verbose
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

# Resolve paths relative to this file
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vad_engine"))
sys.path.insert(0, str(REPO_ROOT / "task0_audio_capture"))

from vad import detect_speech_segments, VADProcessor  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# VAD settings kept constant across all tests
# ---------------------------------------------------------------------------
VAD_KWARGS = dict(
    sample_rate=16000,
    frame_ms=20,
    aggressiveness=2,
    speech_pad_ms=300,
    silence_pad_ms=400,
    min_speech_ms=250,
    verbose=False,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results = []


def _load(name: str):
    path = FIXTURES_DIR / name
    if not path.exists():
        return None, 0
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def run_test(name: str, fn):
    """Execute a test function, record PASS/FAIL/SKIP."""
    try:
        result = fn()
        status = PASS if result else FAIL
        msg = "" if result else "assertion returned False"
    except FileNotFoundError as exc:
        status = SKIP
        msg = str(exc)
    except AssertionError as exc:
        status = FAIL
        msg = str(exc)
    except Exception as exc:
        status = FAIL
        msg = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    _results.append((name, status, msg))
    marker = {"PASS": ".", "FAIL": "F", "SKIP": "s"}[status]
    print(marker, end="", flush=True)
    return status


def assert_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected!r}, got {actual!r}"
        )
    return True


def assert_gte(actual, minimum, label=""):
    if actual < minimum:
        raise AssertionError(f"{label}: {actual!r} < minimum {minimum!r}")
    return True


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_silence_only():
    """Pure silence must produce zero segments."""
    audio, sr = _load("silence_only.wav")
    if audio is None:
        raise FileNotFoundError("silence_only.wav not found — run generate_test_audio.py first")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_eq(len(segs), 0, "silence_only segment count")
    return True


def test_speech_only_count():
    """Continuous speech must produce exactly 1 segment."""
    audio, sr = _load("speech_only.wav")
    if audio is None:
        raise FileNotFoundError("speech_only.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_eq(len(segs), 1, "speech_only segment count")
    return True


def test_speech_only_duration():
    """The single segment should span most of the 3 s file."""
    audio, sr = _load("speech_only.wav")
    if audio is None:
        raise FileNotFoundError("speech_only.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert len(segs) == 1, "Expected 1 segment"
    assert_gte(segs[0].duration, 2.0, "speech_only segment duration")
    return True


def test_short_burst_discarded():
    """A 100 ms burst is shorter than min_speech_ms=250 ms and must be discarded."""
    audio, sr = _load("short_burst.wav")
    if audio is None:
        raise FileNotFoundError("short_burst.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_eq(len(segs), 0, "short_burst segment count")
    return True


def test_speech_gap_speech_count():
    """Two utterances separated by 1 s silence must yield exactly 2 segments."""
    audio, sr = _load("speech_gap_speech.wav")
    if audio is None:
        raise FileNotFoundError("speech_gap_speech.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_eq(len(segs), 2, "speech_gap_speech segment count")
    return True


def test_speech_gap_speech_order():
    """Second segment must start after first segment ends."""
    audio, sr = _load("speech_gap_speech.wav")
    if audio is None:
        raise FileNotFoundError("speech_gap_speech.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert len(segs) == 2, "Need 2 segments to check order"
    if segs[0].end >= segs[1].start:
        raise AssertionError(
            f"Segments overlap: first ends at {segs[0].end:.3f}s, "
            f"second starts at {segs[1].start:.3f}s"
        )
    return True


def test_multi_segment_count():
    """Four speech islands must produce exactly 4 segments."""
    audio, sr = _load("multi_segment.wav")
    if audio is None:
        raise FileNotFoundError("multi_segment.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_eq(len(segs), 4, "multi_segment count")
    return True


def test_noisy_speech_detected():
    """VAD must find speech even when background noise is present."""
    audio, sr = _load("noisy_speech.wav")
    if audio is None:
        raise FileNotFoundError("noisy_speech.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_gte(len(segs), 1, "noisy_speech: at least 1 segment")
    return True


def test_quiet_speech_detected():
    """Low-amplitude (0.08) speech should still be detected."""
    audio, sr = _load("quiet_speech.wav")
    if audio is None:
        raise FileNotFoundError("quiet_speech.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert_gte(len(segs), 1, "quiet_speech: at least 1 segment")
    return True


def test_segment_timestamps_within_bounds():
    """Every segment's start/end must lie within [0, file_duration]."""
    audio, sr = _load("multi_segment.wav")
    if audio is None:
        raise FileNotFoundError("multi_segment.wav not found")
    total = len(audio) / sr
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    for i, seg in enumerate(segs):
        if seg.start < 0:
            raise AssertionError(f"Segment {i} start {seg.start:.3f}s < 0")
        if seg.end > total + 0.1:   # allow 100 ms rounding slack
            raise AssertionError(
                f"Segment {i} end {seg.end:.3f}s > total {total:.3f}s"
            )
        if seg.start >= seg.end:
            raise AssertionError(
                f"Segment {i}: start {seg.start:.3f}s >= end {seg.end:.3f}s"
            )
    return True


def test_segment_min_duration():
    """Every emitted segment must be >= min_speech_ms long."""
    min_s = VAD_KWARGS["min_speech_ms"] / 1000.0
    audio, sr = _load("multi_segment.wav")
    if audio is None:
        raise FileNotFoundError("multi_segment.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    for i, seg in enumerate(segs):
        if seg.duration < min_s - 0.02:   # 20 ms frame rounding tolerance
            raise AssertionError(
                f"Segment {i} duration {seg.duration:.3f}s < min {min_s:.3f}s"
            )
    return True


def test_segment_audio_dtype_and_shape():
    """seg.audio must be a 1-D float32 numpy array with the right sample count."""
    audio, sr = _load("speech_only.wav")
    if audio is None:
        raise FileNotFoundError("speech_only.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    assert len(segs) == 1, "Need 1 segment for dtype/shape test"
    seg = segs[0]
    if seg.audio.ndim != 1:
        raise AssertionError(f"seg.audio.ndim={seg.audio.ndim}, expected 1")
    if seg.audio.dtype != np.float32:
        raise AssertionError(f"seg.audio.dtype={seg.audio.dtype}, expected float32")
    expected_samples = int(seg.duration * sr)
    tolerance = sr // 10   # 100 ms tolerance
    diff = abs(len(seg.audio) - expected_samples)
    if diff > tolerance:
        raise AssertionError(
            f"seg.audio length {len(seg.audio)} differs from "
            f"expected {expected_samples} by {diff} samples (>{tolerance})"
        )
    return True


def test_no_overlap():
    """No two segments may overlap in time."""
    audio, sr = _load("multi_segment.wav")
    if audio is None:
        raise FileNotFoundError("multi_segment.wav not found")
    segs = detect_speech_segments(audio, **VAD_KWARGS)
    for i in range(len(segs) - 1):
        if segs[i].end > segs[i + 1].start:
            raise AssertionError(
                f"Segments {i} and {i+1} overlap: "
                f"{segs[i].end:.3f}s > {segs[i+1].start:.3f}s"
            )
    return True


def test_aggressiveness_modes():
    """VADProcessor must initialise and produce results for all aggressiveness levels."""
    audio, sr = _load("speech_only.wav")
    if audio is None:
        raise FileNotFoundError("speech_only.wav not found")
    for mode in [0, 1, 2, 3]:
        kwargs = {**VAD_KWARGS, "aggressiveness": mode}
        segs = detect_speech_segments(audio, **kwargs)
        # We just need it not to crash; segment count may vary by mode
        assert isinstance(segs, list), f"mode={mode}: result is not a list"
    return True


def test_stream_offset():
    """stream_offset_s must shift all timestamps by the given offset."""
    audio, sr = _load("speech_only.wav")
    if audio is None:
        raise FileNotFoundError("speech_only.wav not found")
    processor = VADProcessor(**VAD_KWARGS)
    offset = 10.0
    segs = list(processor.process_array(audio, stream_offset_s=offset))
    assert len(segs) >= 1, "Expected at least 1 segment"
    if segs[0].start < offset - 0.1:
        raise AssertionError(
            f"start={segs[0].start:.3f}s not shifted by offset={offset}s"
        )
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("silence_only: 0 segments",         test_silence_only),
    ("speech_only: 1 segment",           test_speech_only_count),
    ("speech_only: segment duration",    test_speech_only_duration),
    ("short_burst: discarded",           test_short_burst_discarded),
    ("speech_gap_speech: 2 segments",    test_speech_gap_speech_count),
    ("speech_gap_speech: order",         test_speech_gap_speech_order),
    ("multi_segment: 4 segments",        test_multi_segment_count),
    ("noisy_speech: detected",           test_noisy_speech_detected),
    ("quiet_speech: detected",           test_quiet_speech_detected),
    ("timestamps within bounds",         test_segment_timestamps_within_bounds),
    ("min segment duration",             test_segment_min_duration),
    ("audio dtype and shape",            test_segment_audio_dtype_and_shape),
    ("no segment overlap",               test_no_overlap),
    ("all aggressiveness modes",         test_aggressiveness_modes),
    ("stream_offset shifts timestamps",  test_stream_offset),
]


def main(verbose: bool = False):
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.wav")):
        print(
            "No fixtures found.  Generate them first:\n"
            "  python tests/generate_test_audio.py\n"
        )
        sys.exit(1)

    print(f"Running {len(ALL_TESTS)} VAD tests\n")
    print("Legend: . = pass   F = fail   s = skip\n")

    for label, fn in ALL_TESTS:
        run_test(label, fn)

    print("\n")
    passed = sum(1 for _, s, _ in _results if s == PASS)
    failed = sum(1 for _, s, _ in _results if s == FAIL)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)

    if verbose or failed:
        print("-" * 60)
        for label, status, msg in _results:
            line = f"  [{status}] {label}"
            if msg:
                line += f"\n         {msg}"
            print(line)
        print("-" * 60)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)
