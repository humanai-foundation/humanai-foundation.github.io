"""
Step 3 Segmenter — test suite
==============================
Loads the JSON fixtures from tests/fixtures/segmenter/ and asserts
correctness of both segmentation strategies.

Each test builds SpeechSegment objects from the fixture's known geometry
(start, end, duration) so results are 100% deterministic — no audio file
loading, no VAD, no randomness.

Run:
    python tests/generate_segmenter_fixtures.py   # once
    python tests/test_segmenter.py
    python tests/test_segmenter.py --verbose
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "utterance_buffer"))
sys.path.insert(0, str(REPO_ROOT / "vad_engine"))

from segmenter import UtteranceSegmenter, segment_utterances, Utterance  # noqa
from vad import SpeechSegment                                             # noqa

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "segmenter"

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

def _make_audio(duration_s: float, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic speech audio of the requested duration."""
    rng = np.random.default_rng(seed)
    n = max(1, int(duration_s * SAMPLE_RATE))
    t = np.linspace(0, duration_s, n, endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 160.0 * t).astype(np.float32)
    sig += rng.normal(0, 0.01, n).astype(np.float32)
    return sig


def load_fixture(name: str):
    """Return (list[SpeechSegment], meta_dict) from a JSON fixture file."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run generate_segmenter_fixtures.py first")
    data = json.loads(path.read_text())
    segments = []
    for i, d in enumerate(data["segments"]):
        audio = _make_audio(d["duration_s"], seed=i)
        segments.append(SpeechSegment(start=d["start"], end=d["end"], audio=audio))
    return segments, data["meta"]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def run_test(label: str, fn):
    try:
        fn()
        status, msg = PASS, ""
    except FileNotFoundError as exc:
        status, msg = SKIP, str(exc)
    except AssertionError as exc:
        status, msg = FAIL, str(exc)
    except Exception as exc:
        status, msg = FAIL, f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    _results.append((label, status, msg))
    print({"PASS": ".", "FAIL": "F", "SKIP": "s"}[status], end="", flush=True)


def check(condition, msg=""):
    if not condition:
        raise AssertionError(msg)


def check_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# Tests — pause_triggered strategy
# ---------------------------------------------------------------------------

def test_single_segment_pause():
    segs, meta = load_fixture("single_segment")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), 1, "single_segment: utterance count")
    check_eq(utts[0].num_vad_segments, 1, "single_segment: vad_segments in utterance")


def test_two_short_gap_merged():
    """Gap 0.2 s < pause_s 0.4 s -> both segments merged into 1 utterance."""
    segs, meta = load_fixture("two_short_gap")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "two_short_gap: utterance count")
    check_eq(utts[0].num_vad_segments, 2, "two_short_gap: both segs merged")


def test_two_long_gap_split():
    """Gap 0.8 s > pause_s 0.4 s -> 2 separate utterances."""
    segs, meta = load_fixture("two_long_gap")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "two_long_gap: utterance count")


def test_four_mixed_gaps():
    """Short/long/short gaps -> exactly 2 utterances."""
    segs, meta = load_fixture("four_mixed_gaps")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "four_mixed_gaps: utterance count")
    check_eq(utts[0].num_vad_segments, 2, "four_mixed_gaps: first utterance vad_segs")
    check_eq(utts[1].num_vad_segments, 2, "four_mixed_gaps: second utterance vad_segs")


def test_safety_valve():
    """Long gap triggers split before max_utterance_s is reached -> 2 utterances."""
    segs, meta = load_fixture("safety_valve")
    utts = segment_utterances(
        segs, strategy="pause_triggered",
        pause_s=meta["pause_s"], max_utterance_s=meta["max_utterance_s"],
        verbose=False,
    )
    check_eq(len(utts), meta["expected_utterances"], "safety_valve: utterance count")


def test_empty_pause():
    segs, _ = load_fixture("empty")
    utts = segment_utterances(segs, strategy="pause_triggered", verbose=False)
    check_eq(len(utts), 0, "empty: no utterances")


def test_single_tiny_pause():
    """Segmenter emits tiny segments; filtering is VAD's responsibility."""
    segs, meta = load_fixture("single_tiny")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "single_tiny: utterance count")


# ---------------------------------------------------------------------------
# Tests — fixed_window strategy
# ---------------------------------------------------------------------------

def test_single_segment_fixed():
    segs, _ = load_fixture("single_segment")
    utts = segment_utterances(segs, strategy="fixed_window", window_s=3.0, verbose=False)
    check_eq(len(utts), 1, "fixed_window single_segment")


def test_fixed_window_exact():
    """3 × 1 s = window_s=3.0 -> emitted exactly on the 3rd segment."""
    segs, meta = load_fixture("fixed_window_exact")
    utts = segment_utterances(segs, strategy="fixed_window", window_s=meta["window_s"], verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "fixed_window_exact: utterance count")
    check_eq(utts[0].num_vad_segments, 3, "fixed_window_exact: all 3 segs in one utt")


def test_fixed_window_overflow():
    """4 × 1 s > window_s=3.0 -> emit at 3 s, flush leftover -> 2 utterances."""
    segs, meta = load_fixture("fixed_window_overflow")
    utts = segment_utterances(segs, strategy="fixed_window", window_s=meta["window_s"], verbose=False)
    check_eq(len(utts), meta["expected_utterances"], "fixed_window_overflow: utterance count")
    check_eq(utts[0].num_vad_segments, 3, "fixed_window_overflow: first utt has 3 segs")
    check_eq(utts[1].num_vad_segments, 1, "fixed_window_overflow: second utt has 1 seg")


def test_empty_fixed():
    segs, _ = load_fixture("empty")
    utts = segment_utterances(segs, strategy="fixed_window", window_s=2.0, verbose=False)
    check_eq(len(utts), 0, "fixed_window empty")


def test_single_tiny_fixed():
    segs, meta = load_fixture("single_tiny")
    utts = segment_utterances(segs, strategy="fixed_window", window_s=2.0, verbose=False)
    check_eq(len(utts), 1, "fixed_window single_tiny")


# ---------------------------------------------------------------------------
# Tests — output correctness (both strategies)
# ---------------------------------------------------------------------------

def test_utterance_start_before_end():
    """start < end for every emitted utterance."""
    segs, _ = load_fixture("four_mixed_gaps")
    for strategy in ("pause_triggered", "fixed_window"):
        utts = segment_utterances(segs, strategy=strategy, pause_s=0.4,
                                  window_s=1.5, verbose=False)
        for i, u in enumerate(utts):
            check(u.start < u.end,
                  f"{strategy} utt[{i}]: start {u.start} >= end {u.end}")


def test_utterance_no_overlap():
    """No two utterances from the same call may overlap in time."""
    segs, _ = load_fixture("four_mixed_gaps")
    for strategy in ("pause_triggered", "fixed_window"):
        utts = segment_utterances(segs, strategy=strategy, pause_s=0.4,
                                  window_s=1.5, verbose=False)
        for i in range(len(utts) - 1):
            check(
                utts[i].end <= utts[i + 1].start,
                f"{strategy}: utt[{i}].end={utts[i].end:.3f} "
                f"> utt[{i+1}].start={utts[i+1].start:.3f}",
            )


def test_utterance_audio_dtype():
    """Every utterance audio must be float32."""
    segs, _ = load_fixture("four_mixed_gaps")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    for i, u in enumerate(utts):
        check(u.audio.dtype == np.float32,
              f"utt[{i}] dtype={u.audio.dtype}, expected float32")


def test_utterance_audio_1d():
    """Every utterance audio must be a 1-D array."""
    segs, _ = load_fixture("four_mixed_gaps")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    for i, u in enumerate(utts):
        check(u.audio.ndim == 1, f"utt[{i}] ndim={u.audio.ndim}, expected 1")


def test_utterance_audio_length_matches_segments():
    """Concatenated audio length should equal sum of constituent segment lengths."""
    segs, _ = load_fixture("two_short_gap")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(len(utts), 1, "pre-condition: 1 utterance")
    expected_samples = sum(len(s.audio) for s in segs)
    check_eq(len(utts[0].audio), expected_samples, "audio length == sum of segment lengths")


def test_strategy_field_set_correctly():
    """utterance.strategy must match the strategy passed to the segmenter."""
    segs, _ = load_fixture("single_segment")
    for strategy in ("pause_triggered", "fixed_window"):
        utts = segment_utterances(segs, strategy=strategy, verbose=False)
        for u in utts:
            check_eq(u.strategy, strategy, "strategy field")


def test_num_vad_segments_accurate():
    """num_vad_segments must equal the actual number of segments merged."""
    segs, _ = load_fixture("two_short_gap")
    utts = segment_utterances(segs, strategy="pause_triggered", pause_s=0.4, verbose=False)
    check_eq(utts[0].num_vad_segments, len(segs), "num_vad_segments")


def test_stream_segments_matches_process_segments():
    """stream_segments() generator must yield identical results to process_segments()."""
    segs, _ = load_fixture("four_mixed_gaps")
    segmenter1 = UtteranceSegmenter(strategy="pause_triggered", pause_s=0.4, verbose=False)
    segmenter2 = UtteranceSegmenter(strategy="pause_triggered", pause_s=0.4, verbose=False)

    batch = segmenter1.process_segments(segs)
    streamed = list(segmenter2.stream_segments(segs))

    check_eq(len(batch), len(streamed), "stream vs batch: count")
    for i, (b, s) in enumerate(zip(batch, streamed)):
        check_eq(b.start, s.start, f"utt[{i}].start")
        check_eq(b.end,   s.end,   f"utt[{i}].end")
        check_eq(b.num_vad_segments, s.num_vad_segments, f"utt[{i}].num_vad_segments")


def test_flush_empty_buffer_returns_none():
    """flush() on an empty segmenter must return None."""
    seg = UtteranceSegmenter(strategy="pause_triggered", verbose=False)
    result = seg.flush()
    check(result is None, "flush on empty buffer should return None")


def test_buffered_duration_tracking():
    """buffered_duration increases as segments are fed, resets after emit."""
    segs, _ = load_fixture("two_long_gap")
    seg = UtteranceSegmenter(strategy="pause_triggered", pause_s=0.4, verbose=False)

    seg.feed_segment(segs[0])
    check(seg.buffered_duration > 0, "buffered_duration > 0 after first feed")

    seg.feed_segment(segs[1])  # long gap -> emits first, then buffers second
    # After emit + new segment buffered, duration equals second segment's duration
    check(
        abs(seg.buffered_duration - segs[1].duration) < 0.01,
        f"after emit, buffered_duration should equal second seg duration "
        f"({segs[1].duration:.3f}s), got {seg.buffered_duration:.3f}s",
    )


def test_pause_threshold_boundary():
    """Verify merge vs split on either side of the pause threshold."""
    from vad import SpeechSegment as SS
    audio = _make_audio(1.0)

    # gap = 0.39 s < pause_s=0.4 s → should merge into 1 utterance
    seg1a = SS(start=0.0, end=1.0, audio=audio)
    seg1b = SS(start=1.39, end=2.39, audio=audio)
    utts_merge = segment_utterances(
        [seg1a, seg1b], strategy="pause_triggered", pause_s=0.4, verbose=False
    )
    check_eq(len(utts_merge), 1, "gap 0.39s < pause_s 0.4s should merge")

    # gap = 0.5 s > pause_s=0.4 s → should split into 2 utterances
    seg2a = SS(start=0.0, end=1.0, audio=audio)
    seg2b = SS(start=1.5, end=2.5, audio=audio)
    utts_split = segment_utterances(
        [seg2a, seg2b], strategy="pause_triggered", pause_s=0.4, verbose=False
    )
    check_eq(len(utts_split), 2, "gap 0.5s > pause_s 0.4s should split")


def test_invalid_strategy_raises():
    """Passing an unknown strategy must raise ValueError immediately."""
    try:
        UtteranceSegmenter(strategy="unknown_strategy", verbose=False)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        pass  # expected


# ---------------------------------------------------------------------------
# Test registry & runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # pause_triggered
    ("pause | single segment",             test_single_segment_pause),
    ("pause | two segs, short gap merged", test_two_short_gap_merged),
    ("pause | two segs, long gap split",   test_two_long_gap_split),
    ("pause | four mixed gaps",            test_four_mixed_gaps),
    ("pause | safety valve",               test_safety_valve),
    ("pause | empty input",                test_empty_pause),
    ("pause | single tiny segment",        test_single_tiny_pause),
    # fixed_window
    ("fixed | single segment",             test_single_segment_fixed),
    ("fixed | exact window fill",          test_fixed_window_exact),
    ("fixed | overflow -> 2 utterances",    test_fixed_window_overflow),
    ("fixed | empty input",                test_empty_fixed),
    ("fixed | single tiny segment",        test_single_tiny_fixed),
    # output correctness
    ("output | start < end",               test_utterance_start_before_end),
    ("output | no overlap",                test_utterance_no_overlap),
    ("output | audio dtype float32",       test_utterance_audio_dtype),
    ("output | audio is 1-D",              test_utterance_audio_1d),
    ("output | audio length == sum segs",  test_utterance_audio_length_matches_segments),
    ("output | strategy field set",        test_strategy_field_set_correctly),
    ("output | num_vad_segments accurate", test_num_vad_segments_accurate),
    # api consistency
    ("api | stream == process",            test_stream_segments_matches_process_segments),
    ("api | flush empty -> None",           test_flush_empty_buffer_returns_none),
    ("api | buffered_duration tracking",   test_buffered_duration_tracking),
    ("api | pause threshold boundary",     test_pause_threshold_boundary),
    ("api | invalid strategy raises",      test_invalid_strategy_raises),
]


def main(verbose: bool = False):
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.json")):
        print(
            "No segmenter fixtures found.  Generate them first:\n"
            "  python tests/generate_segmenter_fixtures.py\n"
        )
        sys.exit(1)

    print(f"Running {len(ALL_TESTS)} segmenter tests\n")
    print("Legend: . = pass   F = fail   s = skip\n")

    for label, fn in ALL_TESTS:
        run_test(label, fn)

    print("\n")
    passed  = sum(1 for _, s, _ in _results if s == PASS)
    failed  = sum(1 for _, s, _ in _results if s == FAIL)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)

    if verbose or failed:
        print("-" * 62)
        for label, status, msg in _results:
            line = f"  [{status}] {label}"
            if msg:
                line += f"\n         {msg}"
            print(line)
        print("-" * 62)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)
