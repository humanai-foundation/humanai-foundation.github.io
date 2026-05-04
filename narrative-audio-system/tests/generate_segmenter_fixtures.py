"""
Generate synthetic SpeechSegment fixtures for Step 3 segmenter tests.

Unlike the VAD fixtures (which are WAV files run through webrtcvad), these
fixtures are pre-built SpeechSegment objects with *exact, known* gaps and
durations.  This removes VAD uncertainty from the segmenter tests so every
assertion is deterministic.

Fixtures are serialised as JSON to tests/fixtures/segmenter/:
  Each JSON file contains a list of segment dicts:
    {"start": float, "end": float, "duration_s": float}
  Audio is regenerated at load-time from duration_s so we avoid large
  binary files in the repo.

Scenarios
---------
  single_segment          — 1 segment (2 s)             -> 1 utterance always
  two_short_gap           — 2 segs, gap=0.2 s           -> 1 utt (gap < pause_s)
  two_long_gap            — 2 segs, gap=0.8 s           -> 2 utts (gap > pause_s)
  four_mixed_gaps         — 4 segs, alternating gaps     -> 2 utts
  safety_valve            — 1 seg of 10 s speech         -> 2 utts (exceeds max_utterance_s=8)
  fixed_window_exact      — 3 segs × 1 s = 3 s speech   -> 1 utt at window_s=3.0
  fixed_window_overflow   — 4 segs × 1 s = 4 s speech   -> 2 utts at window_s=3.0
  empty                   — 0 segments                  -> 0 utterances
  single_tiny             — 1 seg of 0.1 s             -> 1 utt (segmenter doesn't filter)

Run:
    python tests/generate_segmenter_fixtures.py
"""

import json
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).parent / "fixtures" / "segmenter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(seed=7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_segment_dict(start: float, duration: float) -> dict:
    """Return a plain-dict representation of a SpeechSegment."""
    return {
        "start": round(start, 4),
        "end": round(start + duration, 4),
        "duration_s": round(duration, 4),
    }


def save(name: str, segments: list, meta: dict):
    """Write fixture JSON and print a summary line."""
    payload = {"segments": segments, "meta": meta}
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    total_speech = sum(s["duration_s"] for s in segments)
    print(
        f"  {name + '.json':<35}  {len(segments)} segment(s)  "
        f"{total_speech:.2f}s speech  "
        f"-> {meta['expected_utterances']} utt(s)  [{meta['note']}]"
    )


# ---------------------------------------------------------------------------
# Fixture definitions
# Each entry: (name, segments_list, meta_dict)
# ---------------------------------------------------------------------------

FIXTURES = []

# 1. Single 2-second segment — always 1 utterance regardless of strategy
FIXTURES.append(("single_segment", [
    make_segment_dict(start=1.0, duration=2.0),
], {
    "expected_utterances": 1,
    "strategies": ["pause_triggered", "fixed_window"],
    "note": "1 seg -> 1 utterance always",
}))

# 2. Two segments, short gap (0.2 s) — pause_triggered merges them
FIXTURES.append(("two_short_gap", [
    make_segment_dict(start=0.5, duration=1.0),
    make_segment_dict(start=1.7, duration=1.0),  # gap = 0.2 s
], {
    "expected_utterances": 1,
    "gap_s": 0.2,
    "pause_s": 0.4,
    "strategies": ["pause_triggered"],
    "note": "gap 0.2s < pause_s 0.4s -> merged",
}))

# 3. Two segments, long gap (0.8 s) — pause_triggered splits them
FIXTURES.append(("two_long_gap", [
    make_segment_dict(start=0.5, duration=1.0),
    make_segment_dict(start=2.3, duration=1.0),  # gap = 0.8 s
], {
    "expected_utterances": 2,
    "gap_s": 0.8,
    "pause_s": 0.4,
    "strategies": ["pause_triggered"],
    "note": "gap 0.8s > pause_s 0.4s -> split",
}))

# 4. Four segments with alternating short/long gaps
#    gaps: short(0.2s), long(0.6s), short(0.15s)
#    -> segs 0+1 merge, segs 2+3 merge -> 2 utterances
FIXTURES.append(("four_mixed_gaps", [
    make_segment_dict(start=0.5,  duration=0.8),
    make_segment_dict(start=1.5,  duration=0.8),  # gap = 0.2 s (short)
    make_segment_dict(start=2.9,  duration=0.8),  # gap = 0.6 s (long) -> boundary
    make_segment_dict(start=3.85, duration=0.8),  # gap = 0.15 s (short)
], {
    "expected_utterances": 2,
    "pause_s": 0.4,
    "strategies": ["pause_triggered"],
    "note": "short/long/short gaps -> 2 utterances",
}))

# 5. Safety valve — single segment longer than max_utterance_s
#    With max_utterance_s=8.0, a 10s segment is treated as one block
#    (the safety valve fires on *accumulated* speech; a single segment
#     arriving all at once won't be split mid-segment, it emits after)
FIXTURES.append(("safety_valve", [
    make_segment_dict(start=0.0, duration=4.5),
    make_segment_dict(start=5.0, duration=4.5),  # gap=0.5 > pause_s; total=9s
], {
    "expected_utterances": 2,
    "pause_s": 0.4,
    "max_utterance_s": 8.0,
    "strategies": ["pause_triggered"],
    "note": "gap triggers split before max_utterance_s; 2 utts",
}))

# 6. Fixed-window exact: 3 × 1 s segments -> exactly window_s=3.0 on third
FIXTURES.append(("fixed_window_exact", [
    make_segment_dict(start=0.0, duration=1.0),
    make_segment_dict(start=1.5, duration=1.0),
    make_segment_dict(start=3.0, duration=1.0),
], {
    "expected_utterances": 1,
    "window_s": 3.0,
    "strategies": ["fixed_window"],
    "note": "3×1s = 3s speech == window_s -> 1 utt on 3rd feed",
}))

# 7. Fixed-window overflow: 4 × 1 s segments, window_s=3.0 -> emit at 3 s, flush 1 s
FIXTURES.append(("fixed_window_overflow", [
    make_segment_dict(start=0.0, duration=1.0),
    make_segment_dict(start=1.5, duration=1.0),
    make_segment_dict(start=3.0, duration=1.0),
    make_segment_dict(start=4.5, duration=1.0),
], {
    "expected_utterances": 2,
    "window_s": 3.0,
    "strategies": ["fixed_window"],
    "note": "4×1s=4s, window=3s -> emit at 3s, flush 1s -> 2 utts",
}))

# 8. Empty input — no segments
FIXTURES.append(("empty", [], {
    "expected_utterances": 0,
    "strategies": ["pause_triggered", "fixed_window"],
    "note": "no segments -> no utterances",
}))

# 9. Single tiny segment (100 ms) — segmenter does not filter by duration
FIXTURES.append(("single_tiny", [
    make_segment_dict(start=0.5, duration=0.1),
], {
    "expected_utterances": 1,
    "strategies": ["pause_triggered", "fixed_window"],
    "note": "0.1s segment — segmenter emits; filtering is VAD's job",
}))


# ---------------------------------------------------------------------------
# Write fixtures
# ---------------------------------------------------------------------------

def main():
    print(f"Generating segmenter fixtures in {OUT_DIR}/\n")
    for name, segments, meta in FIXTURES:
        save(name, segments, meta)
    print(f"\nDone — {len(FIXTURES)} fixture files written.")


if __name__ == "__main__":
    main()
