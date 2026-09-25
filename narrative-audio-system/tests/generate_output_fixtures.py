"""
Generate synthetic fixtures for Step 6 — Output Generator tests.

Produces:
  tests/fixtures/output/
    mock_pairs.json          — list of (transcript, emotion) pair dicts
    srt_expected.srt         — expected SRT output for the mock pairs
    caption_lines.json       — expected CaptionLine.to_dict() for each pair
    atmosphere_schedules.json — expected CrossfadeSchedule.to_dict() for each pair

Run:
    python tests/generate_output_fixtures.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "output"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Mock (transcript, emotion) pairs
# ---------------------------------------------------------------------------

MOCK_PAIRS = [
    {
        "transcript": {
            "text":       "The forest was quiet that night.",
            "start":      0.0,
            "end":        2.5,
            "latency_ms": 180.0,
            "backend":    "faster-whisper",
            "confidence": 0.95,
            "language":   "en",
        },
        "emotion": {
            "label":      "calm",
            "confidence": 0.88,
            "start":      0.0,
            "end":        2.5,
            "latency_ms": 12.0,
            "backend":    "mfcc-mlp",
            "all_scores": {"calm": 0.88, "neutral": 0.08, "happy": 0.04},
        },
    },
    {
        "transcript": {
            "text":       "Until the branch snapped.",
            "start":      3.1,
            "end":        4.8,
            "latency_ms": 190.0,
            "backend":    "faster-whisper",
            "confidence": 0.91,
            "language":   "en",
        },
        "emotion": {
            "label":      "tense",
            "confidence": 0.79,
            "start":      3.1,
            "end":        4.8,
            "latency_ms": 11.5,
            "backend":    "mfcc-mlp",
            "all_scores": {"tense": 0.79, "fearful": 0.15, "neutral": 0.06},
        },
    },
    {
        "transcript": {
            "text":       "She ran without looking back.",
            "start":      5.2,
            "end":        7.1,
            "latency_ms": 185.0,
            "backend":    "faster-whisper",
            "confidence": 0.93,
            "language":   "en",
        },
        "emotion": {
            "label":      "fearful",
            "confidence": 0.82,
            "start":      5.2,
            "end":        7.1,
            "latency_ms": 13.0,
            "backend":    "mfcc-mlp",
            "all_scores": {"fearful": 0.82, "tense": 0.12, "angry": 0.06},
        },
    },
    {
        "transcript": {
            "text":       "The morning light brought relief.",
            "start":      15.0,
            "end":        17.2,
            "latency_ms": 175.0,
            "backend":    "faster-whisper",
            "confidence": 0.97,
            "language":   "en",
        },
        "emotion": {
            "label":      "happy",
            "confidence": 0.76,
            "start":      15.0,
            "end":        17.2,
            "latency_ms": 10.5,
            "backend":    "mfcc-mlp",
            "all_scores": {"happy": 0.76, "calm": 0.18, "neutral": 0.06},
        },
    },
    # Duplicate emotion within cooldown — should NOT produce a new atmosphere
    {
        "transcript": {
            "text":       "She smiled at the sunrise.",
            "start":      17.5,
            "end":        19.0,
            "latency_ms": 170.0,
            "backend":    "faster-whisper",
            "confidence": 0.94,
            "language":   "en",
        },
        "emotion": {
            "label":      "happy",
            "confidence": 0.81,
            "start":      17.5,
            "end":        19.0,
            "latency_ms": 11.0,
            "backend":    "mfcc-mlp",
            "all_scores": {"happy": 0.81, "calm": 0.14, "neutral": 0.05},
        },
    },
    # Very short transcript — should be filtered out by CaptionFormatter
    {
        "transcript": {
            "text":       ".",
            "start":      20.0,
            "end":        20.3,
            "latency_ms": 155.0,
            "backend":    "faster-whisper",
            "confidence": 0.40,
            "language":   "en",
        },
        "emotion": {
            "label":      "neutral",
            "confidence": 0.60,
            "start":      20.0,
            "end":        20.3,
            "latency_ms": 9.0,
            "backend":    "mfcc-mlp",
            "all_scores": {"neutral": 0.60, "calm": 0.30, "happy": 0.10},
        },
    },
]

# ---------------------------------------------------------------------------
# Expected CaptionLine dicts (skip the too-short entry at index 5)
# ---------------------------------------------------------------------------

TONE_COLOURS = {
    "neutral":   "#a8b4c0",
    "calm":      "#7ec8a0",
    "happy":     "#f7c948",
    "sad":       "#6b9bcf",
    "angry":     "#e05c5c",
    "fearful":   "#c07ecf",
    "disgust":   "#8a9e6b",
    "surprised": "#f0a045",
    "tense":     "#e07a5f",
    "unknown":   "#ffffff",
}

def _tone_colour(label):
    return TONE_COLOURS.get(label.lower(), TONE_COLOURS["unknown"])


def _ts(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


caption_lines = []
srt_lines = []
caption_idx = 0

for pair in MOCK_PAIRS:
    tr = pair["transcript"]
    em = pair["emotion"]
    if len(tr["text"].strip()) < 2:
        continue
    caption_idx += 1
    cl = {
        "type":       "caption",
        "index":      caption_idx,
        "label":      em["label"],
        "text":       tr["text"].strip(),
        "start":      round(tr["start"], 3),
        "end":        round(tr["end"], 3),
        "confidence": round(em["confidence"], 3),
        "color":      _tone_colour(em["label"]),
    }
    caption_lines.append(cl)

    srt_block = (
        f"{caption_idx}\n"
        f"{_ts(tr['start'])} --> {_ts(tr['end'])}\n"
        f"[{em['label']}] {tr['text'].strip()}\n"
    )
    srt_lines.append(srt_block)


# ---------------------------------------------------------------------------
# Expected atmosphere schedules
# (cooldown logic: pairs[1] tense, pairs[2] fearful trigger; pairs[4] happy
#  immediately follows pairs[3] happy and is within cooldown_s=8s — suppressed)
# ---------------------------------------------------------------------------

TONE_QUERIES = {
    "calm":      "calm gentle wind soft water ambient",
    "neutral":   "neutral quiet indoor room tone",
    "happy":     "upbeat bright cheerful outdoor birds",
    "sad":       "sad melancholy quiet rain distant",
    "angry":     "urgent high-energy dramatic intense",
    "fearful":   "tense dark forest night ambient",
    "tense":     "tense forest night ambience suspense",
    "disgust":   "dark low rumble ominous underground",
    "surprised": "sudden bright stab high-energy reveal",
}

FALLBACK_AMBIENT = {
    "calm":      {"description": "gentle wind, soft water",        "energy": "low"},
    "neutral":   {"description": "quiet room tone, light hum",     "energy": "low"},
    "happy":     {"description": "birdsong, light breeze",         "energy": "medium"},
    "sad":       {"description": "distant rain, sparse piano",     "energy": "low"},
    "angry":     {"description": "driving percussion, wind",       "energy": "high"},
    "fearful":   {"description": "dark forest, distant owl, creak","energy": "medium"},
    "tense":     {"description": "tense forest night, branch snap","energy": "medium"},
    "disgust":   {"description": "low rumble, dripping, echo",     "energy": "low"},
    "surprised": {"description": "bright orchestral stab, rush",   "energy": "high"},
}

# Pairs that trigger an atmosphere change (label changes OR first time):
# pairs[0]: calm   (first)  -> triggers
# pairs[1]: tense           -> triggers (label changed)
# pairs[2]: fearful         -> triggers (label changed)
# pairs[3]: happy           -> triggers (label changed)
# pairs[4]: happy           -> suppressed (same label, within ~2s of pairs[3] which is << 8s cooldown)
# pairs[5]: neutral (.)     -> caption filtered, atmosphere still evaluated — label changed but text filtered

atmosphere_schedules = []
last_label = None

# Note: in reality cooldown is time-based; for the fixture we just record which
# ones WOULD trigger assuming the test processes them quickly (within 8s each).
# The fixture documents the expected output, not the timing.

for pair in MOCK_PAIRS:
    em = pair["emotion"]
    label = em["label"].lower()
    query = TONE_QUERIES.get(label, f"{label} ambient atmosphere")
    fb    = FALLBACK_AMBIENT.get(label, {"description": f"{label} ambient", "energy": "low"})
    sched = {
        "type":                  "atmosphere",
        "emotion_label":         label,
        "query":                 query,
        "suggested_clip":        fb["description"],
        "suggested_description": fb["description"],
        "fade_in_s":             2.0,
        "lag_s":                 6.0,
    }
    atmosphere_schedules.append({
        "pair_index": MOCK_PAIRS.index(pair),
        "label":      label,
        "suppressed": (label == last_label),   # simplified: time-based cooldown not modelled here
        "schedule":   sched if label != last_label else None,
    })
    last_label = label


# ---------------------------------------------------------------------------
# Write fixtures
# ---------------------------------------------------------------------------

pairs_path = FIXTURE_DIR / "mock_pairs.json"
pairs_path.write_text(json.dumps(MOCK_PAIRS, indent=2), encoding="utf-8")
print(f"[Output fixtures] Wrote {pairs_path}  ({len(MOCK_PAIRS)} pairs)")

captions_path = FIXTURE_DIR / "caption_lines.json"
captions_path.write_text(json.dumps(caption_lines, indent=2), encoding="utf-8")
print(f"[Output fixtures] Wrote {captions_path}  ({len(caption_lines)} captions)")

srt_path = FIXTURE_DIR / "srt_expected.srt"
srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
print(f"[Output fixtures] Wrote {srt_path}")

atm_path = FIXTURE_DIR / "atmosphere_schedules.json"
atm_path.write_text(json.dumps(atmosphere_schedules, indent=2), encoding="utf-8")
print(f"[Output fixtures] Wrote {atm_path}  ({len(atmosphere_schedules)} entries)")

print("\n[Output fixtures] Done.")
