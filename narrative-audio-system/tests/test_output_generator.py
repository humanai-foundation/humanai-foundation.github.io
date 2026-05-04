"""
Tests for Step 6 — Output Generator
=====================================
Tests cover:
  Track A (captions.py)
    - CaptionLine: rendering, SRT formatting, to_dict schema
    - CaptionFormatter: short-text filtering, index auto-increment
    - SRTWriter: file creation, correct block format
    - CaptionBroadcaster: async start/stop, broadcast to mock client

  Track B (atmosphere.py)
    - AtmosphereMapper: known + unknown labels
    - CrossfadeSchedule: to_dict schema
    - CrossfadeScheduler: first-time trigger, cooldown suppression, label-change trigger
    - RetrievalBridge: graceful fallback when CSV absent

  Combined (output_generator.py)
    - OutputGenerator.process: returns (CaptionLine, CrossfadeSchedule) tuple
    - OutputGenerator.process_all: batch processing
    - OutputGenerator.summary: contains captions + atmosphere entries
    - OutputGenerator parallel dispatch: both tracks run concurrently

Run:
    python tests/test_output_generator.py
    python tests/test_output_generator.py --verbose
    python tests/test_output_generator.py --skip-ws   # skip WebSocket tests
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "output"

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

for mod_dir in ("output_generator",):
    p = str(ROOT / mod_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Lightweight mock result objects (mimic Step 4 + Step 5 outputs)
# ---------------------------------------------------------------------------

@dataclass
class _MockTranscript:
    text:       str
    start:      float
    end:        float
    latency_ms: float = 180.0
    backend:    str   = "faster-whisper"
    confidence: float = 0.90
    language:   str   = "en"


@dataclass
class _MockEmotion:
    label:      str
    confidence: float
    start:      float
    end:        float
    latency_ms: float = 12.0
    backend:    str   = "mfcc-mlp"
    all_scores: dict  = None

    def __post_init__(self):
        if self.all_scores is None:
            self.all_scores = {self.label: self.confidence}


def _load_pairs():
    """Load mock_pairs.json and build _MockTranscript / _MockEmotion objects."""
    with open(FIXTURE_DIR / "mock_pairs.json", encoding="utf-8") as f:
        raw = json.load(f)
    pairs = []
    for item in raw:
        tr = _MockTranscript(**item["transcript"])
        em = _MockEmotion(**item["emotion"])
        pairs.append((tr, em))
    return pairs


# ===========================================================================
# Track A — captions.py
# ===========================================================================

class TestCaptionLine(unittest.TestCase):

    def setUp(self):
        from captions import CaptionLine
        self.CaptionLine = CaptionLine

    def _make(self, label="calm", text="The forest was quiet.", start=0.0, end=2.5,
              confidence=0.88, color="#7ec8a0", index=1):
        return self.CaptionLine(label=label, text=text, start=start, end=end,
                                confidence=confidence, color=color, index=index)

    def test_render_contains_label(self):
        line = self._make(label="tense", text="Until the branch snapped.")
        rendered = line.render()
        self.assertIn("[tense]", rendered)
        self.assertIn("Until the branch snapped.", rendered)

    def test_render_label_padded(self):
        line = self._make(label="calm")
        rendered = line.render(width=10)
        # Tag should be left-justified within width
        self.assertTrue(rendered.startswith("[calm]"))

    def test_duration_property(self):
        line = self._make(start=1.0, end=3.5)
        self.assertAlmostEqual(line.duration, 2.5, places=5)

    def test_to_dict_has_required_keys(self):
        line = self._make()
        d = line.to_dict()
        for key in ("type", "index", "label", "text", "start", "end",
                    "confidence", "color"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_to_dict_type_is_caption(self):
        self.assertEqual(self._make().to_dict()["type"], "caption")

    def test_to_dict_values(self):
        line = self._make(label="calm", text="Hello.", start=1.0, end=2.0,
                          confidence=0.88, color="#7ec8a0", index=3)
        d = line.to_dict()
        self.assertEqual(d["label"], "calm")
        self.assertEqual(d["text"], "Hello.")
        self.assertAlmostEqual(d["start"], 1.0, places=2)
        self.assertAlmostEqual(d["end"], 2.0, places=2)
        self.assertEqual(d["color"], "#7ec8a0")
        self.assertEqual(d["index"], 3)

    def test_to_srt_block_format(self):
        line = self._make(start=0.0, end=2.5, index=1, label="calm",
                          text="The forest was quiet.")
        block = line.to_srt_block()
        lines = block.strip().splitlines()
        self.assertEqual(lines[0], "1")                       # sequence number
        self.assertIn("-->", lines[1])                        # timestamp line
        self.assertIn(",", lines[1])                          # SRT uses comma decimal
        self.assertIn("[calm]", lines[2])                     # label in subtitle
        self.assertIn("The forest was quiet.", lines[2])      # text

    def test_srt_timestamp_zero(self):
        line = self._make(start=0.0, end=2.5)
        block = line.to_srt_block()
        self.assertIn("00:00:00,000 --> 00:00:02,500", block)

    def test_srt_timestamp_nonzero(self):
        line = self._make(start=62.3, end=65.1)
        block = line.to_srt_block()
        self.assertIn("00:01:02,300 --> 00:01:05,100", block)


class TestCaptionFormatter(unittest.TestCase):

    def setUp(self):
        from captions import CaptionFormatter
        self.fmt = CaptionFormatter(min_text_length=2, verbose=False)

    def _pair(self, text="Hello world.", label="calm", start=0.0, end=2.0,
              confidence=0.9):
        tr = _MockTranscript(text=text, start=start, end=end)
        em = _MockEmotion(label=label, confidence=confidence, start=start, end=end)
        return tr, em

    def test_format_returns_caption_line(self):
        from captions import CaptionLine
        tr, em = self._pair()
        line = self.fmt.format(tr, em)
        self.assertIsInstance(line, CaptionLine)

    def test_format_short_text_returns_none(self):
        tr, em = self._pair(text=".")
        self.assertIsNone(self.fmt.format(tr, em))

    def test_format_empty_text_returns_none(self):
        tr, em = self._pair(text="")
        self.assertIsNone(self.fmt.format(tr, em))

    def test_format_strips_whitespace(self):
        tr, em = self._pair(text="  Hello world.  ")
        line = self.fmt.format(tr, em)
        self.assertEqual(line.text, "Hello world.")

    def test_index_auto_increments(self):
        tr1, em1 = self._pair(text="First sentence.", start=0.0, end=1.0)
        tr2, em2 = self._pair(text="Second sentence.", start=1.5, end=3.0)
        line1 = self.fmt.format(tr1, em1)
        line2 = self.fmt.format(tr2, em2)
        self.assertEqual(line1.index, 1)
        self.assertEqual(line2.index, 2)

    def test_index_skips_filtered(self):
        tr_short, em_short = self._pair(text=".")
        tr_ok, em_ok = self._pair(text="Valid caption here.")
        self.fmt.format(tr_short, em_short)   # filtered
        line = self.fmt.format(tr_ok, em_ok)
        # Index is still 1 because filtered entries don't increment
        self.assertEqual(line.index, 1)

    def test_reset_index(self):
        tr, em = self._pair(text="Hello world.")
        self.fmt.format(tr, em)
        self.fmt.format(tr, em)
        self.fmt.reset_index()
        line = self.fmt.format(tr, em)
        self.assertEqual(line.index, 1)

    def test_colour_assigned(self):
        tr, em = self._pair(label="tense")
        line = self.fmt.format(tr, em)
        self.assertEqual(line.color, "#e07a5f")

    def test_colour_unknown_label(self):
        tr, em = self._pair(label="mystery_emotion")
        line = self.fmt.format(tr, em)
        self.assertEqual(line.color, "#ffffff")


class TestSRTWriter(unittest.TestCase):

    def setUp(self):
        from captions import CaptionLine, SRTWriter
        self.CaptionLine = CaptionLine
        self.SRTWriter = SRTWriter

    def _make_line(self, idx=1, label="calm", text="Hello.", start=0.0, end=2.0):
        return self.CaptionLine(label=label, text=text, start=start, end=end,
                                confidence=0.9, color="#7ec8a0", index=idx)

    def test_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name
        writer = self.SRTWriter(path, verbose=False)
        writer.write(self._make_line())
        self.assertTrue(Path(path).exists())
        Path(path).unlink(missing_ok=True)

    def test_write_content(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False,
                                         mode="w") as f:
            path = f.name
        writer = self.SRTWriter(path, verbose=False)
        line = self._make_line(label="tense", text="Branch snapped.")
        writer.write(line)
        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("1", content)
        self.assertIn("[tense]", content)
        self.assertIn("Branch snapped.", content)
        Path(path).unlink(missing_ok=True)

    def test_write_multiple_blocks(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False,
                                         mode="w") as f:
            path = f.name
        writer = self.SRTWriter(path, verbose=False)
        writer.write(self._make_line(idx=1, text="First."))
        writer.write(self._make_line(idx=2, text="Second.", start=3.0, end=5.0))
        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("First.", content)
        self.assertIn("Second.", content)
        self.assertEqual(writer.count, 2)
        Path(path).unlink(missing_ok=True)

    def test_clears_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write("OLD CONTENT\n")
            path = f.name
        writer = self.SRTWriter(path, verbose=False)
        content = Path(path).read_text(encoding="utf-8")
        self.assertNotIn("OLD CONTENT", content)
        Path(path).unlink(missing_ok=True)

    def test_write_all(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False,
                                         mode="w") as f:
            path = f.name
        writer = self.SRTWriter(path, verbose=False)
        lines = [
            self._make_line(idx=1, text="First sentence."),
            None,   # should be skipped
            self._make_line(idx=2, text="Third sentence.", start=4.0, end=6.0),
        ]
        writer.write_all(lines)
        self.assertEqual(writer.count, 2)
        Path(path).unlink(missing_ok=True)

    def test_fixture_srt_matches_expected(self):
        """Verify generator produced valid SRT blocks."""
        srt_file = FIXTURE_DIR / "srt_expected.srt"
        if not srt_file.exists():
            self.skipTest("Fixture not found — run generate_output_fixtures.py")
        content = srt_file.read_text(encoding="utf-8")
        # At least 3 proper timestamp lines
        self.assertGreaterEqual(content.count("-->"), 3)


class TestCaptionBroadcaster(unittest.IsolatedAsyncioTestCase):

    async def test_start_and_stop(self):
        """CaptionBroadcaster starts and stops without error."""
        try:
            from captions import CaptionBroadcaster
        except ImportError:
            self.skipTest("websockets not installed")
        broadcaster = CaptionBroadcaster(host="localhost", port=18765, verbose=False)
        await broadcaster.start()
        await broadcaster.stop()

    async def test_broadcast_no_clients(self):
        """Broadcast with no clients connected is a no-op."""
        try:
            from captions import CaptionBroadcaster
        except ImportError:
            self.skipTest("websockets not installed")
        broadcaster = CaptionBroadcaster(host="localhost", port=18766, verbose=False)
        await broadcaster.start()
        # Should not raise
        await broadcaster.broadcast({"type": "caption", "label": "calm", "text": "test"})
        await broadcaster.stop()

    async def test_connected_clients_count(self):
        """connected_clients starts at 0."""
        try:
            from captions import CaptionBroadcaster
        except ImportError:
            self.skipTest("websockets not installed")
        broadcaster = CaptionBroadcaster(host="localhost", port=18767, verbose=False)
        await broadcaster.start()
        self.assertEqual(broadcaster.connected_clients, 0)
        await broadcaster.stop()


# ===========================================================================
# Track B — atmosphere.py
# ===========================================================================

class TestAtmosphereMapper(unittest.TestCase):

    def setUp(self):
        from atmosphere import AtmosphereMapper
        self.mapper = AtmosphereMapper(verbose=False)

    def test_known_label_calm(self):
        q = self.mapper.query_for("calm")
        self.assertIn("calm", q.lower())

    def test_known_label_tense(self):
        q = self.mapper.query_for("tense")
        self.assertIn("tense", q.lower())

    def test_unknown_label_fallback(self):
        q = self.mapper.query_for("unicorn_emotion")
        self.assertIn("unicorn_emotion", q.lower())

    def test_case_insensitive(self):
        q1 = self.mapper.query_for("CALM")
        q2 = self.mapper.query_for("calm")
        self.assertEqual(q1, q2)

    def test_fallback_for_known(self):
        fb = self.mapper.fallback_for("sad")
        self.assertIn("description", fb)
        self.assertIn("energy", fb)

    def test_fallback_for_unknown(self):
        fb = self.mapper.fallback_for("mystery")
        self.assertIn("description", fb)

    def test_custom_queries_override(self):
        from atmosphere import AtmosphereMapper
        mapper = AtmosphereMapper(custom_queries={"calm": "custom calm override"}, verbose=False)
        self.assertEqual(mapper.query_for("calm"), "custom calm override")

    def test_all_labels_have_queries(self):
        from atmosphere import TONE_QUERIES
        for label in ("calm", "neutral", "happy", "sad", "angry",
                      "fearful", "tense", "disgust", "surprised"):
            q = self.mapper.query_for(label)
            self.assertIsInstance(q, str)
            self.assertGreater(len(q), 0)


class TestCrossfadeSchedule(unittest.TestCase):

    def setUp(self):
        from atmosphere import CrossfadeSchedule
        self.CrossfadeSchedule = CrossfadeSchedule

    def _make(self, label="calm", query="calm ambient", clip="gentle wind",
              desc="gentle wind, soft water", fade=2.0, lag=6.0):
        return self.CrossfadeSchedule(
            emotion_label=label, query=query,
            suggested_clip=clip, suggested_description=desc,
            fade_in_s=fade, lag_s=lag,
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("type", "emotion_label", "query", "suggested_clip",
                    "suggested_description", "fade_in_s", "lag_s"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_to_dict_type_is_atmosphere(self):
        self.assertEqual(self._make().to_dict()["type"], "atmosphere")

    def test_to_dict_values(self):
        d = self._make(label="tense", fade=3.0, lag=5.0).to_dict()
        self.assertEqual(d["emotion_label"], "tense")
        self.assertAlmostEqual(d["fade_in_s"], 3.0)
        self.assertAlmostEqual(d["lag_s"], 5.0)

    def test_repr(self):
        s = repr(self._make(label="fearful"))
        self.assertIn("fearful", s)

    def test_scheduled_at_is_float(self):
        sched = self._make()
        self.assertIsInstance(sched.scheduled_at, float)
        self.assertGreater(sched.scheduled_at, 0.0)


class TestRetrievalBridge(unittest.TestCase):

    def test_graceful_fallback_missing_csv(self):
        from atmosphere import RetrievalBridge
        bridge = RetrievalBridge(features_csv="/nonexistent/path.csv", verbose=False)
        self.assertFalse(bridge.available)
        results = bridge.search("calm ambient")
        self.assertEqual(results, [])

    def test_returns_empty_list_on_fallback(self):
        from atmosphere import RetrievalBridge
        bridge = RetrievalBridge(features_csv=None, verbose=False)
        results = bridge.search("tense forest")
        self.assertIsInstance(results, list)


class TestCrossfadeScheduler(unittest.TestCase):

    def setUp(self):
        from atmosphere import CrossfadeScheduler
        # Short cooldown so we can test both suppress and trigger quickly
        self.scheduler = CrossfadeScheduler(lag_s=6.0, fade_s=2.0,
                                            cooldown_s=0.1, verbose=False)

    def _emotion(self, label, start=0.0, end=2.0):
        return _MockEmotion(label=label, confidence=0.8, start=start, end=end)

    def test_first_call_returns_schedule(self):
        from atmosphere import CrossfadeSchedule
        result = self.scheduler.schedule(self._emotion("calm"))
        self.assertIsInstance(result, CrossfadeSchedule)

    def test_same_label_within_cooldown_suppressed(self):
        # Use a scheduler with long cooldown
        from atmosphere import CrossfadeScheduler
        sched = CrossfadeScheduler(lag_s=6.0, fade_s=2.0, cooldown_s=60.0, verbose=False)
        sched.schedule(self._emotion("calm"))
        result = sched.schedule(self._emotion("calm"))
        self.assertIsNone(result)

    def test_different_label_triggers_even_in_cooldown(self):
        from atmosphere import CrossfadeScheduler, CrossfadeSchedule
        sched = CrossfadeScheduler(lag_s=6.0, fade_s=2.0, cooldown_s=60.0, verbose=False)
        sched.schedule(self._emotion("calm"))
        result = sched.schedule(self._emotion("tense"))
        self.assertIsInstance(result, CrossfadeSchedule)

    def test_same_label_after_cooldown_triggers(self):
        from atmosphere import CrossfadeScheduler, CrossfadeSchedule
        sched = CrossfadeScheduler(lag_s=6.0, fade_s=2.0, cooldown_s=0.05, verbose=False)
        sched.schedule(self._emotion("calm"))
        time.sleep(0.1)  # wait for cooldown
        result = sched.schedule(self._emotion("calm"))
        self.assertIsInstance(result, CrossfadeSchedule)

    def test_schedule_label_stored(self):
        from atmosphere import CrossfadeSchedule
        result = self.scheduler.schedule(self._emotion("angry"))
        self.assertEqual(result.emotion_label, "angry")

    def test_schedule_fade_and_lag(self):
        result = self.scheduler.schedule(self._emotion("sad"))
        self.assertAlmostEqual(result.fade_in_s, 2.0)
        self.assertAlmostEqual(result.lag_s, 6.0)

    def test_sequence_calm_tense_fearful(self):
        from atmosphere import CrossfadeSchedule
        r1 = self.scheduler.schedule(self._emotion("calm"))
        time.sleep(0.15)
        r2 = self.scheduler.schedule(self._emotion("tense"))
        time.sleep(0.15)
        r3 = self.scheduler.schedule(self._emotion("fearful"))
        self.assertIsInstance(r1, CrossfadeSchedule)
        self.assertIsInstance(r2, CrossfadeSchedule)
        self.assertIsInstance(r3, CrossfadeSchedule)
        self.assertEqual(r1.emotion_label, "calm")
        self.assertEqual(r2.emotion_label, "tense")
        self.assertEqual(r3.emotion_label, "fearful")


# ===========================================================================
# Combined — output_generator.py
# ===========================================================================

class TestOutputGenerator(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, str(ROOT / "output_generator"))
        from output_generator import OutputGenerator
        self.OutputGenerator = OutputGenerator

    def _gen(self, srt_path=None, cooldown_s=0.05):
        return self.OutputGenerator(
            srt_path=srt_path,
            enable_websocket=False,
            cooldown_s=cooldown_s,
            verbose=False,
        )

    def _pair(self, text="The forest was quiet.", label="calm",
              start=0.0, end=2.5, confidence=0.88):
        tr = _MockTranscript(text=text, start=start, end=end)
        em = _MockEmotion(label=label, confidence=confidence, start=start, end=end)
        return tr, em

    # --- process() return type ---

    def test_process_returns_tuple(self):
        gen = self._gen()
        tr, em = self._pair()
        result = gen.process(tr, em)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_process_caption_type(self):
        from captions import CaptionLine
        gen = self._gen()
        tr, em = self._pair()
        caption, _ = gen.process(tr, em)
        self.assertIsInstance(caption, CaptionLine)

    def test_process_atmosphere_type(self):
        from atmosphere import CrossfadeSchedule
        gen = self._gen()
        tr, em = self._pair()
        _, schedule = gen.process(tr, em)
        self.assertIsInstance(schedule, CrossfadeSchedule)

    def test_process_short_text_caption_is_none(self):
        gen = self._gen()
        tr, em = self._pair(text=".")
        caption, _ = gen.process(tr, em)
        self.assertIsNone(caption)

    # --- captions property ---

    def test_captions_accumulated(self):
        gen = self._gen(cooldown_s=0.0)
        pairs = [
            self._pair(text="First sentence.", label="calm", start=0.0, end=2.0),
            self._pair(text="Second sentence.", label="tense", start=3.0, end=5.0),
        ]
        for tr, em in pairs:
            gen.process(tr, em)
        self.assertEqual(len(gen.captions), 2)

    def test_captions_property_is_copy(self):
        gen = self._gen()
        gen.process(*self._pair(text="Hello world."))
        caps = gen.captions
        caps.clear()
        self.assertEqual(len(gen.captions), 1)

    # --- atmosphere_log property ---

    def test_atmosphere_log_accumulated(self):
        gen = self._gen(cooldown_s=0.0)
        for label in ("calm", "tense", "fearful"):
            tr, em = self._pair(text="Some text here.", label=label)
            gen.process(tr, em)
            time.sleep(0.06)
        self.assertGreaterEqual(len(gen.atmosphere_log), 1)

    # --- process_all() ---

    def test_process_all_returns_list(self):
        gen = self._gen(cooldown_s=0.0)
        pairs = [
            self._pair(text="Sentence one.", label="calm", start=0.0, end=2.0),
            self._pair(text="Sentence two.", label="tense", start=3.0, end=5.0),
        ]
        results = gen.process_all(pairs)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_process_all_each_result_is_tuple(self):
        gen = self._gen(cooldown_s=0.0)
        pairs = [self._pair(text=f"Sentence {i}.", label="calm",
                             start=float(i * 3), end=float(i * 3 + 2))
                 for i in range(3)]
        for res in gen.process_all(pairs):
            self.assertIsInstance(res, tuple)
            self.assertEqual(len(res), 2)

    # --- SRT integration ---

    def test_srt_file_written(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name
        gen = self._gen(srt_path=path)
        gen.process(*self._pair(text="The forest was quiet that night."))
        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("forest was quiet", content)
        Path(path).unlink(missing_ok=True)

    # --- summary() ---

    def test_summary_contains_counts(self):
        gen = self._gen(cooldown_s=0.0)
        gen.process(*self._pair(text="One sentence."))
        summary = gen.summary()
        self.assertIn("Captions generated", summary)
        self.assertIn("Atmosphere changes", summary)

    def test_summary_contains_transcript(self):
        gen = self._gen()
        gen.process(*self._pair(text="The forest was quiet that night."))
        summary = gen.summary()
        self.assertIn("forest was quiet", summary)

    def test_summary_atmosphere_log(self):
        gen = self._gen(cooldown_s=0.0)
        gen.process(*self._pair(text="One sentence.", label="tense"))
        summary = gen.summary()
        self.assertIn("Atmosphere log", summary)

    # --- fixture-based integration ---

    def test_fixture_pairs_processed(self):
        """Process all mock_pairs from fixture and check counts."""
        if not (FIXTURE_DIR / "mock_pairs.json").exists():
            self.skipTest("Fixtures not generated — run generate_output_fixtures.py")
        pairs = _load_pairs()
        gen = self._gen(cooldown_s=0.05)
        results = gen.process_all(pairs)
        self.assertEqual(len(results), len(pairs))

    def test_fixture_caption_count(self):
        """Expect 5 captions (6 pairs minus 1 filtered short text)."""
        if not (FIXTURE_DIR / "mock_pairs.json").exists():
            self.skipTest("Fixtures not generated")
        pairs = _load_pairs()
        gen = self._gen(cooldown_s=0.0)
        gen.process_all(pairs)
        # 5 pairs have text >= 2 chars (one pair has "." which is filtered)
        self.assertEqual(len(gen.captions), 5)

    def test_fixture_caption_labels(self):
        """Caption labels should match emotion labels from fixture."""
        if not (FIXTURE_DIR / "mock_pairs.json").exists():
            self.skipTest("Fixtures not generated")
        pairs = _load_pairs()
        gen = self._gen(cooldown_s=0.0)
        gen.process_all(pairs)
        expected_labels = ["calm", "tense", "fearful", "happy", "happy"]
        for cap, exp in zip(gen.captions, expected_labels):
            self.assertEqual(cap.label, exp)


# ===========================================================================
# Parallel dispatch verification
# ===========================================================================

class TestParallelDispatch(unittest.TestCase):
    """Verify that Track A and Track B run concurrently (not sequentially)."""

    def test_parallel_is_faster_than_sequential(self):
        """
        Inject sleep delays into both tracks and confirm process() completes
        faster than sum-of-delays (proving concurrency).
        """
        from output_generator import OutputGenerator
        gen = OutputGenerator(enable_websocket=False, cooldown_s=0.0, verbose=False)

        DELAY = 0.12  # seconds per track

        original_a = gen._track_a
        original_b = gen._track_b

        def slow_track_a(*args, **kwargs):
            time.sleep(DELAY)
            return original_a(*args, **kwargs)

        def slow_track_b(*args, **kwargs):
            time.sleep(DELAY)
            return original_b(*args, **kwargs)

        gen._track_a = slow_track_a
        gen._track_b = slow_track_b

        tr = _MockTranscript(text="Timing test sentence.", start=0.0, end=2.0)
        em = _MockEmotion(label="calm", confidence=0.9, start=0.0, end=2.0)

        t0 = time.perf_counter()
        gen.process(tr, em)
        elapsed = time.perf_counter() - t0

        # Sequential would take 2 * DELAY; parallel should be ~DELAY
        self.assertLess(elapsed, DELAY * 1.8,
                        f"process() took {elapsed:.3f}s, expected < {DELAY * 1.8:.3f}s "
                        f"(tracks don't appear to run in parallel)")


# ===========================================================================
# Fixture schema validation
# ===========================================================================

class TestFixtureSchemas(unittest.TestCase):

    def test_caption_lines_fixture_schema(self):
        path = FIXTURE_DIR / "caption_lines.json"
        if not path.exists():
            self.skipTest("Fixture not found")
        with open(path, encoding="utf-8") as f:
            lines = json.load(f)
        self.assertGreater(len(lines), 0)
        for cl in lines:
            for key in ("type", "index", "label", "text", "start", "end",
                        "confidence", "color"):
                self.assertIn(key, cl)
            self.assertEqual(cl["type"], "caption")

    def test_atmosphere_schedules_fixture_schema(self):
        path = FIXTURE_DIR / "atmosphere_schedules.json"
        if not path.exists():
            self.skipTest("Fixture not found")
        with open(path, encoding="utf-8") as f:
            schedules = json.load(f)
        self.assertGreater(len(schedules), 0)
        for entry in schedules:
            self.assertIn("pair_index", entry)
            self.assertIn("label", entry)
            self.assertIn("suppressed", entry)

    def test_atmosphere_fixture_first_is_not_suppressed(self):
        path = FIXTURE_DIR / "atmosphere_schedules.json"
        if not path.exists():
            self.skipTest("Fixture not found")
        with open(path, encoding="utf-8") as f:
            schedules = json.load(f)
        self.assertFalse(schedules[0]["suppressed"],
                         "First atmosphere entry should never be suppressed")

    def test_mock_pairs_has_six_entries(self):
        path = FIXTURE_DIR / "mock_pairs.json"
        if not path.exists():
            self.skipTest("Fixture not found")
        with open(path, encoding="utf-8") as f:
            pairs = json.load(f)
        self.assertEqual(len(pairs), 6)


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="Step 6 output generator tests")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-ws", action="store_true",
                        help="Skip WebSocket broadcaster tests")
    return parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = _parse_args()

    if args.skip_ws:
        # Remove WebSocket test class
        del TestCaptionBroadcaster

    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
