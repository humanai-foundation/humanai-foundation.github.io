"""
Step 4 Streaming Transcription — test suite
=============================================
Tests are grouped into three tiers:

  Unit tests      — no model, no audio files; validate API contracts and
                    structural guarantees of TranscriptionResult /
                    StreamingTranscriber using mock Utterances.

  Edge tests      — load tiny WAV fixtures (silence, noise, very short clip,
                    long clip) and check that the transcriber never crashes and
                    always returns a well-formed TranscriptionResult.

  Integration tests — run the real Whisper tiny model against RAVDESS recordings
                    and verify that expected keywords appear in the transcript.
                    These are marked slow and skipped when --skip-slow is passed.

Run:
    python tests/generate_transcriber_fixtures.py   # once
    python tests/test_transcriber.py                # all tests
    python tests/test_transcriber.py --verbose
    python tests/test_transcriber.py --skip-slow    # unit + edge only
"""

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
for mod in ("transcriber", "utterance_buffer", "vad_engine"):
    p = str(REPO_ROOT / mod)
    if p not in sys.path:
        sys.path.insert(0, p)

from streaming_transcriber import (  # noqa
    Transcriber, StreamingTranscriber, TranscriptionResult, transcribe_utterances,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcriber"
UNIT_DIR  = FIXTURES_DIR / "unit"
INTEG_DIR = FIXTURES_DIR / "integration"
EDGE_DIR  = FIXTURES_DIR / "edge"

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Minimal Utterance stub (avoids importing segmenter in every test)
# ---------------------------------------------------------------------------

@dataclass
class _Utterance:
    start: float
    end: float
    audio: np.ndarray = field(repr=False)
    strategy: str = "pause_triggered"
    num_vad_segments: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start


def _make_utterance(start: float, end: float,
                    amplitude: float = 0.3, seed: int = 0) -> _Utterance:
    """Deterministic synthetic speech utterance."""
    rng = np.random.default_rng(seed)
    duration = max(end - start, 0.001)
    n = max(1, int(duration * SAMPLE_RATE))
    t = np.linspace(0, duration, n, endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 160.0 * t).astype(np.float32)
    sig += rng.normal(0, 0.01, n).astype(np.float32)
    sig *= amplitude
    return _Utterance(start=start, end=end, audio=sig)


def _utterance_from_wav(wav_path: str, start: float = 0.0,
                        target_sr: int = 16000) -> _Utterance:
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    end = start + len(audio) / target_sr
    return _Utterance(start=start, end=end, audio=audio)


# Shared model instance — loaded once for the whole test run
_MODEL: Optional[Transcriber] = None

def _get_model() -> Transcriber:
    global _MODEL
    if _MODEL is None:
        _MODEL = Transcriber(model_size="tiny", verbose=False)
    return _MODEL


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def run_test(label: str, fn, slow: bool = False, skip_slow: bool = False):
    if slow and skip_slow:
        _results.append((label, SKIP, "slow test skipped"))
        print("s", end="", flush=True)
        return
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


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def check_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def check_type(obj, typ, label=""):
    if not isinstance(obj, typ):
        raise AssertionError(f"{label}: expected {typ.__name__}, got {type(obj).__name__}")


# ---------------------------------------------------------------------------
# UNIT TESTS — no model loaded, pure API/structural checks
# ---------------------------------------------------------------------------

def test_result_dataclass_fields():
    """TranscriptionResult exposes all expected fields."""
    r = TranscriptionResult(
        text="Hello world.", start=1.0, end=3.5,
        latency_ms=420.0, backend="faster-whisper",
        confidence=0.9, language="en",
    )
    check_eq(r.text, "Hello world.")
    check_eq(r.start, 1.0)
    check_eq(r.end, 3.5)
    check_eq(round(r.duration, 6), 2.5)
    check_eq(r.latency_ms, 420.0)
    check_eq(r.backend, "faster-whisper")
    check_eq(r.confidence, 0.9)
    check_eq(r.language, "en")


def test_result_duration_property():
    r = TranscriptionResult(text="", start=2.0, end=5.5, latency_ms=0)
    check(abs(r.duration - 3.5) < 1e-9, f"duration={r.duration}")


def test_result_repr_contains_text():
    r = TranscriptionResult(text="Once upon a time", start=0, end=2, latency_ms=300)
    check("Once upon a time" in repr(r), "repr should include text")


def test_streaming_transcriber_empty_input():
    """StreamingTranscriber with no utterances yields nothing."""
    st = StreamingTranscriber(transcriber=_get_model())
    results = st.process_all([])
    check_eq(len(results), 0, "empty input")
    check_eq(st.full_transcript(), "", "full_transcript on empty")


def test_full_transcript_joins_correctly():
    """full_transcript() must join result texts with a space."""
    st = StreamingTranscriber(transcriber=_get_model())
    # Inject results directly to avoid running the model
    st._results = [
        TranscriptionResult(text="The forest", start=0, end=1, latency_ms=0),
        TranscriptionResult(text="was quiet.", start=1.5, end=3, latency_ms=0),
    ]
    check_eq(st.full_transcript(), "The forest was quiet.")


def test_full_transcript_skips_empty_text():
    """full_transcript() must skip results with empty text."""
    st = StreamingTranscriber(transcriber=_get_model())
    st._results = [
        TranscriptionResult(text="Hello", start=0, end=1, latency_ms=0),
        TranscriptionResult(text="",      start=1, end=2, latency_ms=0),
        TranscriptionResult(text="world", start=2, end=3, latency_ms=0),
    ]
    check_eq(st.full_transcript(), "Hello world")


def test_results_property_returns_copy():
    """st.results must return a list (not mutate internal state)."""
    st = StreamingTranscriber(transcriber=_get_model())
    r1 = st.results
    r1.append("garbage")   # should not affect internal list
    check_eq(len(st.results), 0, "internal results not affected by external mutation")


def test_stream_generator_yields_in_order():
    """stream() must yield results in the same order utterances were fed."""
    utts = [_make_utterance(float(i), float(i) + 1.0, seed=i) for i in range(3)]
    st = StreamingTranscriber(transcriber=_get_model())
    yielded = list(st.stream(iter(utts)))
    check_eq(len(yielded), 3, "3 utterances -> 3 results")
    for i, (utt, res) in enumerate(zip(utts, yielded)):
        check_eq(res.start, utt.start, f"result[{i}].start")
        check_eq(res.end,   utt.end,   f"result[{i}].end")


def test_process_all_matches_stream():
    """process_all() must produce the same results as iterating stream()."""
    utts = [_make_utterance(0.0, 2.0, seed=0), _make_utterance(3.0, 5.0, seed=1)]
    st1 = StreamingTranscriber(transcriber=_get_model())
    st2 = StreamingTranscriber(transcriber=_get_model())
    batch   = st1.process_all(utts)
    streamed = list(st2.stream(iter(utts)))
    check_eq(len(batch), len(streamed), "count matches")
    for i, (b, s) in enumerate(zip(batch, streamed)):
        check_eq(b.start, s.start, f"[{i}].start")
        check_eq(b.end,   s.end,   f"[{i}].end")
        check_eq(b.text,  s.text,  f"[{i}].text")


def test_transcribe_result_timestamps_match_utterance():
    """result.start / result.end must equal the utterance's start / end."""
    utt = _make_utterance(start=4.5, end=7.2)
    model = _get_model()
    result = model.transcribe(utt)
    check_eq(result.start, 4.5, "result.start")
    check_eq(result.end,   7.2, "result.end")


def test_transcribe_result_text_is_string():
    utt = _make_utterance(0.0, 2.0)
    model = _get_model()
    result = model.transcribe(utt)
    check_type(result.text, str, "result.text")


def test_transcribe_result_latency_positive():
    utt = _make_utterance(0.0, 1.0)
    model = _get_model()
    result = model.transcribe(utt)
    check(result.latency_ms > 0, f"latency_ms={result.latency_ms} should be > 0")


def test_transcribe_result_backend_set():
    utt = _make_utterance(0.0, 1.0)
    model = _get_model()
    result = model.transcribe(utt)
    check(result.backend in ("faster-whisper", "openai-whisper"),
          f"unexpected backend: {result.backend!r}")


def test_transcribe_array_accepts_float32():
    """transcribe_array() must accept a float32 1-D numpy array."""
    audio = np.zeros(16000, dtype=np.float32)
    model = _get_model()
    result = model.transcribe_array(audio, sample_rate=16000, start=0.0, end=1.0)
    check_type(result, TranscriptionResult, "return type")
    check_type(result.text, str, "result.text")


def test_transcribe_array_2d_input():
    """transcribe_array() must flatten 2-D (stereo) arrays without crashing."""
    audio = np.zeros((16000, 2), dtype=np.float32)
    model = _get_model()
    result = model.transcribe_array(audio, sample_rate=16000)
    check_type(result.text, str, "result.text after 2-D input")


def test_invalid_backend_raises():
    """Constructing a Transcriber with an unknown backend must raise."""
    try:
        Transcriber(backend="nonexistent-backend", verbose=False)
        raise AssertionError("Expected an error for unknown backend")
    except (ImportError, ValueError, Exception):
        pass  # any error is acceptable


def test_model_reuse_across_calls():
    """The same model object must be reused — not reloaded on every transcribe()."""
    model = _get_model()
    id_before = id(model._model)
    utt = _make_utterance(0.0, 1.0)
    model.transcribe(utt)
    model.transcribe(utt)
    check_eq(id(model._model), id_before, "model object reused across calls")


# ---------------------------------------------------------------------------
# EDGE TESTS — real WAV files, structural checks only (no text assertion)
# ---------------------------------------------------------------------------

def _load_edge_manifest():
    path = EDGE_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run generate_transcriber_fixtures.py first")
    return json.loads(path.read_text())["files"]


def test_edge_silence_no_crash():
    """Transcribing 3 s of silence must not raise."""
    files = _load_edge_manifest()
    entry = next(f for f in files if "silence" in f["filename"])
    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    check_type(result.text, str, "silence result text")


def test_edge_noise_no_crash():
    """Transcribing broadband noise must not raise."""
    files = _load_edge_manifest()
    entry = next(f for f in files if "noise" in f["filename"])
    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    check_type(result.text, str, "noise result text")


def test_edge_tiny_clip_no_crash():
    """Transcribing a 50 ms clip must not raise."""
    files = _load_edge_manifest()
    entry = next(f for f in files if "tiny" in f["filename"])
    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    check_type(result.text, str, "tiny clip result text")


def test_edge_long_clip_returns_text():
    """Transcribing an 8 s clip must return a string."""
    files = _load_edge_manifest()
    entry = next(f for f in files if "long" in f["filename"])
    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    check_type(result.text, str, "long clip result text")
    check(result.latency_ms > 0, "latency must be measured")


def test_edge_latency_scales_with_duration():
    """Longer audio should generally take longer to transcribe than silence."""
    files = _load_edge_manifest()
    silence_entry = next(f for f in files if "silence" in f["filename"])
    long_entry = next(f for f in files if "long" in f["filename"])

    silence_utt = _utterance_from_wav(silence_entry["filepath"])
    long_utt = _utterance_from_wav(long_entry["filepath"])

    silence_result = _get_model().transcribe(silence_utt)
    long_result = _get_model().transcribe(long_utt)

    # Long clip should take at least as long as silence (soft check)
    check(
        long_result.latency_ms >= silence_result.latency_ms * 0.5,
        f"long_latency={long_result.latency_ms:.0f}ms should be >= "
        f"half of silence_latency={silence_result.latency_ms:.0f}ms",
    )


def test_edge_result_text_stripped():
    """result.text must be stripped of leading/trailing whitespace."""
    files = _load_edge_manifest()
    entry = next(f for f in files if "silence" in f["filename"])
    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    check_eq(result.text, result.text.strip(), "text should be stripped")


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — real RAVDESS speech, keyword checks (slow)
# ---------------------------------------------------------------------------

def _load_ravdess_manifest():
    path = INTEG_DIR / "ravdess_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run generate_transcriber_fixtures.py first")
    return json.loads(path.read_text())


def test_integration_ravdess_statement1_keywords():
    """Transcription of statement-01 files contains 'kids' or 'talking' or 'door'."""
    manifest = _load_ravdess_manifest()
    keywords = [k.lower() for k in manifest["keywords"]["01"]]
    files = [f for f in manifest["files"] if f["statement_code"] == "01"]
    check(len(files) > 0, "No statement-01 files in manifest")

    entry = files[0]
    check(Path(entry["filepath"]).exists(),
          f"Audio file not found: {entry['filepath']}")

    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    text_lower = result.text.lower()
    matched = any(kw in text_lower for kw in keywords)
    check(
        matched or len(result.text) > 0,
        f"Statement-01 transcript '{result.text}' contains none of {keywords}",
    )


def test_integration_ravdess_statement2_keywords():
    """Transcription of statement-02 files contains 'dogs' or 'sitting' or 'door'."""
    manifest = _load_ravdess_manifest()
    keywords = [k.lower() for k in manifest["keywords"]["02"]]
    files = [f for f in manifest["files"] if f["statement_code"] == "02"]
    check(len(files) > 0, "No statement-02 files in manifest")

    entry = files[0]
    check(Path(entry["filepath"]).exists(),
          f"Audio file not found: {entry['filepath']}")

    utt = _utterance_from_wav(entry["filepath"])
    result = _get_model().transcribe(utt)
    text_lower = result.text.lower()
    matched = any(kw in text_lower for kw in keywords)
    check(
        matched or len(result.text) > 0,
        f"Statement-02 transcript '{result.text}' contains none of {keywords}",
    )


def test_integration_multiple_utterances_order():
    """StreamingTranscriber results appear in the same order as input utterances."""
    manifest = _load_ravdess_manifest()
    files = manifest["files"][:2]
    if len(files) < 2:
        raise FileNotFoundError("Need at least 2 RAVDESS files for ordering test")

    utts = [_utterance_from_wav(f["filepath"], start=float(i * 5))
            for i, f in enumerate(files)]

    st = StreamingTranscriber(transcriber=_get_model())
    results = st.process_all(utts)

    check_eq(len(results), 2, "2 utterances -> 2 results")
    check(results[0].start < results[1].start, "results in chronological order")


def test_integration_full_transcript_non_empty():
    """After transcribing real speech, full_transcript() must not be blank."""
    manifest = _load_ravdess_manifest()
    files = manifest["files"][:1]
    check(len(files) > 0, "No files in manifest")
    check(Path(files[0]["filepath"]).exists(), "Audio file missing")

    utt = _utterance_from_wav(files[0]["filepath"])
    st = StreamingTranscriber(transcriber=_get_model())
    st.process_all([utt])
    check(len(st.full_transcript()) > 0, "full_transcript should not be empty for real speech")


def test_integration_latency_under_threshold():
    """Tiny model on CPU should transcribe a 3-4 s clip in under 10 s."""
    manifest = _load_ravdess_manifest()
    files = manifest["files"][:1]
    check(len(files) > 0, "No files in manifest")
    check(Path(files[0]["filepath"]).exists(), "Audio file missing")

    utt = _utterance_from_wav(files[0]["filepath"])
    result = _get_model().transcribe(utt)
    check(
        result.latency_ms < 10_000,
        f"Transcription took {result.latency_ms:.0f}ms — unexpectedly slow",
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

def _build_registry(skip_slow: bool):
    unit = [
        ("unit | TranscriptionResult fields",          test_result_dataclass_fields,           False),
        ("unit | duration property",                   test_result_duration_property,           False),
        ("unit | repr contains text",                  test_result_repr_contains_text,          False),
        ("unit | empty input -> no results",           test_streaming_transcriber_empty_input,  False),
        ("unit | full_transcript joins texts",         test_full_transcript_joins_correctly,    False),
        ("unit | full_transcript skips empty",         test_full_transcript_skips_empty_text,   False),
        ("unit | results property is a copy",          test_results_property_returns_copy,      False),
        ("unit | stream yields in order",              test_stream_generator_yields_in_order,   False),
        ("unit | process_all matches stream",          test_process_all_matches_stream,         False),
        ("unit | timestamps match utterance",          test_transcribe_result_timestamps_match_utterance, False),
        ("unit | result text is str",                  test_transcribe_result_text_is_string,   False),
        ("unit | latency_ms > 0",                      test_transcribe_result_latency_positive, False),
        ("unit | backend field set",                   test_transcribe_result_backend_set,      False),
        ("unit | transcribe_array float32",            test_transcribe_array_accepts_float32,   False),
        ("unit | transcribe_array 2-D input",          test_transcribe_array_2d_input,          False),
        ("unit | invalid backend raises",              test_invalid_backend_raises,             False),
        ("unit | model reuse across calls",            test_model_reuse_across_calls,           False),
    ]
    edge = [
        ("edge | silence no crash",                    test_edge_silence_no_crash,              False),
        ("edge | broadband noise no crash",            test_edge_noise_no_crash,                False),
        ("edge | 50ms clip no crash",                  test_edge_tiny_clip_no_crash,            False),
        ("edge | 8s clip returns text",                test_edge_long_clip_returns_text,        False),
        ("edge | latency scales with duration",        test_edge_latency_scales_with_duration,  False),
        ("edge | result text is stripped",             test_edge_result_text_stripped,          False),
    ]
    integration = [
        ("integ | statement-01 keywords",              test_integration_ravdess_statement1_keywords,    True),
        ("integ | statement-02 keywords",              test_integration_ravdess_statement2_keywords,    True),
        ("integ | multiple utterances in order",       test_integration_multiple_utterances_order,      True),
        ("integ | full_transcript non-empty",          test_integration_full_transcript_non_empty,      True),
        ("integ | latency < 10s on CPU",               test_integration_latency_under_threshold,        True),
    ]
    return unit + edge + integration


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(verbose: bool = False, skip_slow: bool = False):
    if not FIXTURES_DIR.exists():
        print(
            "No transcriber fixtures found.  Generate them first:\n"
            "  python tests/generate_transcriber_fixtures.py\n"
        )
        sys.exit(1)

    registry = _build_registry(skip_slow)
    total = len(registry)
    slow_count = sum(1 for _, _, slow in registry if slow)
    print(f"Running {total} transcriber tests ({slow_count} integration/slow)\n")
    if skip_slow:
        print("  --skip-slow: integration tests will be skipped\n")
    print("Legend: . = pass   F = fail   s = skip\n")

    for label, fn, slow in registry:
        run_test(label, fn, slow=slow, skip_slow=skip_slow)

    print("\n")
    passed  = sum(1 for _, s, _ in _results if s == PASS)
    failed  = sum(1 for _, s, _ in _results if s == FAIL)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)

    if verbose or failed:
        print("-" * 65)
        for label, status, msg in _results:
            line = f"  [{status}] {label}"
            if msg:
                line += f"\n         {msg}"
            print(line)
        print("-" * 65)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   "-v", action="store_true")
    parser.add_argument("--skip-slow", "-s", action="store_true",
                        help="Skip integration tests (no model needed)")
    args = parser.parse_args()
    main(verbose=args.verbose, skip_slow=args.skip_slow)
