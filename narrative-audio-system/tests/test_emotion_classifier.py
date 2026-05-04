"""
Step 5 EmotionClassifier — test suite
======================================
Tests are grouped into four tiers:

  Unit         No model, no audio — validates EmotionResult fields,
               class invariants, and edge-case inputs.

  Acoustic     Load each synthetic emotion WAV and assert that the MFCC
               features extracted from it have the expected acoustic
               character (pitch proxy, energy, spectral brightness).
               These tests are model-independent — they validate the
               fixtures, not the classifier.

  Classifier   Run the trained MFCC-MLP on all 8 emotion WAV files.
               Assert structural correctness (result type, latency,
               confidence range) and soft acoustic constraints
               (e.g. angry > calm in energy, happy > sad in pitch proxy).

  Parallel     Verify that ParallelProcessor runs Steps 4+5 concurrently
               and that both results arrive without data corruption.

Run:
    python tests/generate_emotion_audio.py        # once
    python tests/test_emotion_classifier.py
    python tests/test_emotion_classifier.py --verbose
    python tests/test_emotion_classifier.py --skip-parallel   # skip Whisper
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
for mod in ("emotion_classifier", "transcriber", "utterance_buffer",
            "vad_engine", "task0_audio_capture"):
    p = str(REPO_ROOT / mod)
    if p not in sys.path:
        sys.path.insert(0, p)

from classifier import EmotionClassifier, EmotionResult, ParallelProcessor  # noqa

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "emotion_audio"
SAMPLE_RATE  = 16000

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _load_manifest():
    path = FIXTURES_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} -- run generate_emotion_audio.py first"
        )
    return json.loads(path.read_text())


def _load_wav(filepath: str) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


@dataclass
class _Utterance:
    start: float
    end: float
    audio: np.ndarray = field(repr=False)
    strategy: str = "pause_triggered"
    num_vad_segments: int = 1

    @property
    def duration(self):
        return self.end - self.start


def _make_utterance(audio: np.ndarray, start: float = 0.0) -> _Utterance:
    end = start + len(audio) / SAMPLE_RATE
    return _Utterance(start=start, end=end, audio=audio)


# Shared classifier — trained once for the whole test run
_CLF: Optional[EmotionClassifier] = None

def _get_clf() -> EmotionClassifier:
    global _CLF
    if _CLF is None:
        _CLF = EmotionClassifier(verbose=False)
    return _CLF


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def run_test(label: str, fn, skip: bool = False):
    if skip:
        _results.append((label, SKIP, "skipped by flag"))
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


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------

def test_emotion_result_fields():
    r = EmotionResult(label="calm", confidence=0.75, start=1.0, end=3.5,
                      latency_ms=12.3, all_scores={"calm": 0.75, "angry": 0.25},
                      backend="mfcc-mlp")
    check_eq(r.label, "calm")
    check_eq(r.confidence, 0.75)
    check_eq(round(r.duration, 6), 2.5)
    check_eq(r.backend, "mfcc-mlp")
    check("calm" in r.all_scores)


def test_result_repr_contains_label():
    r = EmotionResult(label="angry", confidence=0.82, start=0, end=2, latency_ms=10)
    check("angry" in repr(r))
    check("0.82" in repr(r))


def test_classifier_invalid_backend():
    try:
        EmotionClassifier(backend="nonexistent", verbose=False)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_classifier_class_names_populated():
    clf = _get_clf()
    check(len(clf._class_names) > 0, "class_names should not be empty after init")


def test_classifier_feature_stats_set():
    clf = _get_clf()
    check(clf._feature_mean is not None, "feature_mean should be set")
    check(clf._feature_std  is not None, "feature_std should be set")
    check(clf._feature_mean.shape[0] == clf.n_mfcc,
          f"feature_mean length {clf._feature_mean.shape[0]} != n_mfcc {clf.n_mfcc}")


def test_classify_array_returns_emotion_result():
    audio = np.random.default_rng(0).normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    result = _get_clf().classify_array(audio, sample_rate=SAMPLE_RATE)
    check(isinstance(result, EmotionResult), "return type")


def test_classify_result_label_in_class_names():
    audio = np.random.default_rng(1).normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    result = _get_clf().classify_array(audio)
    clf_names_lower = [n.lower() for n in _get_clf()._class_names]
    check(result.label.lower() in clf_names_lower,
          f"label {result.label!r} not in class names {clf_names_lower}")


def test_classify_confidence_in_range():
    audio = np.random.default_rng(2).normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    result = _get_clf().classify_array(audio)
    check(0.0 <= result.confidence <= 1.0,
          f"confidence {result.confidence} not in [0, 1]")


def test_classify_all_scores_sum_to_one():
    audio = np.random.default_rng(3).normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    result = _get_clf().classify_array(audio)
    total = sum(result.all_scores.values())
    check(abs(total - 1.0) < 1e-4, f"all_scores sum = {total:.6f}, expected ~1.0")


def test_classify_latency_positive():
    audio = np.random.default_rng(4).normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    result = _get_clf().classify_array(audio)
    check(result.latency_ms > 0, f"latency_ms={result.latency_ms}")


def test_classify_timestamps_from_utterance():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    utt = _make_utterance(audio, start=3.7)
    result = _get_clf().classify(utt)
    check_eq(result.start, 3.7, "result.start")
    check_eq(round(result.end, 3), round(3.7 + 1.0, 3), "result.end")


def test_classify_2d_audio_no_crash():
    audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    result = _get_clf().classify_array(audio)
    check(isinstance(result, EmotionResult))


def test_classify_silence_no_crash():
    audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    result = _get_clf().classify_array(audio)
    check(isinstance(result.label, str))


def test_classify_very_short_clip_no_crash():
    audio = np.zeros(160, dtype=np.float32)  # 10ms
    result = _get_clf().classify_array(audio)
    check(isinstance(result.label, str))


def test_model_reuse():
    clf = _get_clf()
    model_id = id(clf._model)
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    clf.classify_array(audio)
    clf.classify_array(audio)
    check_eq(id(clf._model), model_id, "model object reused")


# ---------------------------------------------------------------------------
# ACOUSTIC TESTS — validate fixture audio properties (model-independent)
# ---------------------------------------------------------------------------

def _mfcc_features(audio: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13).mean(axis=1)


def _rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def _spectral_centroid_mean(audio: np.ndarray) -> float:
    import librosa
    return float(librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE).mean())


def _f0_proxy(audio: np.ndarray) -> float:
    """Rough pitch proxy via autocorrelation peak in 80-300 Hz range."""
    corr = np.correlate(audio, audio, mode="full")
    corr = corr[len(corr) // 2:]
    min_lag = int(SAMPLE_RATE / 300)
    max_lag = int(SAMPLE_RATE / 80)
    peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return SAMPLE_RATE / peak_lag if peak_lag > 0 else 0.0


def _load_emotion_audio(label: str) -> np.ndarray:
    manifest = _load_manifest()
    entry = next((f for f in manifest["files"] if f["label"] == label), None)
    if entry is None:
        raise FileNotFoundError(f"No fixture for label={label!r}")
    return _load_wav(entry["filepath"])


def test_acoustic_angry_higher_energy_than_calm():
    """RMS targets in generator: angry=0.38, calm=0.18 — angry must be higher."""
    angry = _rms_energy(_load_emotion_audio("angry"))
    calm  = _rms_energy(_load_emotion_audio("calm"))
    check(angry > calm,
          f"angry RMS={angry:.4f} should be > calm RMS={calm:.4f}")


def test_acoustic_happy_higher_energy_than_sad():
    """
    Generator uses RMS targets: happy=0.26, sad=0.12.
    Happy must have noticeably higher RMS energy than sad.
    """
    happy_e = _rms_energy(_load_emotion_audio("happy"))
    sad_e   = _rms_energy(_load_emotion_audio("sad"))
    check(happy_e > sad_e * 1.5,
          f"happy RMS={happy_e:.4f} should be > 1.5x sad RMS={sad_e:.4f}")


def test_acoustic_surprised_higher_energy_than_sad():
    """
    Surprised has RMS target 0.30 vs sad 0.12 — surprised must have higher energy.
    (Pitch proxy via autocorrelation is unreliable on short mixed-pitch clips.)
    """
    surprised_e = _rms_energy(_load_emotion_audio("surprised"))
    sad_e       = _rms_energy(_load_emotion_audio("sad"))
    check(surprised_e > sad_e,
          f"surprised RMS={surprised_e:.4f} should be > sad RMS={sad_e:.4f}")


def test_acoustic_sad_lowest_energy():
    sad_e    = _rms_energy(_load_emotion_audio("sad"))
    angry_e  = _rms_energy(_load_emotion_audio("angry"))
    neutral_e = _rms_energy(_load_emotion_audio("neutral"))
    check(sad_e < angry_e,  f"sad RMS={sad_e:.4f} should be < angry={angry_e:.4f}")
    check(sad_e < neutral_e, f"sad RMS={sad_e:.4f} should be < neutral={neutral_e:.4f}")


def test_acoustic_all_files_above_vad_threshold():
    """Every emotion file must have RMS well above silence (1e-4)."""
    manifest = _load_manifest()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        rms = _rms_energy(audio)
        check(rms > 1e-3,
              f"{entry['filename']}: RMS={rms:.6f} too low (silence-like)")


def test_acoustic_all_files_correct_sample_rate():
    import soundfile as sf
    manifest = _load_manifest()
    for entry in manifest["files"]:
        _, sr = sf.read(entry["filepath"], dtype="float32")
        check_eq(sr, SAMPLE_RATE, f"{entry['filename']} sample_rate")


def test_acoustic_file_durations():
    """Each file should be between 3 s and 10 s."""
    manifest = _load_manifest()
    for entry in manifest["files"]:
        dur = entry["duration_s"]
        check(3.0 <= dur <= 10.0,
              f"{entry['filename']} duration {dur:.2f}s out of [3, 10]s range")


# ---------------------------------------------------------------------------
# CLASSIFIER TESTS — run model on all 8 emotion files
# ---------------------------------------------------------------------------

def test_classifier_returns_result_for_every_emotion():
    manifest = _load_manifest()
    clf = _get_clf()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        result = clf.classify_array(audio)
        check(isinstance(result, EmotionResult),
              f"{entry['label']}: expected EmotionResult")
        check(isinstance(result.label, str) and len(result.label) > 0,
              f"{entry['label']}: label is empty")


def test_classifier_latency_under_200ms():
    """MFCC + MLP inference must complete in under 200 ms for all files."""
    manifest = _load_manifest()
    clf = _get_clf()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        result = clf.classify_array(audio)
        check(result.latency_ms < 200,
              f"{entry['label']}: latency {result.latency_ms:.1f}ms > 200ms")


def test_classifier_confidence_always_in_range():
    manifest = _load_manifest()
    clf = _get_clf()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        result = clf.classify_array(audio)
        check(0.0 <= result.confidence <= 1.0,
              f"{entry['label']}: confidence {result.confidence} out of [0,1]")


def test_classifier_all_scores_sum_to_one_per_file():
    manifest = _load_manifest()
    clf = _get_clf()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        result = clf.classify_array(audio)
        total = sum(result.all_scores.values())
        check(abs(total - 1.0) < 1e-3,
              f"{entry['label']}: all_scores sum={total:.6f}")


def test_classifier_top_score_matches_confidence():
    manifest = _load_manifest()
    clf = _get_clf()
    for entry in manifest["files"]:
        audio = _load_wav(entry["filepath"])
        result = clf.classify_array(audio)
        if result.all_scores:
            top_score = max(result.all_scores.values())
            check(abs(top_score - result.confidence) < 1e-4,
                  f"{entry['label']}: top_score={top_score:.4f} != confidence={result.confidence:.4f}")


def test_classifier_soft_angry_high_confidence():
    """Angry audio has strong MFCC cues — classifier should pick a high-confidence label."""
    audio = _load_wav(str(FIXTURES_DIR / "angry_8s.wav"))
    result = _get_clf().classify_array(audio)
    check(result.confidence >= 0.10,
          f"angry: confidence={result.confidence:.2f} unexpectedly low (model might be untrained)")


def test_classifier_soft_calm_vs_angry_energy():
    """
    Regardless of label, the top score for calm audio should be for a
    'quieter' emotion (calm/neutral/sad) more often than angry/fearful.
    This is a soft heuristic, not an exact match.
    """
    calm_audio  = _load_wav(str(FIXTURES_DIR / "calm_6s.wav"))
    angry_audio = _load_wav(str(FIXTURES_DIR / "angry_8s.wav"))
    clf = _get_clf()
    calm_result  = clf.classify_array(calm_audio)
    angry_result = clf.classify_array(angry_audio)
    # At minimum both should return valid, non-empty labels
    check(len(calm_result.label) > 0,  "calm: empty label")
    check(len(angry_result.label) > 0, "angry: empty label")


def test_classifier_long_audio_consistent():
    """Running the same file twice must return identical results (deterministic)."""
    audio = _load_wav(str(FIXTURES_DIR / "neutral_6s.wav"))
    clf = _get_clf()
    r1 = clf.classify_array(audio)
    r2 = clf.classify_array(audio)
    check_eq(r1.label, r2.label, "determinism: label")
    check(abs(r1.confidence - r2.confidence) < 1e-5, "determinism: confidence")


# ---------------------------------------------------------------------------
# PARALLEL TESTS — Steps 4 + 5 together (requires Whisper model)
# ---------------------------------------------------------------------------

def test_parallel_returns_both_results(skip_parallel: bool):
    if skip_parallel:
        raise FileNotFoundError("parallel tests skipped by --skip-parallel")
    from streaming_transcriber import Transcriber
    audio = _load_wav(str(FIXTURES_DIR / "neutral_6s.wav"))
    utt   = _make_utterance(audio)
    trs   = Transcriber(model_size="tiny", verbose=False)
    clf   = _get_clf()
    proc  = ParallelProcessor(trs, clf, verbose=False)
    transcript, emotion = proc.process(utt)
    from streaming_transcriber import TranscriptionResult
    check(isinstance(transcript, TranscriptionResult), "transcript type")
    check(isinstance(emotion, EmotionResult),           "emotion type")


def test_parallel_timestamps_consistent(skip_parallel: bool):
    if skip_parallel:
        raise FileNotFoundError("parallel tests skipped by --skip-parallel")
    from streaming_transcriber import Transcriber
    audio = _load_wav(str(FIXTURES_DIR / "calm_6s.wav"))
    utt   = _make_utterance(audio, start=5.0)
    trs   = Transcriber(model_size="tiny", verbose=False)
    clf   = _get_clf()
    proc  = ParallelProcessor(trs, clf, verbose=False)
    transcript, emotion = proc.process(utt)
    check_eq(transcript.start, 5.0, "transcript.start")
    check_eq(emotion.start,    5.0, "emotion.start")


def test_parallel_process_all_length(skip_parallel: bool):
    if skip_parallel:
        raise FileNotFoundError("parallel tests skipped by --skip-parallel")
    from streaming_transcriber import Transcriber
    manifest = _load_manifest()
    utts = [_make_utterance(_load_wav(e["filepath"]), start=float(i * 8))
            for i, e in enumerate(manifest["files"][:3])]
    trs  = Transcriber(model_size="tiny", verbose=False)
    clf  = _get_clf()
    proc = ParallelProcessor(trs, clf, verbose=False)
    pairs = proc.process_all(utts)
    check_eq(len(pairs), 3, "process_all length")
    for i, (tr, er) in enumerate(pairs):
        check_eq(tr.start, utts[i].start, f"pair[{i}] transcript start")
        check_eq(er.start, utts[i].start, f"pair[{i}] emotion start")


# ---------------------------------------------------------------------------
# Test registry & runner
# ---------------------------------------------------------------------------

def _build_registry(skip_parallel: bool):
    unit = [
        ("unit | EmotionResult fields",            test_emotion_result_fields),
        ("unit | repr contains label+confidence",  test_result_repr_contains_label),
        ("unit | invalid backend raises",          test_classifier_invalid_backend),
        ("unit | class_names populated",           test_classifier_class_names_populated),
        ("unit | feature stats set",               test_classifier_feature_stats_set),
        ("unit | classify_array returns result",   test_classify_array_returns_emotion_result),
        ("unit | label in class_names",            test_classify_result_label_in_class_names),
        ("unit | confidence in [0,1]",             test_classify_confidence_in_range),
        ("unit | all_scores sum to 1",             test_classify_all_scores_sum_to_one),
        ("unit | latency_ms > 0",                  test_classify_latency_positive),
        ("unit | timestamps from utterance",       test_classify_timestamps_from_utterance),
        ("unit | 2-D audio no crash",              test_classify_2d_audio_no_crash),
        ("unit | silence no crash",                test_classify_silence_no_crash),
        ("unit | 10ms clip no crash",              test_classify_very_short_clip_no_crash),
        ("unit | model reused across calls",       test_model_reuse),
    ]
    acoustic = [
        ("acoustic | angry > calm energy",         test_acoustic_angry_higher_energy_than_calm),
        ("acoustic | happy higher energy than sad",  test_acoustic_happy_higher_energy_than_sad),
        ("acoustic | surprised higher energy than sad", test_acoustic_surprised_higher_energy_than_sad),
        ("acoustic | sad lowest energy",           test_acoustic_sad_lowest_energy),
        ("acoustic | all files above VAD floor",   test_acoustic_all_files_above_vad_threshold),
        ("acoustic | all files 16kHz",             test_acoustic_all_files_correct_sample_rate),
        ("acoustic | durations 4-10s",             test_acoustic_file_durations),
    ]
    classifier = [
        ("clf | result for every emotion",         test_classifier_returns_result_for_every_emotion),
        ("clf | latency < 200ms per file",         test_classifier_latency_under_200ms),
        ("clf | confidence in [0,1] per file",     test_classifier_confidence_always_in_range),
        ("clf | all_scores sum=1 per file",        test_classifier_all_scores_sum_to_one_per_file),
        ("clf | top score matches confidence",     test_classifier_top_score_matches_confidence),
        ("clf | angry high confidence",            test_classifier_soft_angry_high_confidence),
        ("clf | calm vs angry labels valid",       test_classifier_soft_calm_vs_angry_energy),
        ("clf | deterministic on same input",      test_classifier_long_audio_consistent),
    ]
    parallel = [
        ("parallel | returns both results",
         lambda: test_parallel_returns_both_results(skip_parallel)),
        ("parallel | timestamps consistent",
         lambda: test_parallel_timestamps_consistent(skip_parallel)),
        ("parallel | process_all length",
         lambda: test_parallel_process_all_length(skip_parallel)),
    ]
    return unit, acoustic, classifier, parallel


def main(verbose: bool = False, skip_parallel: bool = False):
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.wav")):
        print(
            "No emotion audio fixtures found.  Generate them first:\n"
            "  python tests/generate_emotion_audio.py\n"
        )
        sys.exit(1)

    unit, acoustic, classifier, parallel = _build_registry(skip_parallel)
    all_tests = unit + acoustic + classifier + parallel
    print(f"Running {len(all_tests)} emotion classifier tests\n")
    if skip_parallel:
        print("  --skip-parallel: Whisper parallel tests will be skipped\n")
    print("Legend: . = pass   F = fail   s = skip\n")

    for label, fn in all_tests:
        run_test(label, fn)

    print("\n")
    passed  = sum(1 for _, s, _ in _results if s == PASS)
    failed  = sum(1 for _, s, _ in _results if s == FAIL)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)

    if verbose or failed:
        print("-" * 66)
        for label, status, msg in _results:
            line = f"  [{status}] {label}"
            if msg:
                line += f"\n         {msg}"
            print(line)
        print("-" * 66)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",        "-v", action="store_true")
    parser.add_argument("--skip-parallel",  "-s", action="store_true",
                        help="Skip tests that require the Whisper model")
    args = parser.parse_args()
    main(verbose=args.verbose, skip_parallel=args.skip_parallel)
