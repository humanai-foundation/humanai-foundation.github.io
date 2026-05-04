"""
Step 4 — Streaming Transcription
==================================
Converts buffered speech chunks (Utterance objects from Step 3) into text.

Backends (auto-selected in priority order)
------------------------------------------
  faster-whisper  — CTranslate2 backend, 4x faster than openai-whisper.
                    tiny model: ~300–600 ms on CPU, <100 ms on GPU.
                    Install: pip install faster-whisper
  openai-whisper  — Original Whisper.  Already in requirements.txt.
                    Slower but requires no extra install.

Sliding-window streaming
------------------------
  StreamingTranscriber wraps the core Transcriber in a generator pipeline:
  it accepts Utterance objects one at a time (from Step 3) and yields
  TranscriptionResult objects as each utterance finishes.  This means the
  transcriber runs in parallel with the next utterance being captured —
  keeping caption lag to one utterance window rather than whole-file latency.

Output
------
  TranscriptionResult(text="The forest was quiet that night.",
                      start=1.2, end=3.84, latency_ms=412.0, backend="faster-whisper")

Usage (standalone demo)
-----------------------
  python transcriber/streaming_transcriber.py --input examples/captured_audio.wav
  python transcriber/streaming_transcriber.py --input audio.wav --backend openai-whisper
  python transcriber/streaming_transcriber.py --duration 10   # live mic

Usage (library)
---------------
  from transcriber.streaming_transcriber import Transcriber, StreamingTranscriber
  from utterance_buffer.segmenter import segment_utterances
  from vad_engine.vad import detect_speech_segments

  t = Transcriber(model_size="tiny")
  for utterance in utterances:
      result = t.transcribe(utterance)
      print(result.text)
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False

try:
    import whisper as _openai_whisper
    _OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    _OPENAI_WHISPER_AVAILABLE = False


def _default_backend() -> str:
    if _FASTER_WHISPER_AVAILABLE:
        return "faster-whisper"
    if _OPENAI_WHISPER_AVAILABLE:
        return "openai-whisper"
    raise ImportError(
        "No Whisper backend found.  Install one:\n"
        "  pip install faster-whisper       (recommended)\n"
        "  pip install openai-whisper"
    )


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """
    Text output for one transcribed utterance.

    Attributes
    ----------
    text : str
        Transcribed text, stripped of leading/trailing whitespace.
    start : float
        Utterance start time in seconds (from stream origin).
    end : float
        Utterance end time in seconds.
    latency_ms : float
        Wall-clock time taken to transcribe this chunk (milliseconds).
    backend : str
        Which backend produced this result.
    confidence : float
        Average log-probability from faster-whisper (0.0 if unavailable).
    language : str
        Detected language code (e.g. "en"), empty string if unavailable.
    """
    text: str
    start: float
    end: float
    latency_ms: float = 0.0
    backend: str = ""
    confidence: float = 0.0
    language: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f'TranscriptionResult(text={self.text!r}, '
            f'start={self.start:.3f}s, end={self.end:.3f}s, '
            f'latency={self.latency_ms:.0f}ms, backend={self.backend!r})'
        )


# ---------------------------------------------------------------------------
# Core Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Loads a Whisper model once and transcribes Utterance objects on demand.

    Parameters
    ----------
    model_size : str
        Whisper model size: "tiny", "base", "small", "medium", "large".
        "tiny" is recommended for real-time use on CPU.
    backend : str | None
        "faster-whisper" or "openai-whisper".  None = auto (prefers faster-whisper).
    device : str
        "cpu" or "cuda".
    compute_type : str
        faster-whisper quantisation: "int8" (fastest CPU), "float16" (GPU),
        "float32" (highest quality).
    language : str | None
        Force a language (e.g. "en") to skip detection and save ~50ms.
        None = auto-detect.
    beam_size : int
        faster-whisper beam size.  1 = greedy (fastest), 5 = default.
    verbose : bool
        Print each result to stdout as it arrives.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        backend: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        beam_size: int = 5,
        verbose: bool = True,
    ):
        self.model_size = model_size
        self.backend = backend or _default_backend()
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.verbose = verbose
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        t0 = time.perf_counter()
        if self.backend == "faster-whisper":
            if not _FASTER_WHISPER_AVAILABLE:
                raise ImportError("faster-whisper not installed.  Run: pip install faster-whisper")
            self._model = _FasterWhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        else:
            if not _OPENAI_WHISPER_AVAILABLE:
                raise ImportError("openai-whisper not installed.  Run: pip install openai-whisper")
            self._model = _openai_whisper.load_model(self.model_size)

        load_ms = (time.perf_counter() - t0) * 1000
        if self.verbose:
            print(
                f"[Transcriber] Model loaded — backend={self.backend}, "
                f"size={self.model_size}, device={self.device}, "
                f"load_time={load_ms:.0f}ms"
            )

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = 16000,
        start: float = 0.0, end: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe a raw float32 PCM numpy array.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array at `sample_rate`.
        sample_rate : int
            Audio sample rate in Hz.
        start, end : float
            Source timestamps for the result metadata.

        Returns
        -------
        TranscriptionResult
        """
        if audio.ndim != 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)

        # Resample to 16 kHz if needed (Whisper requirement)
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                pass  # hope it's already 16 kHz

        t0 = time.perf_counter()
        text, confidence, language = self._run_backend(audio)
        latency_ms = (time.perf_counter() - t0) * 1000

        result = TranscriptionResult(
            text=text.strip(),
            start=start,
            end=end,
            latency_ms=latency_ms,
            backend=self.backend,
            confidence=confidence,
            language=language,
        )

        if self.verbose:
            print(
                f"[Transcriber] {start:.3f}s -> {end:.3f}s  "
                f"({latency_ms:.0f}ms)  \"{result.text}\""
            )
        return result

    def transcribe(self, utterance) -> TranscriptionResult:
        """
        Transcribe an Utterance object (from Step 3 segmenter).

        Parameters
        ----------
        utterance : Utterance
            Must have .audio (float32 ndarray), .start, .end attributes.

        Returns
        -------
        TranscriptionResult
        """
        return self.transcribe_array(
            audio=utterance.audio,
            sample_rate=16000,
            start=utterance.start,
            end=utterance.end,
        )

    def _run_backend(self, audio: np.ndarray):
        """Run the loaded backend and return (text, confidence, language)."""
        if self.backend == "faster-whisper":
            return self._run_faster_whisper(audio)
        return self._run_openai_whisper(audio)

    def _run_faster_whisper(self, audio: np.ndarray):
        segments_gen, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,   # VAD already handled by Step 2
        )
        segments = list(segments_gen)
        text = " ".join(s.text for s in segments)
        # Average log-probability across all words as a confidence proxy
        all_words = [w for s in segments for w in (s.words or [])]
        if all_words:
            confidence = float(np.mean([w.probability for w in all_words]))
        else:
            confidence = float(np.mean([s.avg_logprob for s in segments])) if segments else 0.0
        language = info.language if info else ""
        return text, confidence, language

    def _run_openai_whisper(self, audio: np.ndarray):
        result = self._model.transcribe(
            audio,
            language=self.language,
            fp16=False,
        )
        text = result.get("text", "")
        language = result.get("language", "")
        # openai-whisper doesn't expose per-word probabilities easily
        confidence = 0.0
        return text, confidence, language


# ---------------------------------------------------------------------------
# StreamingTranscriber — generator pipeline
# ---------------------------------------------------------------------------

class StreamingTranscriber:
    """
    Wraps Transcriber in a generator pipeline that accepts Utterance objects
    one at a time and yields TranscriptionResult objects as each finishes.

    This keeps caption lag to one utterance window rather than the full
    recording length — the transcription of chunk N runs while chunk N+1
    is still being captured.

    Parameters
    ----------
    transcriber : Transcriber | None
        A pre-loaded Transcriber.  If None, one is created with defaults.
    **transcriber_kwargs
        Passed to Transcriber() if transcriber is None.
    """

    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        **transcriber_kwargs,
    ):
        self._transcriber = transcriber or Transcriber(**transcriber_kwargs)
        self._results: List[TranscriptionResult] = []

    def stream(self, utterances: Iterator) -> Generator[TranscriptionResult, None, None]:
        """
        Yield a TranscriptionResult for each Utterance as it is processed.

        Parameters
        ----------
        utterances : iterable of Utterance

        Yields
        ------
        TranscriptionResult
        """
        for utterance in utterances:
            result = self._transcriber.transcribe(utterance)
            self._results.append(result)
            yield result

    def process_all(self, utterances) -> List[TranscriptionResult]:
        """Transcribe all utterances and return results as a list."""
        return list(self.stream(utterances))

    @property
    def results(self) -> List[TranscriptionResult]:
        """All results produced so far."""
        return list(self._results)

    def full_transcript(self, separator: str = " ") -> str:
        """Concatenate all result texts in order."""
        return separator.join(r.text for r in self._results if r.text)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def transcribe_utterances(
    utterances,
    model_size: str = "tiny",
    backend: Optional[str] = None,
    device: str = "cpu",
    language: Optional[str] = None,
    verbose: bool = True,
) -> List[TranscriptionResult]:
    """
    One-call helper: transcribe a list of Utterances and return results.

    Parameters
    ----------
    utterances : list of Utterance
    model_size : str
    backend : str | None
    device : str
    language : str | None
    verbose : bool

    Returns
    -------
    List[TranscriptionResult]
    """
    t = Transcriber(
        model_size=model_size,
        backend=backend,
        device=device,
        language=language,
        verbose=verbose,
    )
    st = StreamingTranscriber(transcriber=t)
    return st.process_all(utterances)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4 — Streaming Transcription demo"
    )
    parser.add_argument("--input", default=None, metavar="FILE.WAV",
                        help="WAV file to transcribe (default: live mic)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Live recording duration in seconds (default: 10)")
    parser.add_argument("--rate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: tiny)")
    parser.add_argument("--backend", default=None,
                        choices=["faster-whisper", "openai-whisper"],
                        help="Transcription backend (default: auto)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--language", default=None,
                        help="Force language, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--strategy", default="pause_triggered",
                        choices=["pause_triggered", "fixed_window"],
                        help="Utterance segmentation strategy (default: pause_triggered)")
    parser.add_argument("--pause", type=float, default=0.4,
                        help="Pause threshold in seconds (default: 0.4)")
    parser.add_argument("--vad-mode", type=int, default=2, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness 0-3 (default: 2)")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    for mod in ("vad_engine", "utterance_buffer", "task0_audio_capture"):
        p = str(root / mod)
        if p not in sys.path:
            sys.path.insert(0, p)

    from vad import detect_speech_segments
    from segmenter import segment_utterances

    print(
        f"\nStep 4 — Streaming Transcription\n"
        f"  Backend     : {args.backend or _default_backend()}\n"
        f"  Model size  : {args.model}\n"
        f"  Device      : {args.device}\n"
        f"  Language    : {args.language or 'auto'}\n"
        f"  VAD mode    : {args.vad_mode}\n"
        f"  Strategy    : {args.strategy}\n"
    )

    # Load audio
    if args.input:
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.rate)
        print(f"[Transcriber] Loaded {args.input}  ({len(audio)/args.rate:.2f} s)\n")
    else:
        from audio_capture import record_for_duration
        print(f"[Transcriber] Recording {args.duration:.1f} s from microphone ...")
        audio = record_for_duration(duration=args.duration, sample_rate=args.rate, verbose=True)

    # Step 2 — VAD
    print("\n[Transcriber] Running VAD ...")
    vad_segs = detect_speech_segments(
        audio, sample_rate=args.rate, frame_ms=20,
        aggressiveness=args.vad_mode, verbose=False,
    )
    print(f"[Transcriber] {len(vad_segs)} VAD segment(s) detected")

    # Step 3 — segmentation
    utterances = segment_utterances(
        vad_segs, strategy=args.strategy,
        pause_s=args.pause, sample_rate=args.rate, verbose=False,
    )
    print(f"[Transcriber] {len(utterances)} utterance(s) to transcribe\n")

    if not utterances:
        print("[Transcriber] No speech detected — nothing to transcribe.")
        return

    # Step 4 — streaming transcription
    transcriber = Transcriber(
        model_size=args.model,
        backend=args.backend,
        device=args.device,
        language=args.language,
        verbose=True,
    )
    st = StreamingTranscriber(transcriber=transcriber)

    print("\n--- Live transcript ---")
    for result in st.stream(iter(utterances)):
        print(f"  [{result.start:.2f}s] {result.text}")

    # Summary
    print("\n--- Full transcript ---")
    print(st.full_transcript())

    total_latency = sum(r.latency_ms for r in st.results)
    avg_latency = total_latency / len(st.results)
    print(
        f"\n[Transcriber] {len(st.results)} result(s)  "
        f"avg latency={avg_latency:.0f}ms  "
        f"total latency={total_latency:.0f}ms"
    )


if __name__ == "__main__":
    main()
