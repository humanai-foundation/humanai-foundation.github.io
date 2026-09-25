"""
Step 3 — Buffering & Utterance Segmentation
=============================================
Collects VAD speech segments (from Step 2) into utterance-sized chunks
that are ready to send to the transcriber.

Two strategies are supported:

  pause_triggered (default / better)
    Emits a chunk whenever the gap between two consecutive VAD segments
    exceeds `pause_s` (default 0.4 s).  This naturally follows breath
    groups and phrasing pauses, approximating sentence boundaries without
    any linguistic knowledge.  Also emits when the accumulated speech
    exceeds `max_utterance_s` (safety valve against very long run-ons).

  fixed_window (simpler / faster)
    Emits a chunk every time `window_s` seconds of *speech* have been
    accumulated, regardless of where pauses fall.  Lower latency but may
    cut words mid-syllable.

Tuning guidance
---------------
  pause_s too small  → fragments (splits on comma pauses mid-sentence)
  pause_s too large  → long lag before the transcriber sees audio
  window_s too small → words cut off at chunk boundaries
  window_s too large → captions noticeably delayed

Typical settings for live stage narration:
  pause_triggered, pause_s=0.4, max_utterance_s=8.0

Usage (standalone demo)
-----------------------
  python utterance_buffer/segmenter.py --input examples/captured_audio.wav

Usage (library)
---------------
  from utterance_buffer.segmenter import UtteranceSegmenter, Utterance
  from vad_engine.vad import detect_speech_segments

  segments = detect_speech_segments(audio, sample_rate=16000)
  segmenter = UtteranceSegmenter(strategy="pause_triggered", pause_s=0.4)
  for utterance in segmenter.process_segments(segments):
      print(utterance)          # -> transcriber
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """
    A transcriber-ready speech chunk assembled from one or more VAD segments.

    Attributes
    ----------
    start : float
        Start time in seconds (relative to stream origin).
    end : float
        End time in seconds.
    audio : np.ndarray
        Concatenated float32 PCM samples for the chunk.
    strategy : str
        Which segmentation strategy produced this utterance.
    num_vad_segments : int
        How many VAD segments were merged into this utterance.
    """
    start: float
    end: float
    audio: np.ndarray = field(repr=False)
    strategy: str = "pause_triggered"
    num_vad_segments: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_duration(self) -> float:
        """Actual audio length (seconds), which may differ from wall-clock span."""
        return len(self.audio) / 16000  # updated by segmenter with real sr

    def __repr__(self) -> str:
        return (
            f"Utterance(start={self.start:.3f}s, end={self.end:.3f}s, "
            f"duration={self.duration:.3f}s, "
            f"vad_segments={self.num_vad_segments}, strategy={self.strategy!r})"
        )


# ---------------------------------------------------------------------------
# Core segmenter
# ---------------------------------------------------------------------------

class UtteranceSegmenter:
    """
    Accumulates VAD SpeechSegments and emits Utterances when a boundary
    is detected.

    Parameters
    ----------
    strategy : {"pause_triggered", "fixed_window"}
        Segmentation strategy (see module docstring).
    pause_s : float
        Pause-triggered: inter-segment gap (seconds) that triggers an emit.
    window_s : float
        Fixed-window: speech duration (seconds) before forcing an emit.
    max_utterance_s : float
        Pause-triggered safety valve: emit when accumulated speech exceeds
        this value regardless of detected pauses.
    sample_rate : int
        Audio sample rate — used for speech_duration calculation.
    verbose : bool
        Print emit events to stdout.
    """

    STRATEGIES = {"pause_triggered", "fixed_window"}

    def __init__(
        self,
        strategy: str = "pause_triggered",
        pause_s: float = 0.4,
        window_s: float = 2.5,
        max_utterance_s: float = 8.0,
        sample_rate: int = 16000,
        verbose: bool = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.strategy = strategy
        self.pause_s = pause_s
        self.window_s = window_s
        self.max_utterance_s = max_utterance_s
        self.sample_rate = sample_rate
        self.verbose = verbose

        # Internal buffer
        self._segments: List = []          # accumulated SpeechSegments
        self._audio_chunks: List[np.ndarray] = []
        self._speech_duration: float = 0.0  # total seconds of speech buffered

        if verbose:
            if strategy == "pause_triggered":
                print(
                    f"[Segmenter] pause_triggered — "
                    f"pause_s={pause_s}s, max_utterance_s={max_utterance_s}s"
                )
            else:
                print(f"[Segmenter] fixed_window — window_s={window_s}s")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self) -> Optional[Utterance]:
        """Assemble and return the buffered utterance, then reset the buffer."""
        if not self._segments:
            return None

        utterance = Utterance(
            start=self._segments[0].start,
            end=self._segments[-1].end,
            audio=np.concatenate(self._audio_chunks).astype(np.float32),
            strategy=self.strategy,
            num_vad_segments=len(self._segments),
        )

        speech_s = self._speech_duration
        if self.verbose:
            print(
                f"[Segmenter] Emit utterance  "
                f"{utterance.start:.3f}s -> {utterance.end:.3f}s  "
                f"({utterance.duration:.3f}s span, "
                f"{speech_s:.3f}s speech, "
                f"{utterance.num_vad_segments} VAD segment(s))"
            )

        self._segments = []
        self._audio_chunks = []
        self._speech_duration = 0.0
        return utterance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_segment(self, segment) -> Optional[Utterance]:
        """
        Feed one VAD SpeechSegment.  Returns an Utterance if one is ready,
        otherwise returns None (keep buffering).

        Parameters
        ----------
        segment : SpeechSegment
            A speech segment from vad_engine.vad.detect_speech_segments().

        Returns
        -------
        Utterance | None
        """
        seg_duration = segment.duration

        if self.strategy == "fixed_window":
            self._audio_chunks.append(segment.audio)
            self._segments.append(segment)
            self._speech_duration += seg_duration

            if self._speech_duration >= self.window_s:
                return self._emit()
            return None

        # pause_triggered ---------------------------------------------------
        result = None

        if self._segments:
            gap = segment.start - self._segments[-1].end

            # Pause long enough → emit what we have, start fresh with this seg
            if gap >= self.pause_s:
                result = self._emit()

            # Safety valve — accumulated speech too long
            elif self._speech_duration >= self.max_utterance_s:
                result = self._emit()

        self._segments.append(segment)
        self._audio_chunks.append(segment.audio)
        self._speech_duration += seg_duration
        return result

    def flush(self) -> Optional[Utterance]:
        """
        Force-emit whatever is currently buffered (call at end of stream).
        Returns None if the buffer is empty.
        """
        if self._segments:
            if self.verbose:
                print("[Segmenter] Flush — end of stream")
            return self._emit()
        return None

    def process_segments(self, segments) -> List[Utterance]:
        """
        Process a complete list of VAD SpeechSegments and return all
        resulting Utterance objects (including a final flush).

        Parameters
        ----------
        segments : iterable of SpeechSegment

        Returns
        -------
        List[Utterance]
        """
        utterances: List[Utterance] = []
        for seg in segments:
            u = self.feed_segment(seg)
            if u is not None:
                utterances.append(u)
        final = self.flush()
        if final is not None:
            utterances.append(final)
        return utterances

    def stream_segments(self, segments) -> Generator[Utterance, None, None]:
        """
        Generator version of process_segments — yields Utterances as they
        become ready.  Useful for live pipelines where segments arrive one
        at a time.

        Parameters
        ----------
        segments : iterable of SpeechSegment

        Yields
        ------
        Utterance
        """
        for seg in segments:
            u = self.feed_segment(seg)
            if u is not None:
                yield u
        final = self.flush()
        if final is not None:
            yield final

    @property
    def buffered_duration(self) -> float:
        """Total speech seconds currently waiting in the buffer."""
        return self._speech_duration

    @property
    def buffered_segments(self) -> int:
        """Number of VAD segments currently in the buffer."""
        return len(self._segments)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def segment_utterances(
    vad_segments,
    strategy: str = "pause_triggered",
    pause_s: float = 0.4,
    window_s: float = 2.5,
    max_utterance_s: float = 8.0,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> List[Utterance]:
    """
    One-call helper: convert a list of VAD SpeechSegments into Utterances.

    Parameters
    ----------
    vad_segments : list of SpeechSegment
    strategy : "pause_triggered" | "fixed_window"
    pause_s : float
        Pause threshold for pause_triggered mode.
    window_s : float
        Window length for fixed_window mode.
    max_utterance_s : float
        Safety-valve maximum for pause_triggered mode.
    sample_rate : int
    verbose : bool

    Returns
    -------
    List[Utterance]
    """
    segmenter = UtteranceSegmenter(
        strategy=strategy,
        pause_s=pause_s,
        window_s=window_s,
        max_utterance_s=max_utterance_s,
        sample_rate=sample_rate,
        verbose=verbose,
    )
    return segmenter.process_segments(vad_segments)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3 — Buffering & Utterance Segmentation demo"
    )
    parser.add_argument("--input", default=None, metavar="FILE.WAV",
                        help="WAV file to process (default: live mic capture)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Live recording duration in seconds (default: 10)")
    parser.add_argument("--rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--strategy", default="pause_triggered",
                        choices=["pause_triggered", "fixed_window"],
                        help="Segmentation strategy (default: pause_triggered)")
    parser.add_argument("--pause", type=float, default=0.4,
                        help="Pause threshold in seconds (default: 0.4)")
    parser.add_argument("--window", type=float, default=2.5,
                        help="Fixed window size in seconds (default: 2.5)")
    parser.add_argument("--max", type=float, default=8.0,
                        help="Max utterance length in seconds (default: 8.0)")
    parser.add_argument("--vad-mode", type=int, default=2, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness 0-3 (default: 2)")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Resolve sibling module paths
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "vad_engine"))
    sys.path.insert(0, str(root / "task0_audio_capture"))

    from vad import detect_speech_segments

    print(
        f"\nStep 3 — Buffering & Utterance Segmentation\n"
        f"  Strategy    : {args.strategy}\n"
        f"  Pause thresh: {args.pause} s\n"
        f"  Window size : {args.window} s\n"
        f"  Max utt     : {args.max} s\n"
        f"  VAD mode    : {args.vad_mode}\n"
    )

    if args.input:
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.rate)
        print(f"[Segmenter] Loaded {args.input}  ({len(audio)/args.rate:.2f} s)\n")
    else:
        from audio_capture import record_for_duration
        print(f"[Segmenter] Recording {args.duration:.1f} s from microphone ...")
        audio = record_for_duration(duration=args.duration, sample_rate=args.rate, verbose=True)

    # Step 2 — VAD
    print("\n[Segmenter] Running VAD ...")
    vad_segments = detect_speech_segments(
        audio,
        sample_rate=args.rate,
        frame_ms=20,
        aggressiveness=args.vad_mode,
        verbose=True,
    )
    print(f"[Segmenter] VAD found {len(vad_segments)} speech segment(s)\n")

    # Step 3 — utterance segmentation
    utterances = segment_utterances(
        vad_segments,
        strategy=args.strategy,
        pause_s=args.pause,
        window_s=args.window,
        max_utterance_s=args.max,
        sample_rate=args.rate,
    )

    # Summary
    total_speech = sum(u.duration for u in utterances)
    print(f"\n[Segmenter] {len(utterances)} utterance(s) ready for transcriber:")
    for i, u in enumerate(utterances, 1):
        print(
            f"  [{i:2d}] {u.start:.3f}s -> {u.end:.3f}s  "
            f"span={u.duration:.3f}s  "
            f"speech={len(u.audio)/args.rate:.3f}s  "
            f"vad_segs={u.num_vad_segments}"
        )
    print(f"\n  Total utterance span : {total_speech:.3f} s")
    print(f"  Strategy used        : {args.strategy}")


if __name__ == "__main__":
    main()
