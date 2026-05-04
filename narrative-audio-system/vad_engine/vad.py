"""
Step 2 — Voice Activity Detection (VAD)
=========================================
Labels each 10–30 ms audio frame as speech or non-speech, then groups
consecutive speech frames into timestamped segments.  Silence is never
forwarded to the transcriber, which saves compute and prevents mid-sentence
cuts.

Two backends are supported (auto-selected at import time):
  1. webrtcvad  — Google's WebRTC VAD, <1 ms per frame, rule-based.
  2. silero-vad — Silero neural VAD, ~5 ms per frame, more accurate on noise.

Typical output
--------------
  [VAD] Speech started  at T =  1.200 s
  [VAD] Speech ended    at T =  3.840 s  (duration 2.640 s)
  SpeechSegment(start=1.2, end=3.84, audio=array([...], dtype=float32))

Usage (standalone demo)
-----------------------
  python vad.py --duration 10 --mode 2 --frame-ms 20

Usage (library)
---------------
  from vad_engine.vad import VADProcessor, SpeechSegment

  processor = VADProcessor(sample_rate=16000, frame_ms=20, aggressiveness=2)
  for segment in processor.process_array(audio_array):
      print(f"Speech {segment.start:.2f}s – {segment.end:.2f}s")
"""

import argparse
import collections
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    import webrtcvad as _webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_SILERO_AVAILABLE = False
if _TORCH_AVAILABLE:
    try:
        _silero_model, _silero_utils = _torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        _SILERO_AVAILABLE = True
    except Exception:
        pass


def _backend_name() -> str:
    if _SILERO_AVAILABLE:
        return "silero-vad"
    if _WEBRTCVAD_AVAILABLE:
        return "webrtcvad"
    return "energy"           # simple energy-threshold fallback


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """A detected speech utterance with absolute timestamps."""
    start: float                         # seconds from stream start
    end: float                           # seconds from stream start
    audio: np.ndarray = field(repr=False)  # float32 PCM samples

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"SpeechSegment(start={self.start:.3f}s, end={self.end:.3f}s, "
            f"duration={self.duration:.3f}s, samples={len(self.audio)})"
        )


# ---------------------------------------------------------------------------
# Frame-level VAD helpers
# ---------------------------------------------------------------------------

def _to_int16_bytes(frame_f32: np.ndarray) -> bytes:
    """Convert a float32 PCM frame to int16 little-endian bytes for webrtcvad."""
    clipped = np.clip(frame_f32, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _energy_is_speech(frame: np.ndarray, threshold: float = 0.005) -> bool:
    """Fallback: RMS energy threshold."""
    return float(np.sqrt(np.mean(frame ** 2))) > threshold


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class VADProcessor:
    """
    Runs VAD over a numpy audio array and yields SpeechSegment objects.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.  webrtcvad supports 8000, 16000, 32000, 48000.
    frame_ms : int
        Frame duration in milliseconds.  webrtcvad supports 10, 20, 30.
    aggressiveness : int
        webrtcvad aggressiveness 0–3 (0 = least, 3 = most aggressive).
        Higher values filter more background noise but may clip soft speech.
    speech_pad_ms : int
        Extra ms to prepend before a detected speech onset (avoids clipping
        the first syllable).
    silence_pad_ms : int
        Extra ms of silence to append after speech ends before closing the
        segment.  Prevents splitting on brief pauses (e.g. comma pauses).
    min_speech_ms : int
        Minimum speech duration to emit.  Shorter bursts (e.g. clicks) are
        discarded.
    backend : str | None
        Force a specific backend: "webrtcvad", "silero", or "energy".
        None = auto-select best available.
    verbose : bool
        Print segment events to stdout.
    """

    VALID_RATES = {8000, 16000, 32000, 48000}
    VALID_FRAME_MS = {10, 20, 30}

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        speech_pad_ms: int = 300,
        silence_pad_ms: int = 400,
        min_speech_ms: int = 250,
        backend: Optional[str] = None,
        verbose: bool = True,
    ):
        if sample_rate not in self.VALID_RATES:
            raise ValueError(f"sample_rate must be one of {self.VALID_RATES}, got {sample_rate}")
        if frame_ms not in self.VALID_FRAME_MS:
            raise ValueError(f"frame_ms must be one of {self.VALID_FRAME_MS}, got {frame_ms}")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness
        self.speech_pad_ms = speech_pad_ms
        self.silence_pad_ms = silence_pad_ms
        self.min_speech_ms = min_speech_ms
        self.verbose = verbose

        self.frame_samples = int(sample_rate * frame_ms / 1000)  # e.g. 320 @ 16kHz/20ms

        # Number of frames for padding windows
        self.num_speech_pad_frames = max(1, speech_pad_ms // frame_ms)
        self.num_silence_pad_frames = max(1, silence_pad_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)

        # Select backend
        self._backend = backend or _backend_name()
        self._vad = None
        self._silero_model = None

        if self._backend == "webrtcvad":
            if not _WEBRTCVAD_AVAILABLE:
                raise ImportError("webrtcvad not installed.  Run: pip install webrtcvad")
            self._vad = _webrtcvad.Vad(aggressiveness)
        elif self._backend == "silero":
            if not _SILERO_AVAILABLE:
                raise ImportError("silero-vad not available.  Ensure torch is installed.")
            self._silero_model = _silero_model
            self._silero_model.reset_states()
        # else: energy fallback — no setup needed

        if verbose:
            print(
                f"[VAD] Initialized — backend={self._backend}, "
                f"rate={sample_rate} Hz, frame={frame_ms} ms "
                f"({self.frame_samples} samples), aggressiveness={aggressiveness}"
            )

    # ------------------------------------------------------------------
    # Frame-level speech decision
    # ------------------------------------------------------------------

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Return True if this frame contains speech."""
        if self._backend == "webrtcvad":
            return self._vad.is_speech(_to_int16_bytes(frame), self.sample_rate)
        elif self._backend == "silero":
            tensor = _torch.tensor(frame, dtype=_torch.float32).unsqueeze(0)
            confidence = float(self._silero_model(tensor, self.sample_rate).item())
            return confidence > 0.5
        else:
            return _energy_is_speech(frame)

    # ------------------------------------------------------------------
    # Segment state machine
    # ------------------------------------------------------------------

    def process_array(
        self, audio: np.ndarray, stream_offset_s: float = 0.0
    ) -> Generator[SpeechSegment, None, None]:
        """
        Process a 1-D float32 audio array and yield SpeechSegment objects.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 PCM array at `self.sample_rate`.
        stream_offset_s : float
            Time offset (seconds) to add to all timestamps.  Useful when
            processing chunks of a longer recording.

        Yields
        ------
        SpeechSegment
        """
        if self._backend == "silero" and self._silero_model is not None:
            self._silero_model.reset_states()

        # Split audio into fixed-size frames; drop the last incomplete frame
        num_complete = len(audio) // self.frame_samples
        frames = [
            audio[i * self.frame_samples: (i + 1) * self.frame_samples]
            for i in range(num_complete)
        ]

        # Sliding window of recent frames (used for pre-speech padding)
        ring = collections.deque(maxlen=self.num_speech_pad_frames)

        in_speech = False
        triggered_frame = 0          # frame index when speech was triggered
        speech_frames: List[np.ndarray] = []
        num_silence_frames = 0
        num_actual_speech_frames = 0  # only frames labeled as speech (excludes padding)

        for idx, frame in enumerate(frames):
            frame_time = stream_offset_s + idx * self.frame_ms / 1000.0
            is_speech = self._is_speech(frame)

            if not in_speech:
                ring.append((frame, is_speech))
                num_speech = sum(1 for _, s in ring if s)

                # Trigger on: majority of ring frames are speech
                if num_speech > self.num_speech_pad_frames // 2:
                    in_speech = True
                    triggered_frame = idx - len(ring) + 1
                    trigger_time = stream_offset_s + triggered_frame * self.frame_ms / 1000.0
                    if self.verbose:
                        print(f"[VAD] Speech started  at T = {trigger_time:7.3f} s")
                    # Include the ring buffer frames as pre-speech padding
                    speech_frames = [f for f, _ in ring]
                    # Count only the speech-labeled frames in the ring
                    num_actual_speech_frames = num_speech
                    num_silence_frames = 0
            else:
                speech_frames.append(frame)

                if not is_speech:
                    num_silence_frames += 1
                else:
                    num_silence_frames = 0
                    num_actual_speech_frames += 1

                # End segment after enough consecutive silence frames
                if num_silence_frames > self.num_silence_pad_frames:
                    # Trim trailing silence down to silence_pad_ms
                    keep = len(speech_frames) - num_silence_frames + self.num_silence_pad_frames
                    speech_frames = speech_frames[:keep]

                    end_time = (
                        stream_offset_s
                        + (triggered_frame + keep) * self.frame_ms / 1000.0
                    )
                    start_time = stream_offset_s + triggered_frame * self.frame_ms / 1000.0

                    # Min-duration check uses only actually-speech-labeled frames
                    if num_actual_speech_frames >= self.min_speech_frames:
                        segment_audio = np.concatenate(speech_frames)
                        if self.verbose:
                            print(
                                f"[VAD] Speech ended    at T = {end_time:7.3f} s  "
                                f"(duration {end_time - start_time:.3f} s)"
                            )
                        yield SpeechSegment(
                            start=start_time,
                            end=end_time,
                            audio=segment_audio,
                        )
                    else:
                        if self.verbose:
                            print(
                                f"[VAD] Discarded short burst "
                                f"({num_actual_speech_frames * self.frame_ms} ms "
                                f"< {self.min_speech_ms} ms)"
                            )

                    # Reset
                    in_speech = False
                    speech_frames = []
                    num_silence_frames = 0
                    num_actual_speech_frames = 0
                    ring.clear()

        # Flush any open segment at end of array
        if in_speech and speech_frames:
            end_time = stream_offset_s + (triggered_frame + len(speech_frames)) * self.frame_ms / 1000.0
            start_time = stream_offset_s + triggered_frame * self.frame_ms / 1000.0
            if num_actual_speech_frames >= self.min_speech_frames:
                segment_audio = np.concatenate(speech_frames)
                if self.verbose:
                    print(
                        f"[VAD] Speech ended    at T = {end_time:7.3f} s  "
                        f"(duration {end_time - start_time:.3f} s)  [end-of-audio]"
                    )
                yield SpeechSegment(start=start_time, end=end_time, audio=segment_audio)


# ---------------------------------------------------------------------------
# Live streaming VAD — wraps AudioCaptureStream + VADProcessor
# ---------------------------------------------------------------------------

class LiveVAD:
    """
    Combines Step 1 (AudioCaptureStream) with VADProcessor to emit
    SpeechSegments in real time from the microphone.

    Parameters
    ----------
    sample_rate : int
        Microphone and VAD sample rate.
    chunk_size : int
        AudioCaptureStream chunk size (samples).  Should be a multiple of
        `frame_samples` for clean alignment.
    frame_ms : int
        VAD frame duration in ms.
    aggressiveness : int
        webrtcvad aggressiveness 0–3.
    on_segment : callable | None
        Called with each SpeechSegment as it is detected.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        speech_pad_ms: int = 300,
        silence_pad_ms: int = 400,
        min_speech_ms: int = 250,
        on_segment=None,
        verbose: bool = True,
    ):
        # Import here to avoid hard dependency at module level
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task0_audio_capture"))
        from audio_capture import AudioCaptureStream, RollingBuffer

        self.sample_rate = sample_rate
        self.on_segment = on_segment
        self.verbose = verbose
        self._segments: List[SpeechSegment] = []
        self._stream_start: float = 0.0
        self._pending_audio = np.array([], dtype=np.float32)

        self._processor = VADProcessor(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            aggressiveness=aggressiveness,
            speech_pad_ms=speech_pad_ms,
            silence_pad_ms=silence_pad_ms,
            min_speech_ms=min_speech_ms,
            verbose=verbose,
        )

        frame_samples = self._processor.frame_samples

        def _on_chunk(chunk: np.ndarray) -> None:
            """Called from the audio thread for every captured chunk."""
            elapsed = time.perf_counter() - self._stream_start
            flat = chunk.flatten().astype(np.float32)

            # Accumulate until we have at least one full VAD frame
            combined = np.concatenate([self._pending_audio, flat])
            n_frames = len(combined) // frame_samples
            usable = n_frames * frame_samples
            self._pending_audio = combined[usable:]

            if n_frames == 0:
                return

            processable = combined[:usable]
            offset = elapsed - len(combined) / sample_rate

            for seg in self._processor.process_array(processable, stream_offset_s=max(0.0, offset)):
                self._segments.append(seg)
                if self.on_segment is not None:
                    self.on_segment(seg)

        self._buffer = RollingBuffer(max_seconds=60.0, sample_rate=sample_rate)
        self._capture = AudioCaptureStream(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            buffer=self._buffer,
            on_chunk=_on_chunk,
        )

    def start(self) -> "LiveVAD":
        self._stream_start = time.perf_counter()
        self._capture.start()
        return self

    def stop(self) -> List[SpeechSegment]:
        self._capture.stop()
        return self._segments

    def __enter__(self) -> "LiveVAD":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    @property
    def segments(self) -> List[SpeechSegment]:
        return list(self._segments)


# ---------------------------------------------------------------------------
# Public helper: detect_speech_segments
# ---------------------------------------------------------------------------

def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: int = 20,
    aggressiveness: int = 2,
    speech_pad_ms: int = 300,
    silence_pad_ms: int = 400,
    min_speech_ms: int = 250,
    verbose: bool = True,
) -> List[SpeechSegment]:
    """
    Run VAD over a pre-recorded audio array and return a list of
    SpeechSegment objects sorted by start time.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 PCM at `sample_rate`.
    sample_rate : int
        Audio sample rate in Hz.
    frame_ms : int
        VAD frame duration in ms (10, 20, or 30).
    aggressiveness : int
        webrtcvad aggressiveness 0–3.
    speech_pad_ms : int
        Pre-speech padding in ms.
    silence_pad_ms : int
        Post-speech silence tolerance in ms.
    min_speech_ms : int
        Minimum segment length in ms.
    verbose : bool
        Print segment events.

    Returns
    -------
    List[SpeechSegment]
    """
    processor = VADProcessor(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        aggressiveness=aggressiveness,
        speech_pad_ms=speech_pad_ms,
        silence_pad_ms=silence_pad_ms,
        min_speech_ms=min_speech_ms,
        verbose=verbose,
    )
    return list(processor.process_array(audio))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Step 2 — Voice Activity Detection demo")
    parser.add_argument("--input", default=None, metavar="FILE.WAV",
                        help="Process a WAV file instead of the microphone")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Live recording duration in seconds (default: 10)")
    parser.add_argument("--rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30],
                        help="VAD frame duration in ms (default: 20)")
    parser.add_argument("--mode", type=int, default=2, choices=[0, 1, 2, 3],
                        help="webrtcvad aggressiveness 0-3 (default: 2)")
    parser.add_argument("--speech-pad", type=int, default=300,
                        help="Pre-speech padding ms (default: 300)")
    parser.add_argument("--silence-pad", type=int, default=400,
                        help="Post-speech silence tolerance ms (default: 400)")
    parser.add_argument("--min-speech", type=int, default=250,
                        help="Minimum speech segment ms (default: 250)")
    parser.add_argument("--backend", default=None, choices=["webrtcvad", "silero", "energy"],
                        help="Force a VAD backend (default: auto)")
    return parser.parse_args()


def main():
    args = _parse_args()

    print(
        f"\nStep 2 — Voice Activity Detection\n"
        f"  Backend     : {args.backend or _backend_name()}\n"
        f"  Sample rate : {args.rate} Hz\n"
        f"  Frame size  : {args.frame_ms} ms\n"
        f"  Aggressiveness: {args.mode}\n"
        f"  Speech pad  : {args.speech_pad} ms\n"
        f"  Silence pad : {args.silence_pad} ms\n"
        f"  Min speech  : {args.min_speech} ms\n"
    )

    if args.input:
        # Process a pre-recorded WAV file
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mix to mono
        if sr != args.rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.rate)
        print(f"[VAD] Loaded {args.input}  ({len(audio)/args.rate:.2f} s)\n")

        segments = detect_speech_segments(
            audio,
            sample_rate=args.rate,
            frame_ms=args.frame_ms,
            aggressiveness=args.mode,
            speech_pad_ms=args.speech_pad,
            silence_pad_ms=args.silence_pad,
            min_speech_ms=args.min_speech,
        )
    else:
        # Live microphone capture
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task0_audio_capture"))
        try:
            from audio_capture import record_for_duration
        except ImportError:
            print("ERROR: task0_audio_capture not found.  Run from narrative-audio-system/.")
            return

        print(f"[VAD] Recording {args.duration:.1f} s from microphone...")
        audio = record_for_duration(duration=args.duration, sample_rate=args.rate, verbose=True)

        segments = detect_speech_segments(
            audio,
            sample_rate=args.rate,
            frame_ms=args.frame_ms,
            aggressiveness=args.mode,
            speech_pad_ms=args.speech_pad,
            silence_pad_ms=args.silence_pad,
            min_speech_ms=args.min_speech,
        )

    # Summary
    total_speech = sum(s.duration for s in segments)
    total_audio = len(audio) / args.rate
    print(
        f"\n[VAD] Detected {len(segments)} speech segment(s)\n"
        f"[VAD] Total speech : {total_speech:.3f} s / {total_audio:.3f} s "
        f"({100*total_speech/total_audio:.1f}%)"
    )
    for i, seg in enumerate(segments, 1):
        print(f"      [{i:2d}] {seg.start:7.3f}s -> {seg.end:7.3f}s  ({seg.duration:.3f} s)")


if __name__ == "__main__":
    main()
