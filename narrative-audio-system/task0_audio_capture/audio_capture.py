"""
Step 1 — Audio Capture & Streaming
====================================
Continuously reads audio from the microphone in small chunks (frames) using
a stream callback that fires every ~30–128 ms with a new buffer of raw PCM
samples.  Accumulated chunks are stored in a thread-safe rolling buffer so
that downstream processing steps (feature extraction, classification, etc.)
can consume audio without blocking capture.

Usage (standalone demo):
    python audio_capture.py --duration 5 --chunk 1024 --rate 16000

Usage (as a library):
    from audio_capture import AudioCaptureStream, RollingBuffer

    buf = RollingBuffer(max_seconds=10, sample_rate=16000)
    with AudioCaptureStream(sample_rate=16000, chunk_size=1024, buffer=buf) as stream:
        time.sleep(5)          # capture for 5 seconds
    audio = buf.get_audio()    # numpy array of all captured samples
"""

import argparse
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 16000   # Hz — standard for speech processing
DEFAULT_CHUNK_SIZE = 1024     # samples (~64 ms at 16 kHz)
DEFAULT_CHANNELS = 1          # mono
DTYPE = np.float32            # PCM format expected by librosa / whisper


# ---------------------------------------------------------------------------
# RollingBuffer — accumulates raw PCM frames from the callback thread
# ---------------------------------------------------------------------------

class RollingBuffer:
    """
    Thread-safe circular buffer that stores raw PCM samples.

    Parameters
    ----------
    max_seconds : float
        Maximum audio duration to hold in memory.  Older frames are dropped
        once the buffer is full (rolling window semantics).
    sample_rate : int
        Audio sample rate in Hz (must match the capture stream).
    """

    def __init__(self, max_seconds: float = 30.0, sample_rate: int = DEFAULT_SAMPLE_RATE):
        max_frames = int(max_seconds * sample_rate)
        self._buffer: deque = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Internal API — called from the capture callback (audio thread)
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """Append a new PCM chunk (1-D float32 array) to the buffer."""
        flat = chunk.flatten().astype(DTYPE)
        with self._lock:
            self._buffer.extend(flat.tolist())

    # ------------------------------------------------------------------
    # Public API — called from the consumer thread
    # ------------------------------------------------------------------

    def get_audio(self) -> np.ndarray:
        """Return a copy of all buffered samples as a 1-D float32 array."""
        with self._lock:
            return np.array(self._buffer, dtype=DTYPE)

    def clear(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._buffer.clear()

    @property
    def duration_seconds(self) -> float:
        """Current buffered audio duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Current number of buffered samples."""
        with self._lock:
            return len(self._buffer)


# ---------------------------------------------------------------------------
# AudioCaptureStream — wraps sounddevice InputStream with callback
# ---------------------------------------------------------------------------

class AudioCaptureStream:
    """
    Continuously captures microphone audio in small fixed-size chunks.

    Each chunk is delivered to `buffer.push()` from the audio callback thread
    as soon as it arrives, providing low-latency streaming behaviour.

    Parameters
    ----------
    sample_rate : int
        Capture sample rate in Hz.
    chunk_size : int
        Number of samples per callback invocation.
        512  → ~32 ms  (lower latency, higher CPU)
        1024 → ~64 ms  (balanced default)
        2048 → ~128 ms (higher latency, lower CPU)
    channels : int
        Number of input channels (1 = mono recommended for speech).
    buffer : RollingBuffer | None
        Where captured chunks are stored.  A new RollingBuffer is created
        automatically when None is passed.
    device : int | str | None
        sounddevice device index or name.  None = system default.
    on_chunk : callable | None
        Optional hook called with each raw chunk array (audio thread context).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        channels: int = DEFAULT_CHANNELS,
        buffer: Optional[RollingBuffer] = None,
        device=None,
        on_chunk=None,
    ):
        if not _SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is not installed.  Run:  pip install sounddevice"
            )

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device
        self.on_chunk = on_chunk

        self.buffer: RollingBuffer = buffer or RollingBuffer(
            max_seconds=30.0, sample_rate=sample_rate
        )

        self._stream: Optional[sd.InputStream] = None
        self._chunk_count: int = 0
        self._error_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Callback — runs in the PortAudio audio thread (must be non-blocking)
    # ------------------------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback: fires every `chunk_size` samples."""
        if status:
            # Put status flags into the error queue; don't block here.
            self._error_queue.put_nowait(str(status))

        self.buffer.push(indata.copy())
        self._chunk_count += 1

        if self.on_chunk is not None:
            self.on_chunk(indata.copy())

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> "AudioCaptureStream":
        """Open the microphone stream and start capturing."""
        self._chunk_count = 0
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype=DTYPE,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        chunk_ms = round(self.chunk_size / self.sample_rate * 1000)
        print(
            f"[AudioCapture] Stream opened — "
            f"rate={self.sample_rate} Hz, "
            f"chunk={self.chunk_size} samples ({chunk_ms} ms), "
            f"channels={self.channels}"
        )
        return self

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print(
                f"[AudioCapture] Stream closed — "
                f"{self._chunk_count} chunks captured, "
                f"{self.buffer.duration_seconds:.2f} s buffered."
            )

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioCaptureStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Convenience — drain any reported PortAudio status warnings
    # ------------------------------------------------------------------

    def drain_errors(self) -> list:
        errors = []
        while not self._error_queue.empty():
            errors.append(self._error_queue.get_nowait())
        return errors


# ---------------------------------------------------------------------------
# Public helper: record_for_duration
# ---------------------------------------------------------------------------

def record_for_duration(
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    channels: int = DEFAULT_CHANNELS,
    device=None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Block until `duration` seconds of audio have been captured, then return
    the full recording as a 1-D float32 numpy array.

    Parameters
    ----------
    duration : float
        Recording length in seconds.
    sample_rate : int
        Microphone sample rate in Hz.
    chunk_size : int
        Callback chunk size in samples.
    channels : int
        Number of input channels.
    device : int | str | None
        sounddevice device index or name.
    verbose : bool
        Print progress dots while recording.

    Returns
    -------
    np.ndarray
        1-D float32 array of raw PCM samples, shape (duration * sample_rate,).
    """
    buf = RollingBuffer(max_seconds=duration + 2.0, sample_rate=sample_rate)
    with AudioCaptureStream(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        channels=channels,
        buffer=buf,
        device=device,
    ) as stream:
        target_samples = int(duration * sample_rate)
        if verbose:
            print(f"[AudioCapture] Recording for {duration:.1f} s ...", end="", flush=True)
        while buf.num_samples < target_samples:
            errs = stream.drain_errors()
            for e in errs:
                print(f"\n[AudioCapture] Warning: {e}")
            if verbose:
                print(".", end="", flush=True)
            time.sleep(0.05)
        if verbose:
            print(" done.")
    return buf.get_audio()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1 — Audio Capture & Streaming demo"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Recording duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--chunk", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in samples (default: 1024 → 64 ms at 16 kHz)"
    )
    parser.add_argument(
        "--rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--channels", type=int, default=DEFAULT_CHANNELS,
        help="Number of input channels (default: 1)"
    )
    parser.add_argument(
        "--device", default=None,
        help="sounddevice device index or name (default: system default)"
    )
    parser.add_argument(
        "--save", default=None, metavar="OUT.WAV",
        help="Save captured audio to a WAV file"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio devices and exit"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not _SOUNDDEVICE_AVAILABLE:
        print("ERROR: sounddevice is not installed.  Run:  pip install sounddevice")
        return

    if args.list_devices:
        print(sd.query_devices())
        return

    chunk_ms = round(args.chunk / args.rate * 1000)
    print(
        f"\nStep 1 — Audio Capture & Streaming\n"
        f"  Sample rate : {args.rate} Hz\n"
        f"  Chunk size  : {args.chunk} samples  ({chunk_ms} ms per frame)\n"
        f"  Channels    : {args.channels}\n"
        f"  Duration    : {args.duration} s\n"
    )

    audio = record_for_duration(
        duration=args.duration,
        sample_rate=args.rate,
        chunk_size=args.chunk,
        channels=args.channels,
        device=args.device,
    )

    print(f"\n[AudioCapture] Captured {len(audio)} samples ({len(audio)/args.rate:.3f} s)")
    print(f"[AudioCapture] Shape: {audio.shape}  dtype: {audio.dtype}")
    print(f"[AudioCapture] Amplitude range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"[AudioCapture] RMS level: {float(np.sqrt(np.mean(audio**2))):.6f}")

    if args.save:
        import soundfile as sf
        sf.write(args.save, audio, args.rate)
        print(f"[AudioCapture] Saved to {args.save}")


if __name__ == "__main__":
    main()
