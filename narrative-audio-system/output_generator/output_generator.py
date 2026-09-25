"""
Step 6 — Output Generation (Tracks A + B combined)
====================================================
Receives (TranscriptionResult, EmotionResult) pairs from the parallel
Steps 4+5 processor and fans out to both output tracks simultaneously:

  Track A — Accessibility Captions
    Format annotated caption line -> stdout + SRT file + WebSocket broadcast

  Track B — Atmospheric Audio Suggestion
    Map emotion label -> retrieval query -> crossfade schedule ->
    WebSocket broadcast (browser/OBS picks this up and triggers audio engine)

Both tracks are dispatched concurrently via ThreadPoolExecutor so neither
blocks the other.

Full pipeline
-------------
  [Step 1] Mic capture
  [Step 2] VAD
  [Step 3] Utterance segmentation
  [Step 4+5] Parallel transcription + emotion classification
       |
       v
  [Step 6] OutputGenerator  <-- this module
       |
       +---> Track A: [calm] "The forest was quiet that night."
       |              -> stdout / SRT / WebSocket ws://localhost:8765
       |
       +---> Track B: "calm" -> "calm gentle wind soft water"
                      -> CrossfadeSchedule (lag=6s, fade=2s)
                      -> WebSocket ws://localhost:8765

WebSocket message types
-----------------------
  Caption:    { "type":"caption",    "label":"calm", "text":"...", "color":"#7ec8a0", ... }
  Atmosphere: { "type":"atmosphere", "query":"calm gentle wind...", "fade_in_s":2, "lag_s":6 }

Usage (standalone demo)
-----------------------
  python output_generator/output_generator.py --input examples/captured_audio.wav
  python output_generator/output_generator.py --input audio.wav --ws   # with WebSocket

Usage (library)
---------------
  from output_generator.output_generator import OutputGenerator

  gen = OutputGenerator(srt_path="session.srt", enable_websocket=False)
  for transcript, emotion in parallel_processor.process_all(utterances):
      gen.process(transcript, emotion)
  gen.close()
"""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from captions import CaptionFormatter, SRTWriter, CaptionBroadcaster, CaptionLine
from atmosphere import AtmosphereMapper, CrossfadeScheduler, CrossfadeSchedule


# ---------------------------------------------------------------------------
# OutputGenerator
# ---------------------------------------------------------------------------

class OutputGenerator:
    """
    Fans out (TranscriptionResult, EmotionResult) to Track A and Track B
    concurrently.

    Parameters
    ----------
    srt_path : str | None
        If set, captions are also appended to this SRT file.
    enable_websocket : bool
        Start a WebSocket server on ws://localhost:<ws_port>.
    ws_host : str
    ws_port : int
    lag_s : float
        Atmosphere crossfade lag in seconds.
    fade_s : float
        Atmosphere crossfade duration in seconds.
    cooldown_s : float
        Minimum seconds between atmosphere changes.
    min_caption_length : int
        Captions shorter than this are dropped.
    verbose : bool
    """

    def __init__(
        self,
        srt_path: Optional[str] = None,
        enable_websocket: bool = False,
        ws_host: str = "localhost",
        ws_port: int = 8765,
        lag_s: float = 6.0,
        fade_s: float = 2.0,
        cooldown_s: float = 8.0,
        min_caption_length: int = 2,
        verbose: bool = True,
    ):
        self.verbose = verbose

        # Track A
        self._formatter  = CaptionFormatter(
            min_text_length=min_caption_length, verbose=verbose
        )
        self._srt_writer = SRTWriter(srt_path, verbose=verbose) if srt_path else None
        self._captions:  List[CaptionLine] = []

        # Track B
        self._scheduler  = CrossfadeScheduler(
            lag_s=lag_s, fade_s=fade_s, cooldown_s=cooldown_s, verbose=verbose
        )
        self._atmospheres: List[CrossfadeSchedule] = []

        # WebSocket (optional)
        self._broadcaster: Optional[CaptionBroadcaster] = None
        self._ws_loop:     Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread:   Optional[threading.Thread] = None

        if enable_websocket:
            self._start_websocket_server(ws_host, ws_port)

    # ------------------------------------------------------------------
    # WebSocket server (runs in a background daemon thread)
    # ------------------------------------------------------------------

    def _start_websocket_server(self, host: str, port: int) -> None:
        """Launch the WebSocket server in a background thread."""
        self._ws_loop = asyncio.new_event_loop()
        self._broadcaster = CaptionBroadcaster(host=host, port=port,
                                               verbose=self.verbose)

        def _run():
            asyncio.set_event_loop(self._ws_loop)
            self._ws_loop.run_until_complete(self._broadcaster.start())
            self._ws_loop.run_forever()

        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()

    def _ws_broadcast(self, payload: dict) -> None:
        """Thread-safe fire-and-forget broadcast to WebSocket clients."""
        if self._broadcaster is None or self._ws_loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcaster.broadcast(payload), self._ws_loop
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _track_a(self, transcript, emotion) -> Optional[CaptionLine]:
        """Track A: format caption, write SRT, broadcast."""
        line = self._formatter.format(transcript, emotion)
        if line is None:
            return None
        self._captions.append(line)
        if self._srt_writer:
            self._srt_writer.write(line)
        self._ws_broadcast(line.to_dict())
        return line

    def _track_b(self, emotion) -> Optional[CrossfadeSchedule]:
        """Track B: schedule atmosphere change, broadcast."""
        schedule = self._scheduler.schedule(emotion)
        if schedule is None:
            return None
        self._atmospheres.append(schedule)
        self._ws_broadcast(schedule.to_dict())
        return schedule

    def process(self, transcript, emotion) -> Tuple[Optional[CaptionLine],
                                                     Optional[CrossfadeSchedule]]:
        """
        Process one (TranscriptionResult, EmotionResult) pair through
        both output tracks concurrently.

        Returns
        -------
        (CaptionLine | None, CrossfadeSchedule | None)
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            a_future = pool.submit(self._track_a, transcript, emotion)
            b_future = pool.submit(self._track_b, emotion)
            caption   = a_future.result()
            schedule  = b_future.result()
        return caption, schedule

    def process_all(self, pairs) -> List[Tuple]:
        """
        Process a list of (TranscriptionResult, EmotionResult) pairs.

        Returns
        -------
        List[(CaptionLine | None, CrossfadeSchedule | None)]
        """
        return [self.process(tr, er) for tr, er in pairs]

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Session Summary",
            f"{'='*55}",
            f"  Captions generated  : {len(self._captions)}",
            f"  Atmosphere changes  : {len(self._atmospheres)}",
        ]
        if self._srt_writer:
            lines.append(f"  SRT file            : {self._srt_writer.path}")
        lines.append(f"\n  Full transcript:")
        for cap in self._captions:
            lines.append(f"    {cap.render()}")
        if self._atmospheres:
            lines.append(f"\n  Atmosphere log:")
            for atm in self._atmospheres:
                lines.append(
                    f"    [{atm.emotion_label}] -> {atm.suggested_description!r}"
                    f"  (lag={atm.lag_s}s, fade={atm.fade_in_s}s)"
                )
        lines.append(f"{'='*55}")
        return "\n".join(lines)

    def close(self) -> None:
        """Print session summary and stop the WebSocket server."""
        print(self.summary())
        if self._ws_loop and self._broadcaster:
            asyncio.run_coroutine_threadsafe(
                self._broadcaster.stop(), self._ws_loop
            )

    @property
    def captions(self) -> List[CaptionLine]:
        return list(self._captions)

    @property
    def atmosphere_log(self) -> List[CrossfadeSchedule]:
        return list(self._atmospheres)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 6 — Output Generation demo"
    )
    parser.add_argument("--input", default=None, metavar="FILE.WAV",
                        help="WAV file to process (default: live mic)")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--srt", default="examples/output.srt",
                        help="SRT output file (default: examples/output.srt)")
    parser.add_argument("--ws", action="store_true",
                        help="Enable WebSocket server on ws://localhost:8765")
    parser.add_argument("--lag",  type=float, default=6.0)
    parser.add_argument("--fade", type=float, default=2.0)
    parser.add_argument("--vad-mode", type=int, default=2)
    parser.add_argument("--ws-wait", type=float, default=3.0,
                        help="Seconds to wait after WS server starts before processing "
                             "(gives browser time to connect). Default: 3")
    parser.add_argument("--linger", type=float, default=15.0,
                        help="Seconds to keep WS server alive after processing "
                             "(so atmosphere lag effects play out). Default: 15")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    for mod in ("vad_engine", "utterance_buffer", "task0_audio_capture",
                "transcriber", "emotion_classifier"):
        p = str(root / mod)
        if p not in sys.path:
            sys.path.insert(0, p)

    from vad import detect_speech_segments
    from segmenter import segment_utterances
    from streaming_transcriber import Transcriber
    from classifier import EmotionClassifier, ParallelProcessor

    print(
        f"\nStep 6 — Output Generation\n"
        f"  Track A : captions -> stdout + {args.srt}\n"
        f"  Track B : atmosphere suggestions (lag={args.lag}s, fade={args.fade}s)\n"
        f"  WebSocket: {'ws://localhost:8765' if args.ws else 'disabled'}\n"
    )

    # Start WebSocket server FIRST so browser can connect before processing begins
    gen = OutputGenerator(
        srt_path=args.srt,
        enable_websocket=args.ws,
        lag_s=args.lag,
        fade_s=args.fade,
        verbose=True,
    )

    if args.ws:
        import time as _time
        print(f"[Output] WebSocket ready — open http://localhost:8000/overlay.html")
        print(f"[Output] Waiting {args.ws_wait}s for browser to connect...\n")
        _time.sleep(args.ws_wait)

    # Load audio
    if args.input:
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.rate)
        print(f"[Output] Loaded {args.input}  ({len(audio)/args.rate:.2f}s)\n")
    else:
        from audio_capture import record_for_duration
        audio = record_for_duration(duration=args.duration, sample_rate=args.rate)

    # Steps 1-5
    vad_segs   = detect_speech_segments(audio, sample_rate=args.rate,
                                        aggressiveness=args.vad_mode, verbose=False)
    utterances = segment_utterances(vad_segs, verbose=False)
    print(f"[Output] {len(utterances)} utterance(s)\n")

    transcriber = Transcriber(model_size="tiny", verbose=False)
    classifier  = EmotionClassifier(verbose=False)
    proc        = ParallelProcessor(transcriber, classifier, verbose=False)
    pairs       = proc.process_all(utterances)

    # Step 6 — output
    gen.process_all(pairs)

    if args.ws and args.linger > 0:
        import time as _time
        print(f"\n[Output] Lingering {args.linger}s so browser atmosphere effects play out...")
        _time.sleep(args.linger)

    gen.close()


if __name__ == "__main__":
    main()
