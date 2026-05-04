"""
Track A — Accessibility Captions
==================================
Formats (TranscriptionResult, EmotionResult) pairs into annotated caption
lines and distributes them over multiple channels simultaneously:

  stdout       — always active; plain-text fallback
  WebSocket    — async server on ws://localhost:8765; browser overlays /
                 OBS browser source connect here
  SRT file     — standard subtitle file for post-production / archiving

Caption format
--------------
  [calm]  "The forest was quiet that night."
  [tense] "Until the branch snapped."

Each caption is also serialised as JSON over the WebSocket so the browser
frontend can style tone labels with CSS classes:

  {
    "type": "caption",
    "label": "tense",
    "text":  "Until the branch snapped.",
    "start": 4.12,
    "end":   5.88,
    "color": "#e05c5c"
  }

Usage (library)
---------------
  from output_generator.captions import CaptionFormatter, SRTWriter

  fmt   = CaptionFormatter()
  line  = fmt.format(transcript_result, emotion_result)
  print(line.render())          # "[calm]  The forest was quiet..."

  writer = SRTWriter("output.srt")
  writer.write(line)

WebSocket server (async)
------------------------
  import asyncio
  from output_generator.captions import CaptionBroadcaster

  async def main():
      server = CaptionBroadcaster(host="localhost", port=8765)
      await server.start()
      # push captions from another coroutine:
      await server.broadcast(line)

  asyncio.run(main())
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Tone colour palette — maps emotion labels to hex colours for the overlay
# ---------------------------------------------------------------------------

TONE_COLOURS: Dict[str, str] = {
    "neutral":   "#a8b4c0",
    "calm":      "#7ec8a0",
    "happy":     "#f7c948",
    "sad":       "#6b9bcf",
    "angry":     "#e05c5c",
    "fearful":   "#c07ecf",
    "disgust":   "#8a9e6b",
    "surprised": "#f0a045",
    "tense":     "#e07a5f",   # alias used in narrative context
    "unknown":   "#ffffff",
}


def _tone_colour(label: str) -> str:
    return TONE_COLOURS.get(label.lower(), TONE_COLOURS["unknown"])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CaptionLine:
    """
    One annotated caption unit, ready for display.

    Attributes
    ----------
    label : str
        Emotion / tone label (e.g. "calm", "tense").
    text : str
        Transcribed utterance text.
    start : float
        Utterance start time in seconds.
    end : float
        Utterance end time in seconds.
    confidence : float
        Emotion classifier confidence (0-1).
    color : str
        Hex colour string for the tone label.
    index : int
        Sequential caption index (1-based, for SRT).
    """
    label:      str
    text:       str
    start:      float
    end:        float
    confidence: float = 0.0
    color:      str   = "#ffffff"
    index:      int   = 1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def render(self, width: int = 10) -> str:
        """Plain-text caption line, e.g. '[calm]   "The forest was quiet."'"""
        tag = f"[{self.label}]"
        return f"{tag:<{width}}  \"{self.text}\""

    def to_dict(self) -> dict:
        return {
            "type":       "caption",
            "index":      self.index,
            "label":      self.label,
            "text":       self.text,
            "start":      round(self.start, 3),
            "end":        round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "color":      self.color,
        }

    def to_srt_block(self) -> str:
        """Return an SRT-formatted block for this caption."""
        def _ts(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = s % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")

        return (
            f"{self.index}\n"
            f"{_ts(self.start)} --> {_ts(self.end)}\n"
            f"[{self.label}] {self.text}\n"
        )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class CaptionFormatter:
    """
    Combines a TranscriptionResult and EmotionResult into a CaptionLine.

    Parameters
    ----------
    min_text_length : int
        Captions shorter than this (characters) are skipped.  Filters out
        noise hallucinations from Whisper ("...", single punctuation, etc.)
    verbose : bool
        Print each formatted caption to stdout.
    """

    def __init__(self, min_text_length: int = 2, verbose: bool = True):
        self.min_text_length = min_text_length
        self.verbose = verbose
        self._index = 0

    def format(self, transcript, emotion) -> Optional[CaptionLine]:
        """
        Build a CaptionLine from Step 4 + Step 5 results.

        Parameters
        ----------
        transcript : TranscriptionResult
        emotion    : EmotionResult

        Returns
        -------
        CaptionLine | None  (None if text is too short to display)
        """
        text = transcript.text.strip()
        if len(text) < self.min_text_length:
            return None

        self._index += 1
        line = CaptionLine(
            label=emotion.label,
            text=text,
            start=transcript.start,
            end=transcript.end,
            confidence=emotion.confidence,
            color=_tone_colour(emotion.label),
            index=self._index,
        )

        if self.verbose:
            print(f"[Caption] {line.render()}")

        return line

    def reset_index(self) -> None:
        self._index = 0


# ---------------------------------------------------------------------------
# SRT writer
# ---------------------------------------------------------------------------

class SRTWriter:
    """
    Appends CaptionLine objects to an SRT subtitle file.

    Parameters
    ----------
    path : str | Path
        Output file path.  File is created (or appended) on first write.
    verbose : bool
    """

    def __init__(self, path: str = "output.srt", verbose: bool = True):
        self.path = Path(path)
        self.verbose = verbose
        self._count = 0
        # Clear existing file on open
        self.path.write_text("", encoding="utf-8")

    def write(self, line: CaptionLine) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line.to_srt_block() + "\n")
        self._count += 1
        if self.verbose:
            print(f"[SRT] Wrote caption {line.index} -> {self.path}")

    def write_all(self, lines: List[CaptionLine]) -> None:
        for line in lines:
            if line is not None:
                self.write(line)

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# WebSocket broadcaster (async)
# ---------------------------------------------------------------------------

class CaptionBroadcaster:
    """
    Async WebSocket server that broadcasts caption JSON to all connected
    clients.  Browser overlays and OBS browser sources connect to
    ws://localhost:<port>.

    Parameters
    ----------
    host : str
    port : int
    verbose : bool

    Example
    -------
    Browser JS:
        const ws = new WebSocket("ws://localhost:8765");
        ws.onmessage = e => {
            const d = JSON.parse(e.data);
            if (d.type === "caption") showCaption(d.label, d.text, d.color);
            if (d.type === "atmosphere") suggestAmbience(d.query);
        };
    """

    def __init__(self, host: str = "localhost", port: int = 8765,
                 verbose: bool = True):
        try:
            import websockets
            self._websockets = websockets
        except ImportError:
            raise ImportError("websockets not installed.  Run: pip install websockets")

        self.host = host
        self.port = port
        self.verbose = verbose
        self._clients: Set = set()
        self._server = None

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        if self.verbose:
            print(f"[WS] Client connected  ({len(self._clients)} total)")
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            if self.verbose:
                print(f"[WS] Client disconnected  ({len(self._clients)} remaining)")

    async def start(self) -> None:
        """Start the WebSocket server (non-blocking coroutine)."""
        self._server = await self._websockets.serve(
            self._handler, self.host, self.port
        )
        if self.verbose:
            print(f"[WS] Caption server listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            if self.verbose:
                print("[WS] Caption server stopped")

    async def broadcast(self, payload: dict) -> None:
        """Send a JSON payload to all connected clients."""
        if not self._clients:
            return
        message = json.dumps(payload)
        await asyncio.gather(
            *[client.send(message) for client in list(self._clients)],
            return_exceptions=True,
        )

    async def send_caption(self, line: CaptionLine) -> None:
        await self.broadcast(line.to_dict())

    async def send_raw(self, payload: dict) -> None:
        await self.broadcast(payload)

    @property
    def connected_clients(self) -> int:
        return len(self._clients)
