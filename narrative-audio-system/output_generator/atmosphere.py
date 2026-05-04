"""
Track B — Atmospheric Audio Suggestion
========================================
Maps the current emotion label to an ambient soundscape query, runs it
through the Task 4 retrieval system (or a curated fallback library),
and schedules a smooth crossfade to the new atmosphere.

Signal flow
-----------
  EmotionResult.label ("tense")
       |
       v
  AtmosphereMapper  ->  query string: "tense forest night ambience"
       |
       v
  RetrievalQuery    ->  ranked audio clips from Task 4 index
       |
       v
  CrossfadeSchedule ->  {clip, fade_in_s=2.0, lag_s=6.0}
       |
       v
  broadcast via WebSocket  {"type": "atmosphere", "query": ..., "clip": ...}

Lag and crossfade
-----------------
  A 5–8 s lag between speech detection and atmosphere change is intentional —
  it feels natural to an audience, matching how a film score would respond
  to dialogue.  The crossfade itself is 2 s by default.

Usage (library)
---------------
  from output_generator.atmosphere import AtmosphereMapper, CrossfadeScheduler

  mapper    = AtmosphereMapper()
  scheduler = CrossfadeScheduler(lag_s=6.0, fade_s=2.0)

  query     = mapper.query_for("tense")
  schedule  = scheduler.schedule(emotion_result, query)
  print(schedule)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Tone -> ambient query mapping
# ---------------------------------------------------------------------------

# Primary queries: chosen to work well with Task 4 semantic ranking
TONE_QUERIES: Dict[str, str] = {
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

# Curated fallback descriptions when no retrieval index is available
FALLBACK_AMBIENT: Dict[str, dict] = {
    "calm":      {"description": "gentle wind, soft water",        "bpm": None,  "energy": "low"},
    "neutral":   {"description": "quiet room tone, light hum",     "bpm": None,  "energy": "low"},
    "happy":     {"description": "birdsong, light breeze",         "bpm": 110,   "energy": "medium"},
    "sad":       {"description": "distant rain, sparse piano",     "bpm": 60,    "energy": "low"},
    "angry":     {"description": "driving percussion, wind",       "bpm": 140,   "energy": "high"},
    "fearful":   {"description": "dark forest, distant owl, creak","bpm": None,  "energy": "medium"},
    "tense":     {"description": "tense forest night, branch snap","bpm": None,  "energy": "medium"},
    "disgust":   {"description": "low rumble, dripping, echo",     "bpm": None,  "energy": "low"},
    "surprised": {"description": "bright orchestral stab, rush",   "bpm": 120,   "energy": "high"},
}


class AtmosphereMapper:
    """
    Maps an emotion label to an ambient soundscape query string.

    Parameters
    ----------
    custom_queries : dict | None
        Override default label->query mappings.
    verbose : bool
    """

    def __init__(self, custom_queries: Optional[Dict[str, str]] = None,
                 verbose: bool = True):
        self._queries = {**TONE_QUERIES}
        if custom_queries:
            self._queries.update(custom_queries)
        self.verbose = verbose

    def query_for(self, label: str) -> str:
        """Return the retrieval query string for an emotion label."""
        query = self._queries.get(label.lower(), f"{label.lower()} ambient atmosphere")
        if self.verbose:
            print(f"[Atmosphere] label={label!r}  query={query!r}")
        return query

    def fallback_for(self, label: str) -> dict:
        """Return a curated fallback description when no retrieval index exists."""
        return FALLBACK_AMBIENT.get(label.lower(),
                                    {"description": f"{label} ambient", "energy": "low"})


# ---------------------------------------------------------------------------
# Retrieval bridge — connects to Task 4
# ---------------------------------------------------------------------------

class RetrievalBridge:
    """
    Wraps the Task 4 retrieval system (retrieval_prototype.py) so Track B
    can query the existing RAVDESS-based index.

    Falls back gracefully to curated descriptions when the index is not built.

    Parameters
    ----------
    features_csv : str | None
        Path to task1_features_dataset.csv
    emotion_labels_json : str | None
        Path to emotion_labels.json
    top_k : int
        Number of candidates to retrieve per query.
    """

    def __init__(
        self,
        features_csv: Optional[str] = None,
        emotion_labels_json: Optional[str] = None,
        top_k: int = 3,
        verbose: bool = True,
    ):
        self.top_k = top_k
        self.verbose = verbose
        self._index = None
        self._search_fn = None
        self._print_fn = None

        root = Path(__file__).resolve().parent.parent
        csv  = Path(features_csv) if features_csv else root / "examples" / "task1_features_dataset.csv"
        json_path = Path(emotion_labels_json) if emotion_labels_json else root / "examples" / "emotion_labels.json"

        self._try_load_index(csv, json_path)

    def _try_load_index(self, csv: Path, json_path: Path) -> None:
        if not csv.exists():
            if self.verbose:
                print(f"[Atmosphere] Retrieval index not found at {csv}; using fallback descriptions.")
            return
        try:
            import sys
            p = str(csv.parent.parent / "task4_audio_retrieval")
            if p not in sys.path:
                sys.path.insert(0, p)
            from retrieval_prototype import build_index, search, print_results
            self._index = build_index(
                features_csv=str(csv),
                emotion_labels_json=str(json_path),
            )
            self._search_fn = search
            self._print_fn = print_results
            if self.verbose:
                print(f"[Atmosphere] Retrieval index loaded  ({len(self._index)} records)")
        except Exception as exc:
            if self.verbose:
                print(f"[Atmosphere] Could not load retrieval index ({exc}); using fallback.")

    def search(self, query: str) -> List[dict]:
        """
        Run a semantic query against the Task 4 index.
        Returns a list of result dicts, or an empty list on fallback.
        """
        if self._search_fn is None or self._index is None:
            return []
        try:
            return self._search_fn(query, self._index, top_k=self.top_k)
        except Exception:
            return []

    @property
    def available(self) -> bool:
        return self._index is not None


# ---------------------------------------------------------------------------
# Crossfade schedule
# ---------------------------------------------------------------------------

@dataclass
class CrossfadeSchedule:
    """
    Instruction for the audio engine to crossfade to a new atmosphere.

    Attributes
    ----------
    emotion_label : str
        The emotion that triggered this atmosphere change.
    query : str
        The search query used.
    suggested_clip : str
        Filename or description of the suggested ambient clip.
    suggested_description : str
        Human-readable description of the soundscape.
    fade_in_s : float
        Crossfade duration in seconds.
    lag_s : float
        Delay before the crossfade begins (natural dramatic lag).
    scheduled_at : float
        Wall-clock time.perf_counter() when this was created.
    retrieval_results : list
        Raw retrieval results (may be empty if using fallback).
    """
    emotion_label:        str
    query:                str
    suggested_clip:       str
    suggested_description: str
    fade_in_s:            float = 2.0
    lag_s:                float = 6.0
    scheduled_at:         float = field(default_factory=time.perf_counter)
    retrieval_results:    List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type":                 "atmosphere",
            "emotion_label":        self.emotion_label,
            "query":                self.query,
            "suggested_clip":       self.suggested_clip,
            "suggested_description": self.suggested_description,
            "fade_in_s":            self.fade_in_s,
            "lag_s":                self.lag_s,
        }

    def __repr__(self) -> str:
        return (
            f"CrossfadeSchedule(emotion={self.emotion_label!r}, "
            f"clip={self.suggested_clip!r}, "
            f"fade={self.fade_in_s}s, lag={self.lag_s}s)"
        )


class CrossfadeScheduler:
    """
    Decides whether to issue a new atmosphere change and builds the schedule.

    Avoids spamming changes by enforcing a minimum time between switches
    (cooldown_s).  Only issues a new schedule when the emotion label changes
    from the previous one.

    Parameters
    ----------
    lag_s : float
        Delay before crossfade begins (seconds).
    fade_s : float
        Crossfade duration (seconds).
    cooldown_s : float
        Minimum interval between atmosphere changes.
    verbose : bool
    """

    def __init__(
        self,
        lag_s: float = 6.0,
        fade_s: float = 2.0,
        cooldown_s: float = 8.0,
        verbose: bool = True,
    ):
        self.lag_s = lag_s
        self.fade_s = fade_s
        self.cooldown_s = cooldown_s
        self.verbose = verbose
        self._last_label: Optional[str] = None
        self._last_change_time: float = 0.0
        self._mapper = AtmosphereMapper(verbose=False)
        self._bridge = RetrievalBridge(verbose=verbose)

    def schedule(self, emotion_result) -> Optional[CrossfadeSchedule]:
        """
        Evaluate whether an atmosphere change is warranted for this emotion.

        Returns a CrossfadeSchedule if a change should happen, else None.

        Parameters
        ----------
        emotion_result : EmotionResult
        """
        label = emotion_result.label.lower()
        now   = time.perf_counter()

        # Suppress change if same emotion or cooldown not elapsed
        same_label    = (label == self._last_label)
        in_cooldown   = (now - self._last_change_time) < self.cooldown_s

        if same_label and in_cooldown:
            if self.verbose:
                print(f"[Atmosphere] No change (same={same_label}, cooldown={in_cooldown})")
            return None

        query = self._mapper.query_for(label)
        results = self._bridge.search(query)

        if results:
            top = results[0]
            clip        = top.get("filename", "unknown")
            description = top.get("description", query)
        else:
            fallback    = self._mapper.fallback_for(label)
            clip        = fallback["description"]
            description = fallback["description"]

        schedule = CrossfadeSchedule(
            emotion_label=label,
            query=query,
            suggested_clip=clip,
            suggested_description=description,
            fade_in_s=self.fade_s,
            lag_s=self.lag_s,
            retrieval_results=results,
        )

        self._last_label = label
        self._last_change_time = now

        if self.verbose:
            print(
                f"[Atmosphere] Schedule -> {description!r}  "
                f"(lag={self.lag_s}s, fade={self.fade_s}s)"
            )
        return schedule
