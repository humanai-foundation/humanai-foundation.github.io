"""
Step 5 — Tone / Emotion Classification
========================================
Classifies the emotional tone of an Utterance in parallel with Step 4
transcription.  Both steps consume the same audio buffer from Step 3 and
are dispatched concurrently via ThreadPoolExecutor so neither waits on the
other.

Pipeline position
-----------------
  [Step 3 Utterance]
       |
       +-----> Step 4 Transcriber  (Whisper, ~300–600 ms)
       |
       +-----> Step 5 Classifier   (MFCC + MLP, ~5–15 ms)  <-- this module
       |
  [merged result: text + emotion]

Backends
--------
  mfcc-mlp   (default)
    Extract 13 MFCC coefficients with librosa (~5 ms), run through the
    2-layer feedforward network from Task 2 (~10 ms on CPU).  Fast enough
    to never be the bottleneck.

  wav2vec2 / hubert   (research direction)
    Fine-tuning on wav2vec2 or HuBERT embeddings generalises far better to
    real microphone audio than raw MFCCs.  These embeddings capture
    phonetic and prosodic context across a full utterance.  Recommended
    for production use with Matthew.
    Not loaded by default — enable with backend="wav2vec2" and
    pip install transformers.

Output
------
  EmotionResult(label="tense", confidence=0.82,
                start=1.2, end=3.84, latency_ms=12.4)

Usage (standalone demo)
-----------------------
  python emotion_classifier/classifier.py --input examples/captured_audio.wav
  python emotion_classifier/classifier.py --train   # retrain on RAVDESS

Usage (library)
---------------
  from emotion_classifier.classifier import EmotionClassifier, EmotionResult

  clf = EmotionClassifier()
  result = clf.classify(utterance)
  print(result.label, result.confidence)

Parallel usage with Step 4
---------------------------
  from concurrent.futures import ThreadPoolExecutor
  from emotion_classifier.classifier import EmotionClassifier
  from transcriber.streaming_transcriber import Transcriber

  clf = EmotionClassifier()
  trs = Transcriber()

  with ThreadPoolExecutor(max_workers=2) as pool:
      t_future = pool.submit(trs.transcribe, utterance)
      e_future = pool.submit(clf.classify, utterance)
      transcript = t_future.result()
      emotion    = e_future.result()
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EmotionResult:
    """
    Tone / emotion prediction for one utterance.

    Attributes
    ----------
    label : str
        Predicted emotion label (e.g. "calm", "angry", "tense").
    confidence : float
        Softmax probability of the top prediction (0–1).
    start : float
        Utterance start time in seconds.
    end : float
        Utterance end time in seconds.
    latency_ms : float
        Wall-clock time for feature extraction + inference (ms).
    all_scores : dict
        Softmax probability for every class: {"calm": 0.7, "angry": 0.2, ...}
    backend : str
        Which backend produced this result.
    """
    label: str
    confidence: float
    start: float = 0.0
    end: float = 0.0
    latency_ms: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    backend: str = "mfcc-mlp"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f'EmotionResult(label={self.label!r}, confidence={self.confidence:.2f}, '
            f'start={self.start:.3f}s, end={self.end:.3f}s, '
            f'latency={self.latency_ms:.1f}ms, backend={self.backend!r})'
        )


# ---------------------------------------------------------------------------
# Neural network (matches Task 2 architecture)
# ---------------------------------------------------------------------------

class _ToneNet(nn.Module):
    """2-layer feedforward classifier — identical to Task 2 ToneClassifier."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# MFCC feature extraction
# ---------------------------------------------------------------------------

def _extract_mfcc(audio: np.ndarray, sample_rate: int = 16000,
                  n_mfcc: int = 13) -> np.ndarray:
    """Return mean MFCC vector (shape: [n_mfcc]) from a float32 audio array."""
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa not installed.  Run: pip install librosa")
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32),
                                 sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class EmotionClassifier:
    """
    Classifies emotional tone from a raw audio array or Utterance object.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a saved checkpoint (.pt) produced by save().  If None and
        train_on_init=True, the model is trained on the RAVDESS examples.
    label_map_path : str | Path | None
        Path to emotion_labels.json.  Defaults to examples/emotion_labels.json.
    backend : str
        "mfcc-mlp"  — fast MFCC + MLP (default).
        "wav2vec2"  — requires transformers; better on real mic audio.
        "hubert"    — requires transformers; similar to wav2vec2.
    n_mfcc : int
        Number of MFCC coefficients (must match trained model).
    hidden_dim : int
        MLP hidden layer width.
    sample_rate : int
        Expected audio sample rate.
    train_on_init : bool
        If True and no model_path, auto-train on RAVDESS examples at startup.
    verbose : bool
        Print status messages.
    """

    RAVDESS_EMOTIONS = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        backend: str = "mfcc-mlp",
        n_mfcc: int = 13,
        hidden_dim: int = 64,
        sample_rate: int = 16000,
        train_on_init: bool = True,
        verbose: bool = True,
    ):
        if backend not in ("mfcc-mlp", "wav2vec2", "hubert"):
            raise ValueError(f"Unknown backend {backend!r}. "
                             "Choose 'mfcc-mlp', 'wav2vec2', or 'hubert'.")

        self.backend = backend
        self.n_mfcc = n_mfcc
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.verbose = verbose

        self._model = None
        self._class_names: List[str] = []
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._transformer_pipeline = None

        # Resolve label map path
        root = Path(__file__).resolve().parent.parent
        self._label_map_path = Path(label_map_path) if label_map_path else (
            root / "examples" / "emotion_labels.json"
        )

        if backend == "mfcc-mlp":
            self._init_mfcc_mlp(model_path, train_on_init)
        else:
            self._init_transformer(backend)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_mfcc_mlp(self, model_path, train_on_init: bool) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed.  Run: pip install torch")

        if model_path and Path(model_path).exists():
            self._load_checkpoint(model_path)
        elif train_on_init and self._label_map_path.exists():
            self._train_from_examples()
        elif train_on_init:
            if self.verbose:
                print(f"[Classifier] label map not found at {self._label_map_path}; "
                      "using untrained model with RAVDESS class names.")
            self._class_names = self.RAVDESS_EMOTIONS
            self._feature_mean = np.zeros(self.n_mfcc, dtype=np.float32)
            self._feature_std  = np.ones(self.n_mfcc,  dtype=np.float32)
            self._model = _ToneNet(self.n_mfcc, self.hidden_dim, len(self._class_names))
        else:
            raise RuntimeError("No model_path provided and train_on_init=False.")

    def _init_transformer(self, backend: str) -> None:
        """
        Research direction: wav2vec2 / HuBERT embeddings.

        These models capture rich phonetic and prosodic context and
        generalise far better to real microphone audio than raw MFCCs.
        Recommended for production with real performer audio.
        """
        try:
            from transformers import pipeline as hf_pipeline
            task = "audio-classification"
            model_id = (
                "facebook/wav2vec2-base"
                if backend == "wav2vec2"
                else "facebook/hubert-base-ls960"
            )
            self._transformer_pipeline = hf_pipeline(
                task, model=model_id, device=-1
            )
            self._class_names = []   # set by the HF model
            if self.verbose:
                print(f"[Classifier] Loaded {backend} pipeline ({model_id})")
        except ImportError:
            raise ImportError(
                f"transformers not installed for backend={backend!r}.\n"
                "  pip install transformers\n"
                "Note: wav2vec2/HuBERT are research-direction backends — "
                "use mfcc-mlp for production."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_from_examples(self) -> None:
        import json
        from sklearn.model_selection import train_test_split

        if self.verbose:
            print(f"[Classifier] Training on {self._label_map_path} ...")

        with open(self._label_map_path, "r", encoding="utf-8-sig") as f:
            label_map = json.load(f)

        audio_root = self._label_map_path.parent
        features, labels = [], []
        for filename, emotion in label_map.items():
            wav = audio_root / filename
            if not wav.is_file():
                continue
            try:
                import soundfile as sf
                audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr,
                                             target_sr=self.sample_rate)
                vec = _extract_mfcc(audio, self.sample_rate, self.n_mfcc)
                features.append(vec)
                labels.append(emotion.strip().lower())
            except Exception:
                continue

        if not features:
            raise RuntimeError("No features could be extracted from examples/.")

        unique, encoded = np.unique(labels, return_inverse=True)
        self._class_names = [str(c).strip().title() for c in unique.tolist()]

        X = np.array(features, dtype=np.float32)
        self._feature_mean = X.mean(axis=0)
        self._feature_std  = X.std(axis=0) + 1e-6
        X_norm = (X - self._feature_mean) / self._feature_std

        if len(set(labels)) > 1:
            X_tr, _, y_tr, _ = train_test_split(
                X_norm, encoded, test_size=0.2, random_state=42, stratify=encoded
            )
        else:
            X_tr, y_tr = X_norm, encoded

        self._model = _ToneNet(self.n_mfcc, self.hidden_dim, len(self._class_names))
        criterion  = torch.nn.CrossEntropyLoss()
        optimizer  = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        X_tensor   = torch.tensor(X_tr, dtype=torch.float32)
        y_tensor   = torch.tensor(y_tr, dtype=torch.long)

        self._model.train()
        for epoch in range(30):
            logits = self._model(X_tensor)
            loss   = criterion(logits, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._model.eval()
        if self.verbose:
            print(
                f"[Classifier] Trained on {len(features)} samples "
                f"({len(self._class_names)} classes): "
                f"{', '.join(self._class_names)}"
            )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights + normalisation stats to a .pt file."""
        torch.save({
            "model_state": self._model.state_dict(),
            "class_names": self._class_names,
            "feature_mean": self._feature_mean,
            "feature_std":  self._feature_std,
            "n_mfcc":       self.n_mfcc,
            "hidden_dim":   self.hidden_dim,
        }, path)
        if self.verbose:
            print(f"[Classifier] Saved to {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self._class_names  = ckpt["class_names"]
        self._feature_mean = ckpt["feature_mean"]
        self._feature_std  = ckpt["feature_std"]
        n_mfcc      = ckpt.get("n_mfcc", self.n_mfcc)
        hidden_dim  = ckpt.get("hidden_dim", self.hidden_dim)
        self._model = _ToneNet(n_mfcc, hidden_dim, len(self._class_names))
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        if self.verbose:
            print(f"[Classifier] Loaded checkpoint from {path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify_array(
        self, audio: np.ndarray, sample_rate: int = 16000,
        start: float = 0.0, end: float = 0.0,
    ) -> EmotionResult:
        """
        Classify a raw float32 PCM array.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array at `sample_rate`.
        sample_rate : int
        start, end : float
            Source timestamps for result metadata.

        Returns
        -------
        EmotionResult
        """
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)

        if sample_rate != self.sample_rate and _LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=self.sample_rate)

        t0 = time.perf_counter()

        if self.backend == "mfcc-mlp":
            label, confidence, all_scores = self._infer_mfcc_mlp(audio)
        else:
            label, confidence, all_scores = self._infer_transformer(audio)

        latency_ms = (time.perf_counter() - t0) * 1000

        result = EmotionResult(
            label=label,
            confidence=confidence,
            start=start,
            end=end,
            latency_ms=latency_ms,
            all_scores=all_scores,
            backend=self.backend,
        )

        if self.verbose:
            print(
                f"[Classifier] {start:.3f}s -> {end:.3f}s  "
                f"({latency_ms:.1f}ms)  "
                f"{label!r}  conf={confidence:.2f}"
            )
        return result

    def classify(self, utterance) -> EmotionResult:
        """
        Classify an Utterance object (from Step 3).

        Parameters
        ----------
        utterance : Utterance
            Must have .audio, .start, .end attributes.

        Returns
        -------
        EmotionResult
        """
        return self.classify_array(
            audio=utterance.audio,
            sample_rate=self.sample_rate,
            start=utterance.start,
            end=utterance.end,
        )

    def _infer_mfcc_mlp(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        vec = _extract_mfcc(audio, self.sample_rate, self.n_mfcc)
        norm = (vec - self._feature_mean) / self._feature_std
        tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()
        idx = int(np.argmax(probs))
        label = self._class_names[idx]
        all_scores = {name: float(probs[i])
                      for i, name in enumerate(self._class_names)}
        return label, float(probs[idx]), all_scores

    def _infer_transformer(self, audio: np.ndarray) -> Tuple[str, float, Dict]:
        results = self._transformer_pipeline(
            {"raw": audio, "sampling_rate": self.sample_rate}
        )
        top = max(results, key=lambda r: r["score"])
        all_scores = {r["label"]: r["score"] for r in results}
        return top["label"], top["score"], all_scores


# ---------------------------------------------------------------------------
# Parallel processor — Step 4 + Step 5 concurrently
# ---------------------------------------------------------------------------

class ParallelProcessor:
    """
    Runs Step 4 (Transcriber) and Step 5 (EmotionClassifier) concurrently
    on the same Utterance using a 2-thread pool.

    Both steps read the utterance's audio array independently — no locking
    needed since the array is read-only during inference.

    Parameters
    ----------
    transcriber : Transcriber
        A loaded Step 4 Transcriber instance.
    classifier : EmotionClassifier
        A loaded Step 5 EmotionClassifier instance.
    verbose : bool
        Print combined results to stdout.
    """

    def __init__(self, transcriber, classifier, verbose: bool = True):
        self._transcriber = transcriber
        self._classifier  = classifier
        self.verbose = verbose

    def process(self, utterance):
        """
        Run transcription and classification in parallel.

        Returns
        -------
        (TranscriptionResult, EmotionResult)
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            t_future = pool.submit(self._transcriber.transcribe, utterance)
            e_future = pool.submit(self._classifier.classify,    utterance)
            transcript = t_future.result()
            emotion    = e_future.result()

        if self.verbose:
            print(
                f"[Parallel] {utterance.start:.3f}s -> {utterance.end:.3f}s\n"
                f"  text     : \"{transcript.text}\"\n"
                f"  emotion  : {emotion.label!r}  ({emotion.confidence:.2f})\n"
                f"  latencies: transcribe={transcript.latency_ms:.0f}ms  "
                f"classify={emotion.latency_ms:.1f}ms"
            )
        return transcript, emotion

    def process_all(self, utterances) -> List[Tuple]:
        """Process every utterance and return [(TranscriptionResult, EmotionResult)]."""
        return [self.process(u) for u in utterances]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def classify_utterances(
    utterances,
    label_map_path: Optional[str] = None,
    verbose: bool = True,
) -> List[EmotionResult]:
    """
    One-call helper: classify a list of Utterances.

    Returns
    -------
    List[EmotionResult]
    """
    clf = EmotionClassifier(
        label_map_path=label_map_path,
        verbose=verbose,
    )
    return [clf.classify(u) for u in utterances]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5 — Tone/Emotion Classification demo"
    )
    parser.add_argument("--input", default=None, metavar="FILE.WAV",
                        help="WAV file to classify (default: live mic)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Live recording duration in seconds (default: 10)")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--backend", default="mfcc-mlp",
                        choices=["mfcc-mlp", "wav2vec2", "hubert"],
                        help="Classification backend (default: mfcc-mlp)")
    parser.add_argument("--model", default=None, metavar="CHECKPOINT.pt",
                        help="Path to saved classifier checkpoint")
    parser.add_argument("--train", action="store_true",
                        help="Retrain on RAVDESS examples and save to classifier.pt")
    parser.add_argument("--vad-mode", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--parallel", action="store_true",
                        help="Run Step 4 transcription + Step 5 in parallel")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    for mod in ("vad_engine", "utterance_buffer", "task0_audio_capture", "transcriber"):
        p = str(root / mod)
        if p not in sys.path:
            sys.path.insert(0, p)

    from vad import detect_speech_segments
    from segmenter import segment_utterances

    print(
        f"\nStep 5 — Tone / Emotion Classification\n"
        f"  Backend   : {args.backend}\n"
        f"  Model     : {args.model or 'auto-train from examples/'}\n"
        f"  Parallel  : {args.parallel}\n"
    )

    # Load or capture audio
    if args.input:
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.rate)
        print(f"[Classifier] Loaded {args.input}  ({len(audio)/args.rate:.2f} s)\n")
    else:
        from audio_capture import record_for_duration
        print(f"[Classifier] Recording {args.duration:.1f} s ...")
        audio = record_for_duration(duration=args.duration, sample_rate=args.rate)

    # VAD + segmentation
    vad_segs = detect_speech_segments(
        audio, sample_rate=args.rate, aggressiveness=args.vad_mode, verbose=False
    )
    utterances = segment_utterances(
        vad_segs, strategy="pause_triggered", sample_rate=args.rate, verbose=False
    )
    print(f"[Classifier] {len(utterances)} utterance(s)\n")

    # Train/save if requested
    clf = EmotionClassifier(
        model_path=args.model,
        backend=args.backend,
        verbose=True,
    )
    if args.train:
        clf.save("classifier.pt")

    if args.parallel:
        from streaming_transcriber import Transcriber
        from parallel_processor import ParallelProcessor
        trs = Transcriber(model_size="tiny", verbose=False)
        proc = ParallelProcessor(trs, clf, verbose=True)
        for utt in utterances:
            proc.process(utt)
    else:
        for utt in utterances:
            clf.classify(utt)


if __name__ == "__main__":
    main()
