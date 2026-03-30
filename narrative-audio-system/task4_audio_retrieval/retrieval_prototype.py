"""Task 4: filter + semantic ranking retrieval for narrative audio queries."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

EMOTION_TO_NARRATIVE = {
    "Calm":      "calm narration",
    "Happy":     "upbeat cheerful dialogue",
    "Sad":       "sorrowful emotional narration",
    "Angry":     "urgent high-energy dramatic speech",
    "Fearful":   "tense suspenseful narration",
    "Disgust":   "intense dramatic emphasis",
    "Surprised": "excited emphatic dialogue",
    "Neutral":   "neutral flat narration",
}

ENERGY_THRESHOLDS = {"low": 0.02, "high": 0.06}
PITCH_THRESHOLDS  = {"low": 120.0, "high": 200.0}


def build_index(features_csv, emotion_labels_json=None):
    features_csv_path = Path(features_csv)
    if not features_csv_path.is_file():
        raise FileNotFoundError(
            f"Task 1 feature CSV not found at {features_csv_path}. "
            "Run Task 1 first to generate it."
        )

    file_rows = {}
    with features_csv_path.open("r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            file_rows.setdefault(row["filename"], []).append(row)

    emotion_map = {}
    if emotion_labels_json and Path(emotion_labels_json).is_file():
        with Path(emotion_labels_json).open("r", encoding="utf-8-sig") as fp:
            raw = json.load(fp)
        emotion_map = {k: v.strip().title() for k, v in raw.items()}

    records = []
    for filename, rows in file_rows.items():
        duration = sum(float(r["duration_seconds"]) for r in rows)
        energy   = float(np.mean([float(r["energy_rms_mean"]) for r in rows]))
        pitch    = float(np.mean([float(r["pitch_mean_hz"])   for r in rows]))

        energy_label = (
            "low"    if energy <  ENERGY_THRESHOLDS["low"]  else
            "high"   if energy >= ENERGY_THRESHOLDS["high"] else
            "medium"
        )
        pitch_label = (
            "low"    if pitch  <  PITCH_THRESHOLDS["low"]   else
            "high"   if pitch  >= PITCH_THRESHOLDS["high"]  else
            "medium"
        )

        emotion   = emotion_map.get(filename, "Unknown")
        narrative = EMOTION_TO_NARRATIVE.get(emotion, "speech")
        description = (
            f"{narrative}, {duration:.1f}s duration, "
            f"{energy_label}-energy, pitch {pitch:.0f}Hz"
        )

        records.append({
            "filename":     filename,
            "duration":     duration,
            "energy":       energy,
            "pitch":        pitch,
            "energy_label": energy_label,
            "pitch_label":  pitch_label,
            "emotion":      emotion,
            "narrative":    narrative,
            "description":  description,
        })

    return records


def _apply_filters(records, query):
    q = query.lower()
    filtered = list(records)

    m = re.search(r"longer\s+than\s+([\d.]+)\s*s", q)
    if m:
        filtered = [r for r in filtered if r["duration"] > float(m.group(1))]

    m = re.search(r"shorter\s+than\s+([\d.]+)\s*s", q)
    if m:
        filtered = [r for r in filtered if r["duration"] < float(m.group(1))]

    if re.search(r"\bhigh[- ]energy\b|\benerget\w+\b|\bloud\b", q):
        filtered = [r for r in filtered if r["energy_label"] == "high"]
    elif re.search(r"\bquiet\b|(?:^|\W)low[- ]energy\b|\bsoft\b|\bsubdued\b", q):
        filtered = [r for r in filtered if r["energy_label"] == "low"]

    for emotion_key in EMOTION_TO_NARRATIVE:
        if emotion_key.lower() in q:
            filtered = [r for r in filtered if r["emotion"] == emotion_key]
            break

    if re.search(r"\bhigh\s+pitch\b|\bhigh-pitched\b", q):
        filtered = [r for r in filtered if r["pitch_label"] == "high"]
    elif re.search(r"(?:^|\W)low\s+pitch\b|low-pitched\b|\bdeep\b", q):
        filtered = [r for r in filtered if r["pitch_label"] == "low"]

    return filtered


def _rank_semantically(query, candidates, top_k):
    if not candidates:
        return []

    descriptions = [r["description"] for r in candidates]

    if _ST_AVAILABLE:
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_emb = model.encode(descriptions, convert_to_tensor=True)
        query_emb  = model.encode(query,        convert_to_tensor=True)
        scores     = st_util.cos_sim(query_emb, corpus_emb)[0].tolist()
    else:
        vec    = TfidfVectorizer()
        matrix = vec.fit_transform(descriptions + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1])[0].tolist()

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{"score": round(s, 3), **r} for s, r in ranked[:top_k]]


def search(query, records, top_k=5):
    """Filter by structured constraints, then rank survivors semantically."""
    candidates = _apply_filters(records, query)
    if not candidates:
        candidates = records
    return _rank_semantically(query, candidates, top_k=top_k)


def print_results(query, results):
    print(f'\nQuery : "{query}"')
    if not results:
        print("  No matching recordings found.")
        return
    for i, r in enumerate(results, 1):
        score_str = f"  (score={r['score']:.3f})" if "score" in r else ""
        print(f"  {i}. {r['filename']}{score_str}\n     {r['description']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: Narrative Audio Retrieval")
    parser.add_argument("--features-csv",   default="../examples/task1_features_dataset.csv")
    parser.add_argument("--emotion-labels", default="../examples/emotion_labels.json")
    parser.add_argument("--query",          default=None, help="Custom query (optional)")
    parser.add_argument("--top-k",          type=int, default=5)
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    features_csv = (script_dir / args.features_csv).resolve()
    emotion_json = (script_dir / args.emotion_labels).resolve()

    print("Building retrieval index from Task 1 features ...")
    records = build_index(str(features_csv), str(emotion_json))
    print(f"Index contains {len(records)} recordings.\n")

    example_queries = [
        "calm narration longer than 4 seconds",
        "high-energy speech",
        "dramatic dialogue",
        "sad quiet voice",
        "angry short clip shorter than 3s",
    ]

    for q in ([args.query] if args.query else example_queries):
        results = search(q, records, top_k=args.top_k)
        print_results(q, results)
