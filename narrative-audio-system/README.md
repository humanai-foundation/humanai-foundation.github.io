# Narrative Audio System

This project is a GSoC 2026 test submission for a narrative-audio pipeline. It implements four required tasks and one bonus analysis task:

1. Audio processing and feature extraction
2. Narrative tone classification
3. AI-based transcription
4. Narrative audio retrieval
5. Bonus: storytelling audio analysis

The system takes a collection of speech recordings, converts them into structured features, classifies emotional or narrative tone, transcribes speech with Whisper, retrieves recordings from natural-language queries, and analyzes storytelling-oriented prosodic signals.

## Project Structure

- `run_pipeline.py`: end-to-end demo pipeline for all tasks
- `task1_audio_pipeline/audio_pipeline.py`: Task 1 audio preprocessing and feature extraction
- `task2_tone_classification/train_classifier.py`: Task 2 narrative tone classification
- `task3_transcription/whisper_transcriber.py`: Task 3 batch transcription and WER evaluation
- `task4_audio_retrieval/retrieval_prototype.py`: Task 4 hybrid retrieval prototype
- `task_bonus_storytelling/storytelling_analysis.py`: bonus storytelling-style analysis
- `examples/`: input audio, labels, and generated outputs
- `TableTalk Narrative Audio System_ Technical Report.pdf`: technical report for the submission

## Task Summary

### Task 1: Audio Processing Pipeline

The Task 1 pipeline:

1. Loads `.wav` files from a directory
2. Normalizes audio amplitude
3. Segments audio into fixed-duration chunks when needed
4. Extracts machine-learning-ready audio features

Extracted features include:

1. MFCC coefficients
2. Pitch
3. Spectral centroid
4. RMS energy
5. Duration

Output:

- `examples/task1_features_dataset.csv`
- `examples/normalized_audio/`

### Task 2: Narrative Tone Classification

The classifier uses MFCC-based features and a lightweight feedforward neural network to predict emotional tone labels. The training script performs a stratified train/test split, standardizes features using train-set statistics, and reports:

1. Test accuracy
2. Weighted F1 score
3. Per-class classification report

### Task 3: AI-Based Transcription

The transcription module uses OpenAI Whisper to:

1. Transcribe multiple audio recordings
2. Save transcripts in text format
3. Measure transcription quality with Word Error Rate on a small subset

Output:

- `examples/transcripts.txt`

### Task 4: Narrative Audio Retrieval

The retrieval system uses a hybrid approach:

1. Structured filtering for constraints such as duration, energy, pitch, and emotion
2. Semantic ranking over natural-language descriptions of recordings

Example queries:

1. `calm narration longer than 4 seconds`
2. `high-energy speech`
3. `dramatic dialogue`

### Bonus: Storytelling Audio Analysis

The bonus script analyzes several recordings using storytelling-oriented features:

1. Pacing and pauses
2. Pitch variation
3. Energy dynamics
4. Sentence length from transcripts

It also computes a heuristic `storytelling_score` and ranks clips by storytelling-like expressiveness.

Output:

- `examples/storytelling_analysis.csv`

## Installation

From the `narrative-audio-system/` directory:

```bash
pip install -r requirements.txt
```

## Run Instructions

### Run the full pipeline

From `narrative-audio-system/`:

```bash
python run_pipeline.py
```

### Run each task individually

Task 1:

```bash
python task1_audio_pipeline/audio_pipeline.py
```

Task 2:

```bash
python task2_tone_classification/train_classifier.py
```

Task 3:

```bash
python task3_transcription/whisper_transcriber.py
```

Task 4:

```bash
python task4_audio_retrieval/retrieval_prototype.py
```

Bonus task:

```bash
python task_bonus_storytelling/storytelling_analysis.py
```

## Example Outputs

Generated artifacts from the system include:

1. `examples/task1_features_dataset.csv`
2. `examples/normalized_audio/`
3. `examples/transcripts.txt`
4. `examples/storytelling_analysis.csv`

The console output also includes:

1. Task 2 evaluation metrics
2. Task 3 WER results
3. Task 4 retrieval results for sample queries
4. Bonus storytelling summary and top-ranked clips

## Approach and Discussion

This project was designed as a compact but complete prototype. The focus was on building an interpretable end-to-end system rather than optimizing each component independently.

- Task 1 converts raw audio into structured numerical features suitable for downstream modeling.
- Task 2 demonstrates that emotional or narrative tone can be predicted from MFCC-based features using a simple neural model.
- Task 3 shows that Whisper can provide practical batch transcription with measurable quality.
- Task 4 combines explicit metadata filters with semantic search to simulate narrative-oriented retrieval.
- The bonus task explores whether prosodic signals such as pauses, pitch variation, and energy range can help distinguish storytelling narration from ordinary conversational speech.

Current limitations include small dataset size, CPU transcription speed, and the heuristic nature of the bonus storytelling score. Future improvements could include pretrained audio embeddings, stronger retrieval models, and dedicated storytelling annotations.