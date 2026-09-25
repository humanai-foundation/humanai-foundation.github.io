# TableTalk Narrative Audio System

This repository contains the Technical Evaluation Test for the **GSoC 2026 HumanAI: TableTalk** project. It implements an end-to-end pipeline for processing, classifying, and retrieving narrative audio for interactive storytelling.

## 📄 Final Submission Documents
* **[Technical Report (PDF)](./TableTalk%20Narrative%20Audio%20System_%20Technical%20Report.pdf)** - Detailed analysis of methodology, results, and storytelling heuristics.
* **[Implementation Roadmap](./TableTalk%20Narrative%20Audio%20System_%20Technical%20Report.pdf#page=4)** - 12-week GSoC project plan.

---

## 🌟 Key Results
* **Task 3 (Transcription):** Achieved an average **Word Error Rate (WER) of 16.67%** using OpenAI Whisper.
* **Task 2 (Classification):** Successfully trained a neural model to **38.6% accuracy** on the RAVDESS subset, identifying key markers for "Calm" vs. "Fearful" tones.
* **Bonus Task:** Developed a storytelling detection heuristic using **Mean Pitch Variation (123.01 Hz)** and **Pause Ratios (0.952)**.

---

## 🛠️ Setup & Installation

### 1. System Requirements
This project requires **FFmpeg** for audio processing.
* **macOS:** `brew install ffmpeg`
* **Ubuntu/Linux:** `sudo apt install ffmpeg`
* **Windows:** Install via [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 2. Python Environment
```bash
# Clone the repository
git clone [YOUR_REPO_URL]
cd [REPO_NAME]

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `sounddevice` (added in Step 1) requires PortAudio.
> * **macOS:** `brew install portaudio`
> * **Ubuntu/Linux:** `sudo apt install portaudio19-dev`
> * **Windows:** included automatically via the pip wheel.

---

## Project Structure

- `run_pipeline.py`: End-to-end pipeline for all required tasks and bonus analysis
- `task0_audio_capture/audio_capture.py`: **Step 1** — real-time microphone capture & streaming
- `step2_vad/vad.py`: **Step 2** — Voice Activity Detection; labels frames as speech/silence and emits timestamped segments
- `task1_audio_pipeline/audio_pipeline.py`: Task 1 audio preprocessing and feature extraction
- `task2_tone_classification/train_classifier.py`: Task 2 tone classification model training and evaluation
- `task3_transcription/whisper_transcriber.py`: Task 3 batch transcription and WER measurement
- `task4_audio_retrieval/retrieval_prototype.py`: Task 4 retrieval prototype (filtering + semantic ranking)
- `task_bonus_storytelling/storytelling_analysis.py`: Bonus storytelling feature analysis and scoring
- `examples/`: Input recordings, labels, and generated output artifacts

---

## Task Summary

### Step 1: Audio Capture & Streaming

The capture module (`task0_audio_capture/audio_capture.py`) provides real-time microphone input as the first stage of the pipeline:

1. Opens a microphone stream via `sounddevice.InputStream` with a callback that fires every `chunk_size` samples
2. Each callback pushes raw PCM samples into a thread-safe `RollingBuffer` (circular deque)
3. Downstream steps read from the buffer without blocking audio capture

Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| Sample rate | 16 000 Hz | Standard for speech |
| Chunk size | 1 024 samples | 64 ms per frame |
| Max buffer | 30 s | Rolling window; oldest frames dropped |

Chunk size trade-offs: 512 samples = 32 ms (lower latency, higher CPU); 2048 samples = 128 ms (lower CPU, higher latency).

**Standalone usage:**

```bash
# Record 5 seconds and print stats
python task0_audio_capture/audio_capture.py --duration 5 --chunk 1024 --rate 16000

# Save captured audio to a WAV file
python task0_audio_capture/audio_capture.py --duration 5 --chunk 1024 --rate 16000 --save examples/captured_audio.wav

# List available microphone devices
python task0_audio_capture/audio_capture.py --list-devices
```

**Library usage:**

```python
from task0_audio_capture.audio_capture import AudioCaptureStream, RollingBuffer

buf = RollingBuffer(max_seconds=10, sample_rate=16000)
with AudioCaptureStream(sample_rate=16000, chunk_size=1024, buffer=buf) as stream:
    time.sleep(5)
audio = buf.get_audio()  # 1-D float32 numpy array
```

When `run_pipeline.py` is executed without an explicit audio path argument, Step 1 automatically captures 5 seconds from the microphone and passes the recording to Task 1 onwards. If no microphone is available the pipeline falls back to a pre-recorded file.

Primary output: `examples/captured_audio.wav`

---

### Step 2: Voice Activity Detection (VAD)

The VAD module (`step2_vad/vad.py`) labels every 10–30 ms audio frame as speech or non-speech and groups consecutive speech frames into timestamped `SpeechSegment` objects.  Only detected speech windows are forwarded to the transcriber — silence is never sent, which prevents wasted compute and mid-sentence cuts.

**Why it matters:** Whisper and similar models need clean speech boundaries to produce coherent transcripts.  Sending raw silence causes hallucinations or empty outputs; cutting too early truncates words.

**Backends (auto-selected)**

| Backend | Latency | Accuracy | Requirement |
|---------|---------|----------|-------------|
| `webrtcvad` | < 1 ms/frame | Good (rule-based) | `pip install webrtcvad` |
| `silero-vad` | ~ 5 ms/frame | Better (neural) | `torch` + internet (auto-download) |
| `energy` | < 0.1 ms/frame | Basic fallback | none |

**Key parameters**

| Parameter | Default | Effect |
|-----------|---------|--------|
| Frame duration | 20 ms | VAD resolution (10/20/30 ms) |
| Aggressiveness | 2 | webrtcvad mode 0–3 (3 = most aggressive noise filtering) |
| Speech pad | 300 ms | Pre-speech buffer to avoid clipping first syllable |
| Silence pad | 400 ms | Post-speech tolerance — allows brief pauses within an utterance |
| Min speech | 250 ms | Discard clicks and noise bursts shorter than this |

**Example output**
```
[VAD] Speech started  at T =   1.200 s
[VAD] Speech ended    at T =   3.840 s  (duration 2.640 s)
[VAD] Detected 1 speech segment(s)
[VAD] Total speech : 2.640 s / 5.056 s (52.2%)
      [ 1]   1.200s ->   3.840s  (2.640 s)
```

**Standalone usage:**

```bash
# Run VAD on a WAV file
python step2_vad/vad.py --input examples/captured_audio.wav --mode 2 --frame-ms 20

# Run VAD on live microphone input (10 seconds)
python step2_vad/vad.py --duration 10 --mode 2

# Force a specific backend
python step2_vad/vad.py --input audio.wav --backend webrtcvad
```

**Library usage:**

```python
from step2_vad.vad import detect_speech_segments

segments = detect_speech_segments(audio_array, sample_rate=16000, aggressiveness=2)
for seg in segments:
    print(f"Speech {seg.start:.2f}s - {seg.end:.2f}s  ({seg.duration:.2f}s)")
    # seg.audio is the float32 PCM for that utterance
```

In `run_pipeline.py`, Step 2 runs immediately after Step 1 and prints a per-segment breakdown before passing audio to Task 1.

---

### Task 1: Audio Processing Pipeline

The Task 1 pipeline:

1. Loads `.wav` files from the input directory
2. Normalizes audio amplitude for consistent feature extraction
3. Segments audio into fixed windows when needed
4. Extracts machine-learning-ready features

Extracted features include:

1. MFCC coefficients
2. Pitch (fundamental frequency summary)
3. Spectral centroid
4. RMS energy
5. Duration

Primary outputs:

- `examples/task1_features_dataset.csv`
- `examples/normalized_audio/`

### Task 2: Narrative Tone Classification

The classifier uses MFCC-based features and a feedforward neural network to predict emotional tone labels. The training pipeline includes:

1. Stratified train/test split
2. Feature standardization using train-set statistics
3. Neural model training with cross-entropy loss
4. Test evaluation

Reported metrics:

1. Accuracy
2. Weighted F1 score
3. Per-class report

### Task 3: AI-Based Transcription

The transcription module uses OpenAI Whisper to:

1. Transcribe multiple recordings in batch
2. Save transcripts to a text output file
3. Measure transcription quality with Word Error Rate (WER) on a subset

Primary output:

- `examples/transcripts.txt`

### Task 4: Narrative Audio Retrieval (TableTalk Simulation)

The retrieval system uses a hybrid strategy:

1. Structured filtering from query constraints (duration, energy, pitch, tone)
2. Semantic ranking over generated recording descriptions

Example queries:

1. `calm narration longer than 4 seconds`
2. `high-energy speech`
3. `dramatic dialogue`

### Bonus: Storytelling Audio Analysis

The bonus module analyzes several recordings for storytelling-related cues:

1. Pacing and pauses
2. Pitch variation
3. Energy dynamics
4. Sentence-length characteristics from transcripts

It also computes a heuristic `storytelling_score` and ranks clips by storytelling-like expressiveness.

Primary output:

- `examples/storytelling_analysis.csv`

---

## Run Instructions

### Run the full pipeline

From the `narrative-audio-system/` directory:

```bash
python run_pipeline.py examples/03-01-04-02-01-01-11.wav
```

### Run tasks individually

Step 1 (audio capture):

```bash
python task0_audio_capture/audio_capture.py --duration 5 --chunk 1024 --rate 16000 --save examples/captured_audio.wav
```

Step 2 (voice activity detection):

```bash
python step2_vad/vad.py --input examples/captured_audio.wav --mode 2 --frame-ms 20
```

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

---

## Example Outputs

Generated artifacts include:

1. `examples/captured_audio.wav` (Step 1 — live microphone recording)
2. `examples/task1_features_dataset.csv`
3. `examples/normalized_audio/`
4. `examples/transcripts.txt`
5. `examples/storytelling_analysis.csv`

Console outputs include:

1. Task 2 test metrics (accuracy, weighted F1, class report)
2. Task 3 WER summary
3. Task 4 retrieval results for sample queries
4. Bonus storytelling summary and top-ranked clips

---

## Approach and Discussion

This project is designed as a practical, reproducible end-to-end prototype for narrative audio processing.

- Task 1 converts raw recordings into structured numerical features.
- Task 2 demonstrates tone classification from audio-derived features.
- Task 3 provides scalable transcription with measurable quality.
- Task 4 combines explicit filtering with semantic retrieval for narrative-style queries.
- The bonus task explores prosodic and transcript-level cues for distinguishing storytelling narration from conversational speech.

Current limitations include dataset scale, CPU transcription speed, and the heuristic nature of storytelling scoring. Future improvements include pretrained audio embeddings, stronger ranking objectives, and dedicated storytelling annotations.