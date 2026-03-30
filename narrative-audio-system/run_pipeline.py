import whisper
import librosa
import soundfile as sf
import shutil
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _add_module_path(folder_name):
    module_path = str(PROJECT_ROOT / folder_name)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


_add_module_path("task1_audio_pipeline")
from audio_pipeline import build_feature_dataset

_add_module_path("task3_transcription")
from whisper_transcriber import transcribe_directory, measure_accuracy

_add_module_path("task4_audio_retrieval")
from retrieval_prototype import build_index as build_retrieval_index, search as retrieval_search, print_results as print_retrieval_results

_add_module_path("task_bonus_storytelling")
from storytelling_analysis import analyze_storytelling, discuss_storytelling_signals
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


def process_audio(input_path, output_path, sample_rate=16000, n_mfcc=13):
    waveform, loaded_sample_rate = librosa.load(input_path, sr=sample_rate)
    mfcc_features = librosa.feature.mfcc(y=waveform, sr=loaded_sample_rate, n_mfcc=n_mfcc)
    print("Task 1: MFCC feature shape:", mfcc_features.shape)
    sf.write(output_path, waveform, loaded_sample_rate)
    return mfcc_features


def extract_mfcc_vector(audio_path, sample_rate=16000, n_mfcc=13):
    waveform, loaded_sample_rate = librosa.load(str(audio_path), sr=sample_rate)
    mfcc_matrix = librosa.feature.mfcc(y=waveform, sr=loaded_sample_rate, n_mfcc=n_mfcc)
    return mfcc_matrix.mean(axis=1)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_features):
        return self.net(input_features)


def train_classifier(features, labels, num_epochs=30, test_size=0.2, random_seed=42):
    label_array = np.asarray(labels)
    unique_classes, encoded_labels = np.unique(label_array, return_inverse=True)
    class_names = [str(n).strip().title() for n in unique_classes.tolist()]
    print("Task 2: Emotions:", ", ".join(class_names))

    feature_array = np.asarray(features, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_array, encoded_labels, test_size=test_size,
        random_state=random_seed, stratify=encoded_labels
    )

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    classifier_model = SimpleClassifier(X_train.shape[1], hidden_dim=64, num_classes=len(unique_classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=1e-3)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    label_tensor = torch.tensor(y_train, dtype=torch.long)

    classifier_model.train()
    for epoch in range(num_epochs):
        logits = classifier_model(train_tensor)
        loss = criterion(logits, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in {0, num_epochs - 1} or (epoch + 1) % 10 == 0:
            preds = torch.argmax(logits, dim=1).numpy()
            acc = accuracy_score(y_train, preds)
            f1 = f1_score(y_train, preds, average="weighted", zero_division=0)
            print(f"Task 2 - Epoch {epoch + 1:3d}: loss={loss.item():.4f}  train_acc={acc:.3f}  train_f1={f1:.3f}")

    # Held-out test evaluation
    classifier_model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(classifier_model(torch.tensor(X_test_norm, dtype=torch.float32)), dim=1).numpy()
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    print(f"Task 2 - Test Accuracy: {test_acc:.3f}  Weighted F1: {test_f1:.3f}")
    print("Task 2 - Per-class report:")
    print(classification_report(y_test, test_preds, target_names=class_names, zero_division=0))

    return classifier_model, unique_classes, feature_mean, feature_std


def predict_emotion(model, class_names, feature_mean, feature_std, mfcc_vector):
    standardized_vector = (np.asarray(mfcc_vector, dtype=np.float32) - feature_mean) / feature_std
    input_tensor = torch.tensor(standardized_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = int(torch.argmax(logits, dim=1).item())
    return str(class_names[predicted_index]).strip().title()


def load_labeled_mfcc_features(label_map_path, sample_rate=16000, n_mfcc=13):
    with Path(label_map_path).open("r", encoding="utf-8-sig") as fp:
        filename_to_emotion = json.load(fp)

    feature_rows = []
    emotion_labels = []
    audio_root = Path(label_map_path).parent

    for filename, emotion in filename_to_emotion.items():
        audio_path = audio_root / filename
        if not audio_path.is_file():
            continue

        mfcc_vector = extract_mfcc_vector(audio_path, sample_rate=sample_rate, n_mfcc=n_mfcc)
        feature_rows.append(mfcc_vector)
        emotion_labels.append(emotion)

    if not feature_rows:
        raise ValueError(f"No labeled audio features could be loaded from {label_map_path}")

    return np.asarray(feature_rows, dtype=np.float32), emotion_labels


def transcribe_audio(input_path, model_size="tiny"):
    if shutil.which("ffmpeg") is None:
        fallback_transcript = "Transcription unavailable because ffmpeg is not installed."
        print("Task 3: Transcription (fallback):", fallback_transcript)
        return fallback_transcript

    whisper_model = whisper.load_model(model_size)
    try:
        transcription_result = whisper_model.transcribe(input_path)
        transcript_text = transcription_result["text"]
        print("Task 3: Transcription:", transcript_text)
        return transcript_text
    except FileNotFoundError:
        fallback_transcript = "Transcription unavailable because ffmpeg is not installed."
        print("Task 3: Transcription (fallback):", fallback_transcript)
        return fallback_transcript



if __name__ == "__main__":
    input_audio_path = "examples/sample_audio.wav"
    if len(sys.argv) == 2:
        arg_path = Path(sys.argv[1])
        if arg_path.parent == Path("."):
            input_audio_path = str(Path("examples") / arg_path.name)
        else:
            input_audio_path = str(arg_path)

    processed_audio_path = "examples/processed_audio.wav"

    task1_output_csv = Path("examples/task1_features_dataset.csv")
    task1_normalized_dir = Path("examples/normalized_audio")
    print("Task 1: running full audio feature extraction pipeline...")
    build_feature_dataset(
        input_dir="examples",
        output_csv=str(task1_output_csv),
        normalized_dir=str(task1_normalized_dir),
    )

    process_audio(input_audio_path, processed_audio_path)

    task3_transcript_file = Path("examples/transcripts.txt")
    print("\nTask 3: transcribing recordings (first 10 files) ...")
    all_transcripts = transcribe_directory(
        input_dir="examples",
        output_txt=str(task3_transcript_file),
        model_size="tiny",
        max_files=10,
    )
    measure_accuracy(all_transcripts, max_samples=10)

    input_filename = Path(input_audio_path).name
    transcript_text = all_transcripts.get(input_filename) or transcribe_audio(input_audio_path)

    label_map_file = Path("examples") / "emotion_labels.json"
    predicted_input_emotion = "Unknown"
    if label_map_file.is_file():
        training_features, training_labels = load_labeled_mfcc_features(label_map_file)
        print(
            f"Task 2: loaded {len(training_labels)} labeled samples across "
            f"{len(set(training_labels))} emotions from {label_map_file}."
        )
        trained_model, class_names, feature_mean, feature_std = train_classifier(training_features, training_labels)
        input_mfcc_vector = extract_mfcc_vector(input_audio_path)
        predicted_input_emotion = predict_emotion(
            trained_model,
            class_names,
            feature_mean,
            feature_std,
            input_mfcc_vector,
        )
        print(f"Task 2: Predicted emotion for input audio: {predicted_input_emotion}")
    else:
        print(f"Task 2: label map not found at {label_map_file}, skipping classifier training.")

    print("\nTask 4: building retrieval index from audio features ...")
    retrieval_records = build_retrieval_index(
        features_csv=str(task1_output_csv),
        emotion_labels_json=str(label_map_file),
    )
    print(f"Task 4: index contains {len(retrieval_records)} recordings.")
    for query in [
        "calm narration longer than 4 seconds",
        "high-energy speech",
        "dramatic dialogue",
    ]:
        results = retrieval_search(query, retrieval_records, top_k=3)
        print_retrieval_results(query, results)

    print("\nBonus: storytelling audio analysis on selected recordings ...")
    bonus_rows = analyze_storytelling(
        input_dir="examples",
        output_csv="examples/storytelling_analysis.csv",
        max_files=8,
        model_size="tiny",
    )
    discuss_storytelling_signals(bonus_rows)
