import whisper
import librosa
import soundfile as sf
import shutil
import sys
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def process_audio(input_path, output_path, sample_rate=16000, n_mfcc=13):
    # Keep audio at a fixed sample rate for reproducible feature extraction.
    waveform, loaded_sample_rate = librosa.load(input_path, sr=sample_rate)
    mfcc_features = librosa.feature.mfcc(y=waveform, sr=loaded_sample_rate, n_mfcc=n_mfcc)
    print("Task 1: MFCC feature shape:", mfcc_features.shape)
    sf.write(output_path, waveform, loaded_sample_rate)
    return mfcc_features


def extract_mfcc_vector(audio_path, sample_rate=16000, n_mfcc=13):
    waveform, loaded_sample_rate = librosa.load(str(audio_path), sr=sample_rate)
    mfcc_matrix = librosa.feature.mfcc(y=waveform, sr=loaded_sample_rate, n_mfcc=n_mfcc)
    # Convert variable-length MFCC sequence to fixed-size vector.
    return mfcc_matrix.mean(axis=1)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, input_features):
        return self.classifier(input_features)


def train_classifier(features, labels, num_epochs=20):
    # Map any incoming label values to contiguous class ids required by CrossEntropyLoss.
    label_array = np.asarray(labels)
    unique_classes, encoded_labels = np.unique(label_array, return_inverse=True)
    emotion_names = [str(name).strip().title() for name in unique_classes.tolist()]
    print("Task 2: Emotions:", ", ".join(emotion_names))

    feature_array = np.asarray(features, dtype=np.float32)
    feature_mean = feature_array.mean(axis=0)
    feature_std = feature_array.std(axis=0) + 1e-6
    standardized_features = (feature_array - feature_mean) / feature_std

    classifier_model = SimpleClassifier(standardized_features.shape[1], len(unique_classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=0.01)

    feature_tensor = torch.tensor(standardized_features, dtype=torch.float32)
    label_tensor = torch.tensor(encoded_labels, dtype=torch.long)

    for epoch in range(num_epochs):
        logits = classifier_model(feature_tensor)
        loss = criterion(logits, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_labels = torch.argmax(logits, dim=1).numpy()
        accuracy = accuracy_score(encoded_labels, predicted_labels)
        weighted_f1 = f1_score(encoded_labels, predicted_labels, average="weighted")
        if epoch in {0, num_epochs - 1} or (epoch + 1) % 5 == 0:
            print(f"Task 2 - Epoch {epoch + 1}: Loss={loss.item():.4f}, Acc={accuracy:.3f}, F1={weighted_f1:.3f}")

    return classifier_model, unique_classes, feature_mean, feature_std


def predict_emotion(model, class_names, feature_mean, feature_std, mfcc_vector):
    standardized_vector = (np.asarray(mfcc_vector, dtype=np.float32) - feature_mean) / feature_std
    input_tensor = torch.tensor(standardized_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = int(torch.argmax(logits, dim=1).item())
    return str(class_names[predicted_index]).strip().title()


def load_labeled_mfcc_features(label_map_path, sample_rate=16000, n_mfcc=13):
    # utf-8-sig transparently handles files saved with a UTF-8 BOM.
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
    # Whisper depends on ffmpeg for audio decoding; keep pipeline runnable with fallback text.
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


def retrieve_best_match(candidate_documents, query_text, min_similarity=0.05):
    # Fit on docs + query so they share one TF-IDF feature space.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(candidate_documents + [query_text])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_index = similarity_scores.argmax()
    best_score = float(similarity_scores[0, best_match_index])
    if best_score < min_similarity:
        return None, best_score
    return candidate_documents[best_match_index], best_score


if __name__ == "__main__":
    # Accept one optional argument: audio filename/path to transcribe.
    input_audio_path = "examples/sample_audio.wav"
    if len(sys.argv) == 2:
        arg_path = Path(sys.argv[1])
        # If only a filename is provided, resolve it under examples/.
        if arg_path.parent == Path("."):
            input_audio_path = str(Path("examples") / arg_path.name)
        else:
            input_audio_path = str(arg_path)

    processed_audio_path = "examples/processed_audio.wav"

    # Task 1: audio preprocessing.
    process_audio(input_audio_path, processed_audio_path)

    # Task 3: speech-to-text.
    transcript_text = transcribe_audio(input_audio_path)

    # Task 2: train on real emotion labels from the JSON mapping.
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

    # Task 4: retrieval over candidate text snippets.
    candidate_documents = [
        transcript_text,
        f"Detected emotion: {predicted_input_emotion}",
        "The hero embarks on a journey",
        "A tragic ending unfolds",
    ]
    retrieval_query = "sad story"
    best_matching_document, similarity_score = retrieve_best_match(candidate_documents, retrieval_query)
    if best_matching_document is None:
        print(f"Task 4: No strong match for query (best cosine score={similarity_score:.3f}).")
    else:
        print(f"Task 4: Best match for query (score={similarity_score:.3f}): {best_matching_document}")
