import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class ToneClassifier(nn.Module):
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


def extract_mfcc_vector(audio_path, sample_rate=16000, n_mfcc=13):
    waveform, loaded_sr = librosa.load(str(audio_path), sr=sample_rate)
    mfcc_matrix = librosa.feature.mfcc(y=waveform, sr=loaded_sr, n_mfcc=n_mfcc)
    return mfcc_matrix.mean(axis=1).astype(np.float32)


def load_dataset(label_map_path, sample_rate=16000, n_mfcc=13):
    """Load MFCC features and integer-encoded labels from an emotion JSON map."""
    with Path(label_map_path).open("r", encoding="utf-8-sig") as fp:
        filename_to_emotion = json.load(fp)

    audio_root = Path(label_map_path).parent
    features, raw_labels = [], []
    for filename, emotion in filename_to_emotion.items():
        audio_path = audio_root / filename
        if not audio_path.is_file():
            continue
        features.append(extract_mfcc_vector(audio_path, sample_rate, n_mfcc))
        raw_labels.append(emotion.strip().title())

    if not features:
        raise ValueError(f"No audio files found relative to {label_map_path}")

    class_names, encoded = np.unique(raw_labels, return_inverse=True)
    return np.array(features, dtype=np.float32), encoded, class_names.tolist()


def train_model(train_loader, model, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss, all_preds, all_true = 0.0, [], []
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).tolist())
            all_true.extend(batch_y.tolist())

        if epoch in {0, num_epochs - 1} or (epoch + 1) % 10 == 0:
            acc = accuracy_score(all_true, all_preds)
            f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
            print(f"  Epoch {epoch + 1:3d}: loss={epoch_loss:.4f}  train_acc={acc:.3f}  train_f1={f1:.3f}")


def evaluate_model(model, X_test, y_test, class_names):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    report = classification_report(y_test, preds, target_names=class_names, zero_division=0)
    return acc, f1, report, preds


def discuss_results(acc, f1, class_names, y_test, preds):
    print("\n--- Discussion of Results ---")
    print(
        f"The model achieved {acc:.1%} test accuracy and a weighted F1 of {f1:.3f} "
        f"on {len(y_test)} held-out samples across {len(class_names)} emotion classes."
    )
    if acc >= 0.5:
        print("Performance is above random chance (12.5% for 8 classes), showing the MFCC")
        print("features carry signal. A deeper model or data augmentation could improve further.")
    else:
        print("Performance is modest, likely due to the small dataset size (~127 samples).")
        print("Pretrained audio embeddings (e.g. wav2vec2) would be a stronger baseline.")
    confused_pairs = [
        (class_names[true_idx], class_names[pred_idx])
        for true_idx, pred_idx in zip(y_test, preds)
        if true_idx != pred_idx
    ]
    if confused_pairs:
        from collections import Counter
        top = Counter(confused_pairs).most_common(3)
        print("Most common confusions:", ", ".join(f"{a}→{b}" for (a, b), _ in top))
    print("----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Narrative Tone Classification")
    parser.add_argument(
        "--label-map",
        default="../examples/emotion_labels.json",
        help="Path to emotion_labels.json",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label_map_path = (Path(__file__).resolve().parent / args.label_map).resolve()
    print(f"\nLoading dataset from {label_map_path} ...")
    features, labels, class_names = load_dataset(
        label_map_path, sample_rate=args.sample_rate, n_mfcc=args.n_mfcc
    )
    print(f"Loaded {len(labels)} samples, {len(class_names)} classes: {', '.join(class_names)}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    # Z-score normalisation (fit on train only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = ToneClassifier(
        input_dim=features.shape[1], hidden_dim=args.hidden_dim, num_classes=len(class_names)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n--- Training ({args.epochs} epochs, {len(X_train)} train / {len(X_test)} test) ---")
    train_model(train_loader, model, criterion, optimizer, num_epochs=args.epochs)

    print("\n--- Test Evaluation ---")
    acc, f1, report, preds = evaluate_model(model, X_test, y_test, class_names)
    print(f"Test Accuracy : {acc:.3f}")
    print(f"Weighted F1   : {f1:.3f}")
    print("\nPer-class Report:")
    print(report)

    discuss_results(acc, f1, class_names, y_test, preds)
