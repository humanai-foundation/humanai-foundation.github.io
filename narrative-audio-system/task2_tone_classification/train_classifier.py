import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score


class ToneClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_features):
        hidden_features = self.activation(self.hidden_layer(input_features))
        return self.output_layer(hidden_features)


def train_model(train_loader, model, criterion, optimizer):
    # Epoch-level metrics are aggregated across mini-batches.
    for epoch in range(5):
        epoch_loss = 0.0
        predicted_labels = []
        true_labels = []

        for batch_features, batch_labels in train_loader:
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_predictions = torch.argmax(logits, dim=1)
            predicted_labels.extend(batch_predictions.tolist())
            true_labels.extend(batch_labels.tolist())

        accuracy = accuracy_score(true_labels, predicted_labels)
        # Weighted F1 is more stable than plain accuracy with class imbalance.
        weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
        print(
            f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, "
            f"Acc={accuracy:.3f}, F1={weighted_f1:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    # Synthetic data keeps this script runnable even without a dataset file.
    num_samples = 64
    input_dim = 20
    num_classes = 8
    hidden_dim = 32
    batch_size = 16

    feature_matrix = torch.randn(num_samples, input_dim)
    class_labels = torch.randint(0, num_classes, (num_samples,))
    train_dataset = TensorDataset(feature_matrix, class_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ToneClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(train_loader, model, criterion, optimizer)
