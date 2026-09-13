import os
import torch
import torch.nn as nn
import numpy as np
from dataset import get_dataloader
from model import DoodleCNN

TEST_PATH = "data/processed/test/test.npz"
CHECKPOINT_PATH = "models/checkpoints/doodle_cnn.pth"
CLASSES_FILE = "data/classes.txt"
BATCH_SIZE = 64


def main():
    device = torch.device("cpu")
    print("Evaluating model performance...")

    # 1. Load class labels
    with open(CLASSES_FILE, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    # 2. Load test data loader
    test_loader = get_dataloader(TEST_PATH, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Instantiate model and load trained weights
    model = DoodleCNN(num_classes=len(classes)).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run python/train.py first!"
        )

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # 4. Metrics evaluation loop
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    overall_acc = (correct / total) * 100
    avg_loss = running_loss / total

    print(f"\nTest Loss: {avg_loss:.4f} | Overall Test Accuracy: {overall_acc:.2f}%\n")
    print("Per-Class Accuracy:")
    print("-" * 35)
    for i in range(len(classes)):
        acc = (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0
        print(f"{classes[i]:<15} : {acc:.2f}%")


if __name__ == "__main__":
    main()
