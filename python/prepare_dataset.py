import os
import sys
import numpy as np

# Ensure Python can locate modules inside python/ directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_array

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CLASSES_FILE = "data/classes.txt"
SAMPLES_PER_CLASS = 10000

os.makedirs(os.path.join(PROCESSED_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "test"), exist_ok=True)

with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f if line.strip()]

all_images = []
all_labels = []

print("Loading and preprocessing raw files...")
for label_idx, cls in enumerate(classes):
    filepath = os.path.join(RAW_DIR, f"{cls}.npy")
    raw_data = np.load(filepath)[:SAMPLES_PER_CLASS]

    processed_data = preprocess_array(raw_data)
    labels = np.full(len(processed_data), label_idx, dtype=np.int64)

    all_images.append(processed_data)
    all_labels.append(labels)

X = np.concatenate(all_images, axis=0)
y = np.concatenate(all_labels, axis=0)

# 2. Shuffle dataset
print("Shuffling dataset...")
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# 3. Compute 80/10/10 split boundaries
total_samples = len(X)
train_end = int(total_samples * 0.8)
val_end = int(total_samples * 0.9)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 4. Save to disk
print("Saving splits to data/processed/...")
np.savez_compressed(
    os.path.join(PROCESSED_DIR, "train/train.npz"), X=X_train, y=y_train
)
np.savez_compressed(os.path.join(PROCESSED_DIR, "val/val.npz"), X=X_val, y=y_val)
np.savez_compressed(os.path.join(PROCESSED_DIR, "test/test.npz"), X=X_test, y=y_test)

print("Done!")
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
