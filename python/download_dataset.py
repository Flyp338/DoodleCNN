import os
import urllib.request
import numpy as np
from preprocess import preprocess_array

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CLASSES_FILE = "data/classes.txt"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
SAMPLES_PER_CLASS = 10000  # Cap samples per class

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f]

all_images = []
all_labels = []

for label_idx, cls in enumerate(classes):
    filename = f"{cls}.npy"
    filepath = os.path.join(RAW_DIR, filename)
    # 1. Download if not locally available
    if not os.path.exists(filepath):
        cls_url = cls.replace(" ", "%20")
        url = f"{BASE_URL}/{cls_url}.npy"
        print(f"Downloading {cls}...")
        urllib.request.urlretrieve(url, filepath)

    # 2. Load raw NumPy array
    raw_data = np.load(filepath)[:SAMPLES_PER_CLASS]

    # 3. Process using your updated preprocessor
    processed_data = preprocess_array(raw_data)

    # 4. Create matching numerical labels (0 to 9)
    labels = np.full(len(processed_data), label_idx, dtype=np.int64)

    all_images.append(processed_data)
    all_labels.append(labels)

X = np.concatenate(all_images, axis = 0)
Y = np.concatenate(all_labels, axis = 0)

print(f"Dataset ready! {X.shape = }, {Y.shape = }")


