from PIL import Image
import numpy as np 


IMAGE_SIZE = 28


def preprocess_image(path: str) -> np.ndarray:
    """Preprocess custom user images (PNG/JPG) for inference."""
    image = Image.open(path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    pixel = np.asarray(image, dtype = np.float32)
    pixel /= 255.0
    return pixel 

def preprocess_array(raw_array: np.ndarray) -> np.ndarray:
    """Preprocess Quick, Draw! .npy flat arrays for training."""
    normalized = raw_array.astype(np.float32) / 255.0
    return normalized.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)