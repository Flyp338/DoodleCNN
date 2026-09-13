import os 
import json
import struct
import torch 
from model import DoodleCNN

CHECKPOINT_PATH = "models/checkpoints/doodle_cnn.pth"
EXPORT_DIR = "models/exported"
BIN_PATH = os.path.join(EXPORT_DIR, "model.bin")
JSON_PATH = os.path.join(EXPORT_DIR, "model.json")
CLASSES_SRC = "data/classes.txt"
CLASSES_DEST = os.path.join(EXPORT_DIR, "classes.txt")

def export():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 1. Load trained PyTorch checkpoint
    device = torch.device("cpu")
    model = DoodleCNN(num_classes=10)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    state_dict = model.state_dict()

    layer_keys = [
        "conv1.weight", "conv1.bias",
        "conv2.weight", "conv2.bias",
        "fc1.weight",   "fc1.bias",
        "fc2.weight",   "fc2.bias"
    ]
    print("Exporting model parameters to binary format...")
    
    total_floats = 0
    with open(BIN_PATH, "wb") as f:
        for key in layer_keys:
            tensor = state_dict[key].detach().cpu().numpy().astype("<f4") # force little-endian float32
            
            # Header for each tensor: [ndim (uint32), dim_0, dim_1, ...]
            shape = tensor.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))
            
            # Data block
            f.write(tensor.tobytes())
            
            num_elements = tensor.size
            total_floats += num_elements
            print(f"  -> Exported {key:<15} Shape: {str(shape):<18} Elements: {num_elements}")

    print(f"\nBinary export complete: {BIN_PATH} ({total_floats * 4 / 1024:.2f} KB)")

    # 3. Export layout metadata JSON
    metadata = {
        "architecture": "DoodleCNN",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "layers": [
            {"name": "conv1", "type": "Conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1},
            {"name": "pool1", "type": "MaxPool2d", "kernel_size": 2, "stride": 2},
            {"name": "conv2", "type": "Conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"name": "pool2", "type": "MaxPool2d", "kernel_size": 2, "stride": 2},
            {"name": "fc1",   "type": "Linear", "in_features": 1568, "out_features": 128},
            {"name": "fc2",   "type": "Linear", "in_features": 128, "out_features": 10}
        ]
    }
    
    with open(JSON_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Copy classes text file into export directory
    if os.path.exists(CLASSES_SRC):
        with open(CLASSES_SRC, "r") as src, open(CLASSES_DEST, "w") as dst:
            dst.write(src.read())

    print(f"Metadata exported: {JSON_PATH}")

if __name__ == "__main__":
    export()

