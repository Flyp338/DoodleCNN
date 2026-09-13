from typing import Any

import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader


class DoodleDataset(Dataset):
    def __init__(self, npz_path:str):
        data = np.load(npz_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32) # Shape: (N, 28, 28)
        self.y = torch.tensor(data['y'], dtype=torch.long)    # Shape: (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index) -> Any:
        image = self.X[index].unsqueeze(0)
        label = self.y[index]
        return image, label

def get_dataloader(npz_path: str, batch_size :int = 64, shuffle: bool = True):
    dataset = DoodleDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle = shuffle)