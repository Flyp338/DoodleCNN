import torch
import torch.nn as nn
import torch.nn.functional as F

class DoodleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(DoodleCNN, self).__init__()
        """
        conv1: Takes 1 input channel (grayscale image) and applies 16 learnable 3x3 filters. padding=1 preserves spatial dimensions at 28x28.
        """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        """
        conv2: Takes 16 channels and applies 32 learnable 3x3 filters. Output remains 28x28.
        """
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #input shape: (Batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        # Conv1: (Batch, 16, 28, 28)
        # Pool1: (Batch, 16, 14, 14)

        x = self.pool(F.relu(self.conv2(x)))
        # Conv2: (Batch, 32, 14, 14)
        # Pool2: (Batch, 32, 7, 7)

        x = x.view(x.size(0), -1)
        # Flatten: (Batch, 1568)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Final shape: (Batch, 10)

        return x



