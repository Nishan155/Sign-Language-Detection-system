import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)

class TCNClassifier(nn.Module):
    def __init__(self, input_size=126, num_classes=150):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_size, 128),
            TCNBlock(128, 256),
            TCNBlock(256, 128)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)