import torch
import torch.nn as nn

from audio.rendering.chain_params import CONTINUOUS_NAMES, N_AMPS, N_CABINETS


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ChainPredictor(nn.Module):
    """log-mel spectrogram [B, 1, n_mels, time] -> a candidate signal chain:
    continuous knobs in [0, 1], an overdrive on/off logit, and amp/cabinet
    class logits.
    """

    def __init__(self, n_continuous=len(CONTINUOUS_NAMES), n_amps=N_AMPS, n_cabinets=N_CABINETS):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.continuous_head = nn.Linear(128, n_continuous)
        self.overdrive_head = nn.Linear(128, 1)
        self.amp_head = nn.Linear(128, n_amps)
        self.cabinet_head = nn.Linear(128, n_cabinets)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.trunk(x)
        return {
            "continuous": torch.sigmoid(self.continuous_head(x)),
            "overdrive_logit": self.overdrive_head(x).squeeze(-1),
            "amp_logits": self.amp_head(x),
            "cabinet_logits": self.cabinet_head(x),
        }
