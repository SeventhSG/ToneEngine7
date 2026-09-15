"""Spec-driven chain predictor (Experiment 3 onward).

Same convolutional trunk as chain_model.ChainPredictor, so results stay
comparable, but the output heads are generated from a ChainSpec instead of
being hard-coded: one sigmoid output per knob, one logit per on/off switch,
and one softmax head per categorical choice.
"""

import torch
import torch.nn as nn

from models.tone_predictor.chain_model import ConvBlock


class MultiHeadChainNet(nn.Module):
    def __init__(self, spec, width=128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, width),
            ConvBlock(width, width),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.continuous_head = nn.Linear(width, len(spec.continuous))
        self.switch_head = nn.Linear(width, len(spec.switch_names))
        self.choice_heads = nn.ModuleList([nn.Linear(width, len(spec.choices[c])) for c in spec.choice_names])

    def forward(self, x):
        x = self.trunk(self.pool(self.features(x)).flatten(1))
        return {
            "continuous": torch.sigmoid(self.continuous_head(x)),
            "switch_logits": self.switch_head(x),
            "choice_logits": [head(x) for head in self.choice_heads],
        }
