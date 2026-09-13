import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio.preprocessing.io import load_wav
from audio.rendering.chain_params import CONTINUOUS_NAMES, PARAM_MIN, PARAM_MAX


class ChainDataset(Dataset):
    def __init__(self, dataset_dir, split, feature_fn, sr):
        self.dataset_dir = Path(dataset_dir)
        self.feature_fn = feature_fn
        self.sr = sr

        with open(self.dataset_dir / "index.csv") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows found for split={split!r} in {dataset_dir}")

    def __len__(self):
        return len(self.rows)

    def example_dir(self, idx):
        return self.dataset_dir / self.rows[idx]["path"]

    def __getitem__(self, idx):
        row = self.rows[idx]
        example_dir = self.dataset_dir / row["path"]

        audio, _ = load_wav(example_dir / "audio.wav", sr=self.sr)
        feature = self.feature_fn(audio)

        continuous = torch.tensor(
            [(float(row[name]) - PARAM_MIN) / (PARAM_MAX - PARAM_MIN) for name in CONTINUOUS_NAMES],
            dtype=torch.float32,
        )
        target = {
            "continuous": continuous,
            "overdrive_on": torch.tensor(float(row["overdrive_on"]), dtype=torch.float32),
            "amp_id": torch.tensor(int(row["amp_id"]), dtype=torch.long),
            "cabinet_id": torch.tensor(int(row["cabinet_id"]), dtype=torch.long),
        }
        return feature, target, row["id"]
