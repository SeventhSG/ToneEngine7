"""Train the Experiment 4 source/performance predictor.

Signal chain is fixed (see training/generate_dataset_exp4.py), so this only
tests whether a CNN can read pickup position, playing dynamics and tuning
from audio. Parameter MAE only: audio-similarity re-rendering would need to
reproduce the exact note events (tuning shifts which MIDI note gets played),
which is out of scope for this smoke test.

Usage (from repo root):
    python -m training.train_exp4 --config configs/experiment4.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from audio.features.spectrogram import LogMelFeature
from audio.synth.source_spec import PARAM_NAMES, PARAM_MAX, PARAM_MIN
from models.tone_predictor.model import TonePredictor
from training.dataset_exp4 import SourceDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def parameter_mae(pred_vector, true_vector):
    scale = PARAM_MAX - PARAM_MIN
    abs_err = np.abs(np.asarray(pred_vector) - np.asarray(true_vector)) * scale
    per_param = {name: float(err) for name, err in zip(PARAM_NAMES, abs_err)}
    per_param["mean"] = float(abs_err.mean())
    return per_param


def run_param_eval(model, dataset, device, n_examples):
    model.eval()
    n = min(n_examples, len(dataset))
    idxs = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)

    per_param_errors = {name: [] for name in PARAM_NAMES}
    with torch.no_grad():
        for idx in idxs:
            feature, target, _ = dataset[idx]
            pred = model(feature.unsqueeze(0).to(device)).cpu().numpy()[0]
            true = target.numpy()
            mae = parameter_mae(pred, true)
            for name in PARAM_NAMES:
                per_param_errors[name].append(mae[name])

    result = {name: float(np.mean(errs)) for name, errs in per_param_errors.items()}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]

    feature_fn = LogMelFeature(
        sr=sr, n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"], n_mels=config["features"]["n_mels"],
    )

    train_ds = SourceDataset(dataset_dir, "train", feature_fn, sr)
    val_ds = SourceDataset(dataset_dir, "val", feature_fn, sr)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
                              num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
                            num_workers=config["training"]["num_workers"])

    model = TonePredictor(n_params=len(PARAM_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = torch.nn.MSELoss()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        t0 = time.time()
        for feature, target, _ in train_loader:
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for feature, target, _ in val_loader:
                feature, target = feature.to(device), target.to(device)
                pred = model(feature)
                val_losses.append(criterion(pred, target).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.time() - t0}

        param_eval = run_param_eval(model, val_ds, device, 200)
        epoch_record["param_mae"] = param_eval
        print(f"epoch {epoch:3d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"param_mae={param_eval['mean']:.3f}")

        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"done. best val_loss={best_val_loss:.5f}. checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
