"""Train the Experiment 1 tone predictor.

Reports two kinds of validation error every `audio_eval_every` epochs,
because parameter accuracy and audio accuracy are not the same thing
(see INSTRUCTIONS.md):

- parameter MAE: how close the predicted knobs are to the true knobs
- audio similarity: re-render the predicted knobs through the same
  virtual amp and dry source, then compare against the reference audio

Usage (from repo root):
    python -m training.train --config configs/experiment1.yaml
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
from audio.preprocessing.io import load_wav
from audio.rendering.params import PARAM_NAMES, denormalize
from audio.rendering.virtual_amp import render as amp_render
from evaluation.metrics import parameter_mae, audio_similarity
from models.tone_predictor.model import TonePredictor
from training.dataset import ToneDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_audio_eval(model, dataset, device, n_examples, sr, feature_fn):
    model.eval()
    n = min(n_examples, len(dataset))
    idxs = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)

    param_errors = []
    similarities = []
    with torch.no_grad():
        for idx in idxs:
            feature, target, _ = dataset[idx]
            pred = model(feature.unsqueeze(0).to(device)).cpu().numpy()[0]
            true = target.numpy()

            param_errors.append(parameter_mae(pred, true)["mean"])

            example_dir = dataset.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            pred_params = dict(zip(PARAM_NAMES, denormalize(pred.tolist())))
            gen_audio = amp_render(source_audio, sr, pred_params)

            sim = audio_similarity(ref_audio, gen_audio)["similarity"]
            similarities.append(sim)

    return {
        "param_mae_mean": float(np.mean(param_errors)),
        "audio_similarity_mean": float(np.mean(similarities)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]

    feature_fn = LogMelFeature(
        sr=sr,
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        n_mels=config["features"]["n_mels"],
    )

    train_ds = ToneDataset(dataset_dir, "train", feature_fn, sr)
    val_ds = ToneDataset(dataset_dir, "val", feature_fn, sr)

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = TonePredictor().to(device)
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
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
        }

        if epoch % config["training"]["audio_eval_every"] == 0 or epoch == config["training"]["epochs"]:
            audio_eval = run_audio_eval(
                model, val_ds, device,
                config["training"]["audio_eval_n_examples"], sr, feature_fn,
            )
            epoch_record.update(audio_eval)
            print(f"epoch {epoch:3d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                  f"param_mae={audio_eval['param_mae_mean']:.3f} "
                  f"audio_sim={audio_eval['audio_similarity_mean']:.3f}")
        else:
            print(f"epoch {epoch:3d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

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
