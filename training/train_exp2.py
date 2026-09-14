"""Train the Experiment 2 chain predictor: overdrive on/off, amp choice,
cabinet choice, and the continuous knobs, all from one spectrogram.

Usage (from repo root):
    python -m training.train_exp2 --config configs/experiment2.yaml

If the config sets dataset.train_pool_dir, training batches come from a shard
pool (training/generate_shards_exp2.py) instead of the dataset's own train
split; validation still uses dataset_dir's val split, so runs stay comparable.
--resume continues a killed run from its last finished epoch.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.chain_params import CONTINUOUS_NAMES, denormalize_continuous
from audio.rendering.signal_chain import render as chain_render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.chain_model import ChainPredictor
from training.dataset_exp2 import ChainDataset
from training.pool_exp2 import ShardPool

REPO_ROOT = Path(__file__).resolve().parents[1]

# overdrive_drive and overdrive_level are meaningless when the pedal is bypassed
_OVERDRIVE_MASK_IDX = [CONTINUOUS_NAMES.index("overdrive_drive"), CONTINUOUS_NAMES.index("overdrive_level")]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_loss(pred, target):
    mask = torch.ones_like(target["continuous"])
    mask[:, _OVERDRIVE_MASK_IDX] = target["overdrive_on"].unsqueeze(1)
    continuous_loss = (((pred["continuous"] - target["continuous"]) ** 2) * mask).sum() / mask.sum().clamp(min=1)

    overdrive_loss = F.binary_cross_entropy_with_logits(pred["overdrive_logit"], target["overdrive_on"])
    amp_loss = F.cross_entropy(pred["amp_logits"], target["amp_id"])
    cabinet_loss = F.cross_entropy(pred["cabinet_logits"], target["cabinet_id"])

    total = continuous_loss + overdrive_loss + amp_loss + cabinet_loss
    return total, {
        "continuous": continuous_loss.item(),
        "overdrive": overdrive_loss.item(),
        "amp": amp_loss.item(),
        "cabinet": cabinet_loss.item(),
    }


def pred_to_chain(pred, idx):
    continuous = denormalize_continuous(pred["continuous"][idx].detach().cpu().numpy().tolist())
    chain = dict(zip(CONTINUOUS_NAMES, continuous))
    chain["overdrive_on"] = bool(torch.sigmoid(pred["overdrive_logit"][idx]).item() > 0.5)
    chain["amp_id"] = int(pred["amp_logits"][idx].argmax().item())
    chain["cabinet_id"] = int(pred["cabinet_logits"][idx].argmax().item())
    return chain


def run_audio_eval(model, dataset, device, n_examples, sr):
    model.eval()
    n = min(n_examples, len(dataset))
    idxs = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)

    similarities = []
    amp_correct = 0
    cabinet_correct = 0
    overdrive_correct = 0
    with torch.no_grad():
        for idx in idxs:
            feature, target, _ = dataset[idx]
            pred = model(feature.unsqueeze(0).to(device))

            amp_correct += int(pred["amp_logits"][0].argmax().item() == target["amp_id"].item())
            cabinet_correct += int(pred["cabinet_logits"][0].argmax().item() == target["cabinet_id"].item())
            overdrive_pred = torch.sigmoid(pred["overdrive_logit"][0]).item() > 0.5
            overdrive_correct += int(overdrive_pred == bool(target["overdrive_on"].item() > 0.5))

            example_dir = dataset.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            chain = pred_to_chain(pred, 0)
            gen_audio = chain_render(source_audio, sr, chain)
            sim = multi_resolution_stft_distance(ref_audio, gen_audio)["similarity"]
            similarities.append(sim)

    return {
        "audio_similarity_mean": float(np.mean(similarities)),
        "amp_accuracy": amp_correct / n,
        "cabinet_accuracy": cabinet_correct / n,
        "overdrive_accuracy": overdrive_correct / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from the last finished epoch")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]

    feature_fn = LogMelFeature(
        sr=sr, n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"], n_mels=config["features"]["n_mels"],
    )

    batch_size = config["training"]["batch_size"]
    pool_dir = config["dataset"].get("train_pool_dir")
    if pool_dir:
        pool = ShardPool(REPO_ROOT / pool_dir, config["dataset"]["n_train"], config["training"].get("shards_per_block", 2))
        print(f"training on {len(pool)} examples from {pool_dir} ({len(pool.shards)} shards)")

        def train_batches(epoch):
            return pool.batches(batch_size, epoch, config["training"]["seed"])
    else:
        train_ds = ChainDataset(dataset_dir, "train", feature_fn, sr)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=config["training"]["num_workers"],
        )

        def train_batches(epoch):
            return ((feature, target) for feature, target, _ in train_loader)

    val_ds = ChainDataset(dataset_dir, "val", feature_fn, sr)
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = ChainPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # optional: decay the learning rate to zero over the configured epochs
    scheduler = None
    if config["training"].get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    history = []
    best_val_loss = float("inf")
    epochs_since_best = 0
    start_epoch = 1
    patience = config["training"].get("patience")
    resume_path = checkpoint_dir / "resume.pt"
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())  # map_location moved it to the GPU; the CPU RNG needs it back
        best_val_loss, epochs_since_best = state["best_val_loss"], state["epochs_since_best"]
        start_epoch = state["epoch"] + 1
        with open(run_dir / "history.json") as f:
            history = json.load(f)[:state["epoch"]]
        print(f"resuming after epoch {state['epoch']} (best val_loss so far {best_val_loss:.4f})")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        if patience and epochs_since_best >= patience:
            break
        model.train()
        train_losses = []
        t0 = time.time()
        for feature, target in train_batches(epoch):
            feature = feature.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            optimizer.zero_grad()
            pred = model(feature)
            loss, _ = compute_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for feature, target, _ in val_loader:
                feature = feature.to(device)
                target = {k: v.to(device) for k, v in target.items()}
                pred = model(feature)
                loss, _ = compute_loss(pred, target)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.time() - t0,
                        "lr": optimizer.param_groups[0]["lr"]}
        if scheduler is not None:
            scheduler.step()

        if epoch % config["training"]["audio_eval_every"] == 0 or epoch == config["training"]["epochs"]:
            audio_eval = run_audio_eval(model, val_ds, device, config["training"]["audio_eval_n_examples"], sr)
            epoch_record.update(audio_eval)
            print(f"epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"audio_sim={audio_eval['audio_similarity_mean']:.3f} "
                  f"amp_acc={audio_eval['amp_accuracy']:.2f} cab_acc={audio_eval['cabinet_accuracy']:.2f} "
                  f"od_acc={audio_eval['overdrive_accuracy']:.2f}")
        else:
            print(f"epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "epoch": epoch, "best_val_loss": best_val_loss, "epochs_since_best": epochs_since_best,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }, resume_path)

    if patience and epochs_since_best >= patience:
        print(f"stopped early: no val_loss improvement for {patience} epochs")

    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"done. best val_loss={best_val_loss:.4f}. checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
