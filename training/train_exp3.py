"""Train the Experiment 3 chain predictor on the full Stage 2 rig.

Usage (from repo root):
    python -m training.train_exp3 --config configs/experiment3_300k.yaml [--resume]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC, render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.multihead import MultiHeadChainNet
from training.exp3_common import chain_from, compute_loss, decode
from training.shards import ShardSet

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_KEYS = ("continuous", "switches", "choices", "mask")


def recalibrate_batchnorm(model, features, device, batch_size=256):
    """Recompute BatchNorm's running statistics exactly over a fixed set of
    training examples. By default BatchNorm keeps a fast exponential average
    (momentum 0.1) over the last few dozen training batches of 32 while the
    weights are still moving, so the statistics it evaluates with are stale
    and noisy. On Experiment 3 that made validation loss jump around by
    tens of percent between epochs and early stopping keep an undertrained
    checkpoint. Only the statistics change here, never the weights."""
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all calibration batches
    model.train()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            model(features[start:start + batch_size].to(device))
    for m in bn_layers:
        m.momentum = 0.1
    model.eval()


def audio_eval(model, val, idxs, device, sr):
    model.eval()
    sims = []
    with torch.no_grad():
        pred = model(val["features"][idxs].to(device))
        for j, i in enumerate(idxs):
            gen = render(val["source"][i], sr, chain_from(*decode(pred, j)))
            sims.append(multi_resolution_stft_distance(val["audio"][i], gen)["similarity"])
    return float(np.mean(sims))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tcfg = config["training"]
    torch.manual_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = REPO_ROOT / config["dataset"]["data_root"]
    sr = config["dataset"]["sr"]

    train = ShardSet(data_root / "train", SPEC, n=config["dataset"]["n_train"],
                     shards_per_block=tcfg.get("shards_per_block", 2))
    val = ShardSet(data_root / "val", SPEC).load_all(with_audio=True)
    print(f"training on {len(train)} examples ({len(train.shards)} shards), validating on {len(val['features'])}")
    audio_idxs = np.random.default_rng(0).choice(len(val["features"]), size=tcfg["audio_eval_n_examples"], replace=False)

    # a fixed slice of training data for BatchNorm recalibration (see recalibrate_batchnorm)
    n_calib = tcfg.get("bn_recalibration_examples", 0)
    calib_features = None
    if n_calib:
        first_shard = train.shards[0][0]
        calib_features = torch.from_numpy(np.load(first_shard / "features.npy", mmap_mode="r")[:n_calib].copy())
        print(f"BatchNorm recalibrated each epoch on {len(calib_features)} training examples")

    model = MultiHeadChainNet(SPEC).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f)

    history, best_val_loss, epochs_since_best, start_epoch = [], float("inf"), 0, 1
    resume_path = checkpoint_dir / "resume.pt"
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        best_val_loss, epochs_since_best, start_epoch = state["best_val_loss"], state["epochs_since_best"], state["epoch"] + 1
        with open(run_dir / "history.json") as f:
            history = json.load(f)[:state["epoch"]]
        print(f"resuming after epoch {state['epoch']} (best val_loss so far {best_val_loss:.4f})")

    patience = tcfg.get("patience")
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        if patience and epochs_since_best >= patience:
            break
        model.train()
        t0, losses = time.time(), []
        for feature, target in train.batches(tcfg["batch_size"], epoch, tcfg["seed"]):
            target = {k: v.to(device) for k, v in target.items()}
            optimizer.zero_grad()
            loss = compute_loss(model(feature.to(device)), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if calib_features is not None:
            recalibrate_batchnorm(model, calib_features, device)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, len(val["features"]), 256):
                sl = slice(start, start + 256)
                target = {k: val[k][sl].to(device) for k in TARGET_KEYS}
                val_losses.append(compute_loss(model(val["features"][sl].to(device)), target).item() * len(val["features"][sl]))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "val_loss": float(np.sum(val_losses) / len(val["features"])), "seconds": time.time() - t0}
        line = f"epoch {epoch:3d} train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f}"
        if epoch % tcfg["audio_eval_every"] == 0:
            record["audio_similarity"] = audio_eval(model, val, audio_idxs, device, sr)
            line += f" audio_sim={record['audio_similarity']:.3f}"
        print(line, flush=True)
        history.append(record)

        if record["val_loss"] < best_val_loss:
            best_val_loss, epochs_since_best = record["val_loss"], 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_since_best += 1
        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch,
                    "best_val_loss": best_val_loss, "epochs_since_best": epochs_since_best}, resume_path)

    if patience and epochs_since_best >= patience:
        print(f"stopped early: no val_loss improvement for {patience} epochs")
    print(f"done. best val_loss={best_val_loss:.4f}. checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
