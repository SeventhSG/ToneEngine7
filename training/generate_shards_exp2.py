"""Generate a large Experiment 2 training pool as compact shards.

generate_dataset_exp2.py writes three files per example, which is fine for
10k examples on an SSD but not for hundreds of thousands on a spinning disk,
where reading them back one by one every epoch would dominate training time.
Training only ever needs the log-mel features and the labels, so each shard
here is a handful of large arrays written sequentially:

    shard_0000/features.npy   float32 [n, 1, n_mels, frames], same LogMelFeature as training
    shard_0000/audio.npy      float32 [n, samples], the rendered audio, kept so the
                              features can be recomputed with other settings
    shard_0000/labels.npz     continuous knobs (0-10), overdrive_on, amp_id, cabinet_id
    shard_0000/done.json      written last; a shard without it is regenerated

Every shard draws from its own random stream (seed, shard index), so shards
are reproducible independently and the run can be resumed or parallelized.
This pool is train-only: validation and test stay the exp2_smoke splits, so
results remain comparable with everything already reported.

Usage (from repo root):
    python -m training.generate_shards_exp2 --n-examples 300000 --out-dir data/generated/exp2_pool --no-audio --workers 2
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from audio.rendering.chain_params import CONTINUOUS_NAMES, random_chain
from audio.rendering.signal_chain import render
from audio.synth.guitar_synth import synth_source

SR = 16000
DURATION = 0.75
FEATURES = {"n_fft": 1024, "hop_length": 256, "n_mels": 64}


def generate_shard(job):
    shard_idx, n, seed, out_dir, keep_audio = job
    shard_dir = Path(out_dir) / f"shard_{shard_idx:04d}"
    if (shard_dir / "done.json").exists():
        return shard_idx, 0.0, True

    # imported here so the parent process stays light
    import torch
    from audio.features.spectrogram import LogMelFeature
    torch.set_num_threads(1)

    shard_dir.mkdir(parents=True, exist_ok=True)
    feature_fn = LogMelFeature(sr=SR, **FEATURES)
    rng = np.random.default_rng([seed, shard_idx])
    n_samples = int(DURATION * SR)
    n_frames = feature_fn(np.zeros(n_samples, dtype=np.float32)).shape[-1]

    audio_out = None
    if keep_audio:
        audio_out = np.lib.format.open_memmap(shard_dir / "audio.npy", mode="w+", dtype=np.float32,
                                              shape=(n, n_samples))
    feat_out = np.lib.format.open_memmap(shard_dir / "features.npy", mode="w+", dtype=np.float32,
                                         shape=(n, 1, FEATURES["n_mels"], n_frames))
    continuous = np.zeros((n, len(CONTINUOUS_NAMES)), dtype=np.float32)
    overdrive_on = np.zeros(n, dtype=np.int8)
    amp_id = np.zeros(n, dtype=np.int8)
    cabinet_id = np.zeros(n, dtype=np.int8)

    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            chain = random_chain(rng)
            source_audio, _ = synth_source(SR, DURATION, rng)
            rendered = render(source_audio, SR, chain)
            if audio_out is not None:
                audio_out[i] = rendered
            feat_out[i] = feature_fn(rendered).numpy()
            continuous[i] = [chain[name] for name in CONTINUOUS_NAMES]
            overdrive_on[i] = int(chain["overdrive_on"])
            amp_id[i] = chain["amp_id"]
            cabinet_id[i] = chain["cabinet_id"]

    if audio_out is not None:
        audio_out.flush()
    feat_out.flush()
    del audio_out, feat_out
    np.savez(shard_dir / "labels.npz", continuous=continuous, overdrive_on=overdrive_on,
             amp_id=amp_id, cabinet_id=cabinet_id)
    seconds = time.time() - t0
    with open(shard_dir / "done.json", "w") as f:
        json.dump({"shard": shard_idx, "n": n, "seed": seed, "seconds": seconds}, f)
    return shard_idx, seconds, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1000,
                        help="must differ from the exp2_smoke seed (42) so the pool never repeats its val/test draws")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-audio", action="store_true",
                        help="skip audio.npy (4x the size of the features); training only reads features and labels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = -(-args.n_examples // args.shard_size)
    sizes = [min(args.shard_size, args.n_examples - i * args.shard_size) for i in range(n_shards)]

    manifest = {
        "n_examples": args.n_examples, "shard_size": args.shard_size, "n_shards": n_shards,
        "seed": args.seed, "sr": SR, "duration": DURATION, "features": FEATURES,
        "continuous_names": CONTINUOUS_NAMES, "split": "train", "audio": not args.no_audio,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    jobs = [(i, sizes[i], args.seed, str(out_dir), not args.no_audio) for i in range(n_shards)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for shard_idx, seconds, skipped in pool.imap_unordered(generate_shard, jobs):
            status = "already done" if skipped else f"{seconds:.0f}s"
            print(f"shard {shard_idx:04d}: {status} ({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"pool complete: {args.n_examples} examples in {n_shards} shards -> {out_dir}")


if __name__ == "__main__":
    main()
