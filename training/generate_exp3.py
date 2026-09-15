"""Generate Experiment 3 data (the full Stage 2 rig, audio/rendering/signal_chain3.py).

All three splits use the same sharded layout (see generate_shards_exp2.py for
why shards instead of one folder per example):

    shard_0000/features.npy  float32 [n, 1, n_mels, frames]
    shard_0000/labels.npz    continuous (0-10), switches, choices, noise_db
    shard_0000/source.npy    float32 [n, samples], the DI input incl. its noise floor  (--keep-audio)
    shard_0000/audio.npy     float32 [n, samples], the rendered reference            (--keep-audio)
    shard_0000/done.json     written last; shards without it are regenerated

Validation and test keep the audio because the optimizer re-renders from the
source and compares against the reference. Each split has its own default
seed, and every shard its own random stream (seed, shard index), so splits
never share draws and a killed run resumes where it stopped.

Usage (from repo root):
    python -m training.generate_exp3 --split test --n-examples 1000 --keep-audio
    python -m training.generate_exp3 --split val --n-examples 1000 --keep-audio
    python -m training.generate_exp3 --split train --n-examples 300000 --workers 3
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from audio.rendering.signal_chain3 import EXP3_SPEC, render
from audio.synth.guitar_synth import add_noise_floor, synth_phrase

REPO_ROOT = Path(__file__).resolve().parents[1]
SR = 16000
DURATION = 0.75
FEATURES = {"n_fft": 1024, "hop_length": 256, "n_mels": 64}
DEFAULT_SEEDS = {"train": 3000, "val": 3001, "test": 3002}


def generate_shard(job):
    shard_idx, n, seed, out_dir, keep_audio = job
    shard_dir = Path(out_dir) / f"shard_{shard_idx:04d}"
    if (shard_dir / "done.json").exists():
        return shard_idx, 0.0, True

    import torch
    from audio.features.spectrogram import LogMelFeature
    torch.set_num_threads(1)

    shard_dir.mkdir(parents=True, exist_ok=True)
    feature_fn = LogMelFeature(sr=SR, **FEATURES)
    rng = np.random.default_rng([seed, shard_idx])
    n_samples = int(DURATION * SR)
    n_frames = feature_fn(np.zeros(n_samples, dtype=np.float32)).shape[-1]
    spec = EXP3_SPEC

    feat_out = np.lib.format.open_memmap(shard_dir / "features.npy", mode="w+", dtype=np.float32,
                                         shape=(n, 1, FEATURES["n_mels"], n_frames))
    audio_out = source_out = None
    if keep_audio:
        audio_out = np.lib.format.open_memmap(shard_dir / "audio.npy", mode="w+", dtype=np.float32, shape=(n, n_samples))
        source_out = np.lib.format.open_memmap(shard_dir / "source.npy", mode="w+", dtype=np.float32, shape=(n, n_samples))
    continuous = np.zeros((n, len(spec.continuous)), dtype=np.float32)
    switches = np.zeros((n, len(spec.switch_names)), dtype=np.int8)
    choices = np.zeros((n, len(spec.choice_names)), dtype=np.int8)
    noise_db = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            chain = spec.random_chain(rng)
            source, _ = synth_phrase(SR, DURATION, rng)
            source, noise_db[i] = add_noise_floor(source, SR, rng)
            rendered = render(source, SR, chain)
            feat_out[i] = feature_fn(rendered).numpy()
            if keep_audio:
                audio_out[i] = rendered
                source_out[i] = source
            continuous[i], switches[i], choices[i] = spec.to_arrays(chain)

    feat_out.flush()
    if keep_audio:
        audio_out.flush()
        source_out.flush()
    del feat_out, audio_out, source_out
    np.savez(shard_dir / "labels.npz", continuous=continuous, switches=switches, choices=choices, noise_db=noise_db)
    seconds = time.time() - t0
    with open(shard_dir / "done.json", "w") as f:
        json.dump({"shard": shard_idx, "n": n, "seed": seed, "seconds": seconds}, f)
    return shard_idx, seconds, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(DEFAULT_SEEDS), required=True)
    parser.add_argument("--n-examples", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None, help="defaults to a fixed per-split seed")
    parser.add_argument("--out-root", type=str, default="data/generated/exp3")
    parser.add_argument("--keep-audio", action="store_true", help="store source and rendered audio (val/test)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    seed = DEFAULT_SEEDS[args.split] if args.seed is None else args.seed
    out_dir = REPO_ROOT / args.out_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = -(-args.n_examples // args.shard_size)
    sizes = [min(args.shard_size, args.n_examples - i * args.shard_size) for i in range(n_shards)]

    manifest = {
        "split": args.split, "n_examples": args.n_examples, "shard_size": args.shard_size, "n_shards": n_shards,
        "seed": seed, "sr": SR, "duration": DURATION, "features": FEATURES, "audio": args.keep_audio,
        "source": "synth_phrase + add_noise_floor",
        "continuous": EXP3_SPEC.continuous, "switches": EXP3_SPEC.switches, "choices": EXP3_SPEC.choices,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    jobs = [(i, sizes[i], seed, str(out_dir), args.keep_audio) for i in range(n_shards)]
    t0 = time.time()
    with Pool(min(args.workers, n_shards)) as pool:
        for shard_idx, seconds, skipped in pool.imap_unordered(generate_shard, jobs):
            status = "already done" if skipped else f"{seconds:.0f}s"
            print(f"{args.split} shard {shard_idx:04d}: {status} ({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"{args.split} complete: {args.n_examples} examples in {n_shards} shards -> {out_dir}")


if __name__ == "__main__":
    main()
