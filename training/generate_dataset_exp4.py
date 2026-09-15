"""Generate the Experiment 4 (source/performance smoke test) dataset.

The signal chain is held fixed (Experiment 1's virtual amp at one preset),
so the only thing that varies between examples is the source: pickup
position, playing dynamics and tuning (audio.synth.source_spec). This
isolates whether a CNN can read source/performance characteristics from
audio at all, before combining them with signal-chain variation.

Usage (from repo root):
    python -m training.generate_dataset_exp4 --n-examples 3000 --out-name exp4_smoke
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from audio.synth import source_spec
from audio.synth.guitar_synth import synth_source_pickup
from audio.rendering.virtual_amp import render
from audio.preprocessing.io import save_wav

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXED_AMP_PARAMS = {"gain": 5.0, "bass": 5.0, "mid": 5.0, "treble": 5.0, "presence": 5.0, "master": 7.0}


def generate_split(split_name, n_examples, sr, duration, out_dir, rng, index_rows):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_examples):
        example_id = f"{i:06d}"
        example_dir = split_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)

        params = source_spec.random_params(rng)
        source_audio, note_events = synth_source_pickup(
            sr, duration, rng, params["pickup_position"], params["dynamics"], params["tuning_semitones"],
        )
        rendered_audio = render(source_audio, sr, FIXED_AMP_PARAMS)

        save_wav(example_dir / "input.wav", source_audio, sr)
        save_wav(example_dir / "audio.wav", rendered_audio, sr)
        with open(example_dir / "parameters.json", "w") as f:
            json.dump({"params": params, "source": note_events, "amp": FIXED_AMP_PARAMS}, f, indent=2)

        index_rows.append({
            "id": example_id, "split": split_name, "path": str(example_dir.relative_to(out_dir)), **params,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=3000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", type=str, required=True)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "data" / "generated" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    n_train = int(args.n_examples * args.train_frac)
    n_val = int(args.n_examples * args.val_frac)
    n_test = args.n_examples - n_train - n_val

    index_rows = []
    t0 = time.time()
    generate_split("train", n_train, args.sr, args.duration, out_dir, rng, index_rows)
    generate_split("val", n_val, args.sr, args.duration, out_dir, rng, index_rows)
    generate_split("test", n_test, args.sr, args.duration, out_dir, rng, index_rows)
    elapsed = time.time() - t0

    with open(out_dir / "index.csv", "w", newline="") as f:
        fieldnames = ["id", "split", "path"] + source_spec.PARAM_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "n_examples": args.n_examples, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "sr": args.sr, "duration": args.duration, "seed": args.seed,
        "param_names": source_spec.PARAM_NAMES, "param_range": [source_spec.PARAM_MIN, source_spec.PARAM_MAX],
        "fixed_amp_params": FIXED_AMP_PARAMS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_seconds": elapsed,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"generated {args.n_examples} examples in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
