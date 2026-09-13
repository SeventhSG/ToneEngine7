"""Generate the Experiment 1 synthetic dataset.

For each example: sample random amp knobs, synthesize a short DI guitar
source, render it through the virtual amp. Both the dry source and the
rendered audio are saved, because evaluation later needs to re-render a
model's predicted parameters through the *same* source to get a fair
audio-similarity comparison against the original.

Usage (from repo root):
    python -m training.generate_dataset --n-examples 2000 --out-name exp1_smoke
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from audio.rendering import params as param_spec
from audio.rendering.virtual_amp import render
from audio.synth.guitar_synth import synth_source
from audio.preprocessing.io import save_wav

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def generate_split(split_name, n_examples, sr, duration, out_dir, rng, index_rows):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_examples):
        example_id = f"{i:06d}"
        example_dir = split_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)

        knob_params = param_spec.random_params(rng)
        source_audio, note_events = synth_source(sr, duration, rng)
        rendered_audio = render(source_audio, sr, knob_params)

        save_wav(example_dir / "input.wav", source_audio, sr)
        save_wav(example_dir / "audio.wav", rendered_audio, sr)
        with open(example_dir / "parameters.json", "w") as f:
            json.dump({"params": knob_params, "source": note_events}, f, indent=2)

        index_rows.append({
            "id": example_id,
            "split": split_name,
            "path": str(example_dir.relative_to(out_dir)),
            **knob_params,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=2000)
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
        fieldnames = ["id", "split", "path"] + param_spec.PARAM_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "n_examples": args.n_examples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "sr": args.sr,
        "duration": args.duration,
        "seed": args.seed,
        "param_names": param_spec.PARAM_NAMES,
        "param_range": [param_spec.PARAM_MIN, param_spec.PARAM_MAX],
        "git_commit": git_commit_hash(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_seconds": elapsed,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"generated {args.n_examples} examples in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
