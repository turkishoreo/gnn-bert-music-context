"""
Copies a small random sample of already-cached graph .pt files into
data/processed/examples/ -- a folder that IS tracked in git (unlike the bulk
gtzan_graphs/mtt_graphs caches, which .gitignore excludes since they can hold
thousands of files). Satisfies spec Section 10's submission requirement:
"Preprocessed graph samples (at least 20 example .pt/.json graphs)".

Run this AFTER prepare_gtzan.py and/or prepare_magnatagatune.py, right before
committing to git.

Usage:
    python make_example_graphs.py --config ../config.yaml --n 25
"""
import argparse
import glob
import os
import random
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--n", type=int, default=25, help="how many example graphs to keep (spec asks for >=20)")
    args = p.parse_args()

    processed_dir = os.path.normpath(os.path.join(os.path.dirname(args.config), "data", "processed"))
    out_dir = os.path.join(processed_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)

    candidates = []
    for sub in ("gtzan_graphs", "mtt_graphs"):
        d = os.path.join(processed_dir, sub)
        if os.path.isdir(d):
            candidates += glob.glob(os.path.join(d, "*.pt"))

    if not candidates:
        raise SystemExit(f"No cached .pt graphs found under {processed_dir}/gtzan_graphs or "
                          f"/mtt_graphs. Run prepare_gtzan.py and/or prepare_magnatagatune.py first.")

    random.seed(42)
    sample = random.sample(candidates, k=min(args.n, len(candidates)))

    for path in sample:
        source_dataset = "gtzan" if "gtzan_graphs" in path else "mtt"
        fname = f"{source_dataset}_{os.path.basename(path)}"
        shutil.copy(path, os.path.join(out_dir, fname))

    print(f"copied {len(sample)} example graphs to {out_dir}")
    print("this folder IS tracked in git (unlike the bulk gtzan_graphs/mtt_graphs caches)")


if __name__ == "__main__":
    main()
