"""
Real-data loader for MagnaTagATune (Section 3 Table 1: BERT-tags + GNN-structure dataset).

Parses annotations_final.csv (clip_id, 188 tag columns, mp3_path -- tab-separated,
per the official dataset format), selects the top-K most frequent tags (spec
Section 4.1: "MagnaTagATune tag subset (top-50 tags)"), assigns the OFFICIAL split
convention used across MTT literature (folder 0-b -> train, c -> val, d-f -> test;
this satisfies spec Section 3 step 5: "Use official ... MagnaTagATune splits"),
builds a segment graph per clip (same graph_builder.py used for GTZAN), and writes
everything needed for both Task 1 (BERT tags) and Task 3 (GNN-BERT fusion).

25,863 clips is too many to process on a laptop CPU in reasonable time, so this
caps how many clips to process per split via --max_per_split (documented,
common practice for coursework-scale reproductions -- state this in your report).

Usage:
    python prepare_magnatagatune.py --config ../config.yaml \\
        --annotations ../../MagnaTagATune/annotations_final.csv \\
        --audio_root ../../MagnaTagATune/MagnaTagATune \\
        --max_per_split 600 100 100
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import audio_features as af
import graph_builder as gb

# official MTT split convention (see e.g. sample-cnn, MIR_sample_cnn repos)
TRAIN_FOLDERS = set("0123456789ab")
VAL_FOLDERS = set("c")
TEST_FOLDERS = set("def")


def load_top_tags(annotations_path, num_tags=50):
    df = pd.read_csv(annotations_path, sep="\t")
    tag_cols = [c for c in df.columns if c not in ("clip_id", "mp3_path")]
    freq = df[tag_cols].sum(axis=0).sort_values(ascending=False)
    top_tags = list(freq.index[:num_tags])
    return df, top_tags


def assign_split(mp3_path):
    folder = mp3_path.split("/")[0].lower()
    if folder in TRAIN_FOLDERS:
        return "train"
    if folder in VAL_FOLDERS:
        return "val"
    if folder in TEST_FOLDERS:
        return "test"
    return None  # unexpected folder name, skip


def build_records(df, top_tags, max_per_split, seed=42):
    """Filters to rows with >=1 active top-tag, assigns official split, then
    randomly subsamples each split down to max_per_split[split] for tractable
    runtime. Returns dict split -> list of row dicts."""
    r = random.Random(seed)
    active_mask = df[top_tags].sum(axis=1) > 0
    df = df[active_mask].reset_index(drop=True)

    by_split = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        split = assign_split(row["mp3_path"])
        if split is None:
            continue
        tags_present = [t for t in top_tags if row[t] == 1]
        by_split[split].append({
            "clip_id": int(row["clip_id"]),
            "mp3_path": row["mp3_path"],
            "tags": tags_present,
            "tag_vector": [int(row[t]) for t in top_tags],
        })

    caps = dict(zip(["train", "val", "test"], max_per_split))
    for split in by_split:
        r.shuffle(by_split[split])
        by_split[split] = by_split[split][:caps[split]]
    return by_split


def build_and_cache_graphs(by_split, audio_root, sr, n_chroma, window_frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    skipped = []
    final = {"train": [], "val": [], "test": []}
    total = sum(len(v) for v in by_split.values())
    pbar = tqdm(total=total, desc="extracting features + building graphs")
    for split, records in by_split.items():
        for rec in records:
            clip_id = rec["clip_id"]
            audio_path = os.path.join(audio_root, *rec["mp3_path"].split("/"))
            graph_path = os.path.join(out_dir, f"{clip_id}.pt")
            try:
                if not os.path.exists(graph_path):
                    y = af.load_audio(audio_path, sr=sr)
                    chroma = af.chroma_features(y, sr=sr, n_chroma=n_chroma)
                    graph = gb.segment_graph(chroma, window=window_frames)
                    torch.save(graph, graph_path)
                rec["graph_path"] = graph_path
                final[split].append(rec)
            except Exception as e:
                skipped.append({"clip_id": clip_id, "path": audio_path, "error": str(e)})
            pbar.update(1)
    pbar.close()
    if skipped:
        print(f"skipped {len(skipped)} unreadable/corrupt file(s) "
              f"(known issue: MTT ships a handful of corrupted mp3s -- this is expected)")
        if len(skipped) <= 10:
            print([s["path"] for s in skipped])
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--annotations", required=True,
                   help="path to annotations_final.csv")
    p.add_argument("--audio_root", required=True,
                   help="path to the folder containing 0/ 1/ ... f/ subfolders of mp3s")
    p.add_argument("--num_tags", type=int, default=None, help="defaults to config.yaml's data.num_tags")
    p.add_argument("--max_per_split", type=int, nargs=3, default=[600, 100, 100],
                   help="cap on clips per split: train val test (default 600 100 100, "
                        "~800 clips total -- adjust up if you have time to spare)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    num_tags = args.num_tags or cfg["data"]["num_tags"]

    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))
    processed_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                   cfg["data"]["processed_dir"], "mtt_graphs"))
    os.makedirs(splits_dir, exist_ok=True)

    print(f"loading {args.annotations} ...")
    df, top_tags = load_top_tags(args.annotations, num_tags=num_tags)
    print(f"selected top-{num_tags} tags by frequency: {top_tags[:10]}... (+{num_tags-10} more)")

    by_split = build_records(df, top_tags, args.max_per_split)
    print(f"after filtering to clips with >=1 top-tag and applying official split: "
          f"{len(by_split['train'])} train / {len(by_split['val'])} val / {len(by_split['test'])} test "
          f"(capped at {args.max_per_split})")

    final = build_and_cache_graphs(
        by_split, args.audio_root, sr=cfg["data"]["sample_rate"],
        n_chroma=cfg["data"]["n_chroma"], window_frames=8, out_dir=processed_dir,
    )
    print(f"cached graphs to {processed_dir}")

    split_path = os.path.join(splits_dir, "mtt_split.json")
    with open(split_path, "w") as f:
        json.dump({"tag_vocab": top_tags, "splits": final}, f, indent=2)
    print(f"wrote split ({len(final['train'])} train / {len(final['val'])} val / "
          f"{len(final['test'])} test) to {split_path}")


if __name__ == "__main__":
    main()
