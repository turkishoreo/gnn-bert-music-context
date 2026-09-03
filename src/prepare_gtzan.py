"""
Real-data loader for GTZAN (Section 3 Table 1: "Easy baseline" audio dataset).

Scans data/raw/gtzan/genres_original/<genre>/*.wav, extracts chroma features
per track (audio_features.py), builds a segment graph per track (graph_builder.py),
caches each graph to data/processed/gtzan_graphs/*.pt (satisfies the spec's
Section 10 deliverable: "at least 20 example .pt/.json graphs"), and writes a
train/val/test split to data/splits/gtzan_split.json.

Usage:
    python prepare_gtzan.py --config ../config.yaml
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

import audio_features as af
import graph_builder as gb

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def find_wav_files(raw_dir):
    items = []
    for genre in GENRES:
        genre_dir = os.path.join(raw_dir, "genres_original", genre)
        if not os.path.isdir(genre_dir):
            print(f"WARNING: missing genre folder {genre_dir}, skipping")
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav"):
                items.append((os.path.join(genre_dir, fname), genre))
    return items


FIXED_MEL_FRAMES = 640  # ~15s of audio at sr=22050, hop=512; tracks are trimmed/padded to this


def _fixed_size_mel(mel, n_frames=FIXED_MEL_FRAMES):
    """Crop or zero-pad a (n_mels, frames) log-mel array to a fixed width so the
    CNN baseline can batch tracks of slightly different lengths together."""
    if mel.shape[1] >= n_frames:
        return mel[:, :n_frames]
    pad = np.zeros((mel.shape[0], n_frames - mel.shape[1]), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=1)


def build_and_cache_graphs(items, sr, n_mels, n_chroma, window_frames, graph_out_dir, mel_out_dir):
    os.makedirs(graph_out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)
    records = []
    skipped = []
    for path, genre in tqdm(items, desc="extracting features + building graphs"):
        track_id = os.path.splitext(os.path.basename(path))[0]
        graph_path = os.path.join(graph_out_dir, f"{track_id}.pt")
        mel_path = os.path.join(mel_out_dir, f"{track_id}.pt")
        try:
            if not os.path.exists(graph_path) or not os.path.exists(mel_path):
                y = af.load_audio(path, sr=sr)
                chroma = af.chroma_features(y, sr=sr, n_chroma=n_chroma)
                graph = gb.segment_graph(chroma, window=window_frames)
                torch.save(graph, graph_path)

                mel = af.log_mel_spectrogram(y, sr=sr, n_mels=n_mels)
                mel = _fixed_size_mel(mel)
                torch.save(torch.tensor(mel, dtype=torch.float32), mel_path)
            records.append({"track_id": track_id, "genre": genre,
                             "graph_path": graph_path, "mel_path": mel_path})
        except Exception as e:
            # GTZAN ships at least one known-corrupted file (jazz.00054.wav);
            # skip rather than crash the whole extraction run.
            skipped.append({"path": path, "error": str(e)})
    if skipped:
        print(f"skipped {len(skipped)} unreadable file(s): "
              f"{[s['path'] for s in skipped]}")
    return records


def make_split(records, val_frac=0.15, test_frac=0.15, seed=42):
    """GTZAN has no artist metadata, so this is a random per-track split
    (not artist-disjoint like FMA/MagnaTagATune -- note this limitation
    in your report if you use GTZAN as your primary dataset)."""
    r = random.Random(seed)
    by_genre = {}
    for rec in records:
        by_genre.setdefault(rec["genre"], []).append(rec)

    split = {"train": [], "val": [], "test": []}
    for genre, recs in by_genre.items():
        r.shuffle(recs)
        n = len(recs)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        split["val"] += recs[:n_val]
        split["test"] += recs[n_val:n_val + n_test]
        split["train"] += recs[n_val + n_test:]
    for k in split:
        r.shuffle(split[k])
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--raw_dir", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.raw_dir:
        raw_dir = os.path.normpath(args.raw_dir)
    else:
        raw_dir = os.path.normpath(os.path.join(
            os.path.dirname(args.config), cfg["data"]["raw_dir"], "gtzan"))
    processed_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                   cfg["data"]["processed_dir"], "gtzan_graphs"))
    mel_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                             cfg["data"]["processed_dir"], "gtzan_mel"))
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))
    os.makedirs(splits_dir, exist_ok=True)

    print(f"scanning {raw_dir} ...")
    items = find_wav_files(raw_dir)
    print(f"found {len(items)} .wav files across {len(GENRES)} genres")
    if len(items) == 0:
        raise SystemExit(f"No .wav files found under {raw_dir}. "
                          f"Check that data/raw/gtzan/genres_original/<genre>/*.wav exists.")

    records = build_and_cache_graphs(
        items, sr=cfg["data"]["sample_rate"], n_mels=cfg["data"]["n_mels"],
        n_chroma=cfg["data"]["n_chroma"], window_frames=8,
        graph_out_dir=processed_dir, mel_out_dir=mel_dir,
    )
    print(f"cached {len(records)} graphs to {processed_dir}")
    print(f"cached {len(records)} mel-spectrograms to {mel_dir}")

    split = make_split(records)
    split_path = os.path.join(splits_dir, "gtzan_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"wrote split ({len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test) to {split_path}")


if __name__ == "__main__":
    main()
