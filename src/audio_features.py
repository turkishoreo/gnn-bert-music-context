"""
Audio preprocessing, per spec Section 3 ("Preprocessing pipeline", steps 1-2):
  1. Resample to 22,050 Hz; extract log-mel spectrogram (128 bins) or chroma (12 bins);
     normalize per track.
  2. Split each track into fixed windows (5-10s) or beat-synchronous segments (librosa).

This module operates on real audio files. For datasets/testing without audio on disk,
use src/synthetic_data.py instead.
"""
import numpy as np
import librosa


def load_audio(path, sr=22050):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def log_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _normalize(log_mel)


def chroma_features(y, sr=22050, n_chroma=12, hop_length=512):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    return _normalize(chroma)


def _normalize(feat):
    mu, sigma = feat.mean(), feat.std() + 1e-8
    return ((feat - mu) / sigma).astype(np.float32)


def fixed_windows(y, sr=22050, window_seconds=5):
    """Split waveform into fixed-length windows (spec step 2, non-beat-synchronous option)."""
    win = int(window_seconds * sr)
    n_windows = max(1, len(y) // win)
    return [y[i * win:(i + 1) * win] for i in range(n_windows)]


def beat_synchronous_segments(y, sr=22050):
    """Split waveform at detected beat frames (spec step 2, beat-synchronous option)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)
    bounds = [0] + list(beat_samples) + [len(y)]
    return [y[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def estimate_chord_sequence(chroma):
    """Toy chord estimator: nearest of 24 major/minor triad templates per frame.
    Good enough to feed graph_builder.chord_transition_graph; swap in a proper
    chord-recognition model (e.g. madmom, Chordino) for anything you plan to report."""
    templates = _triad_templates()  # (24, 12)
    # chroma: (12, frames) -> (frames, 12)
    frames = chroma.T
    sims = frames @ templates.T  # (frames, 24)
    return sims.argmax(axis=1)


def _triad_templates():
    major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    templates = []
    for shift in range(12):
        templates.append(np.roll(major, shift))
    for shift in range(12):
        templates.append(np.roll(minor, shift))
    return np.stack(templates)


if __name__ == "__main__":
    print("This module expects real audio files (see docstring). "
          "For an offline demo, run src/synthetic_data.py via train.py --synthetic.")
