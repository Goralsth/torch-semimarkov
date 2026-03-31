#!/usr/bin/env python3
"""TIMIT data loading, feature extraction, and dataset classes."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import importlib.util

    HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None
except ImportError:
    HAS_SOUNDFILE = False

logger = logging.getLogger(__name__)


# =============================================================================
# TIMIT Phone Set and Mappings
# =============================================================================

# Original 61 TIMIT phonemes
TIMIT_61_PHONES = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ax-h",
    "axr",
    "ay",
    "b",
    "bcl",
    "ch",
    "d",
    "dcl",
    "dh",
    "dx",
    "eh",
    "el",
    "em",
    "en",
    "eng",
    "epi",
    "er",
    "ey",
    "f",
    "g",
    "gcl",
    "h#",
    "hh",
    "hv",
    "ih",
    "ix",
    "iy",
    "jh",
    "k",
    "kcl",
    "l",
    "m",
    "n",
    "ng",
    "nx",
    "ow",
    "oy",
    "p",
    "pau",
    "pcl",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "tcl",
    "th",
    "uh",
    "uw",
    "ux",
    "v",
    "w",
    "y",
    "z",
    "zh",
]

# Standard 39-phone folding (Lee & Hon, 1989; Kaldi/ESPnet convention)
# Maps 61 phones to 39 classes
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONE_61_TO_39 = {
    # Vowels
    "iy": "iy",
    "ih": "ih",
    "eh": "eh",
    "ae": "ae",
    "ah": "ah",
    "uw": "uw",
    "uh": "uh",
    "aa": "aa",
    "ey": "ey",
    "ay": "ay",
    "oy": "oy",
    "aw": "aw",
    "ow": "ow",
    "er": "er",
    "ao": "aa",  # ao folds to aa (Kaldi convention)
    # Vowel reductions
    "ax": "ah",
    "ix": "ih",
    "axr": "er",
    "ax-h": "ah",
    "ux": "uw",
    # Semivowels
    "l": "l",
    "r": "r",
    "w": "w",
    "y": "y",
    "el": "l",
    "hh": "hh",
    "hv": "hh",
    # Nasals
    "m": "m",
    "n": "n",
    "ng": "ng",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "nx": "n",
    # Fricatives
    "f": "f",
    "th": "th",
    "s": "s",
    "sh": "sh",
    "v": "v",
    "dh": "dh",
    "z": "z",
    "zh": "sh",  # zh folds to sh (Kaldi convention)
    # Affricates
    "ch": "ch",
    "jh": "jh",
    # Stops
    "p": "p",
    "t": "t",
    "k": "k",
    "b": "b",
    "d": "d",
    "g": "g",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "dx": "dx",
    "q": "sil",
    # Silence
    "pau": "sil",
    "epi": "sil",
    "h#": "sil",
}

# Standard 39-phone set (Kaldi/ESPnet convention, Lee & Hon 1989)
# Reference: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
PHONES_39 = [
    "aa",
    "ae",
    "ah",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "dx",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "sil",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
]

PHONE_TO_IDX = {p: i for i, p in enumerate(PHONES_39)}
NUM_PHONES = len(PHONES_39)
MAX_HEATMAP_LABELS = 39  # show all phone classes in posterior heatmaps

# Colorblind-friendly categorical palette generated with glasbey.create_block_palette.
# Phones grouped by manner of articulation so phonetically related sounds share
# a hue band:  vowels (warm reds/oranges), diphthongs (blues), stops (greens),
# affricates (pinks), fricatives (grays/tans), nasals (teals), liquids (purples),
# glides (browns), silence (magenta).
PHONE_COLORS_HEX = {
    "aa": "#590000",
    "ae": "#750028",
    "ah": "#96000c",
    "aw": "#00009e",
    "ay": "#1839e3",
    "b": "#102804",
    "ch": "#db45db",
    "d": "#004500",
    "dh": "#282428",
    "dx": "#b6f735",
    "eh": "#b62d08",
    "er": "#db3d39",
    "ey": "#317dfb",
    "f": "#393d49",
    "g": "#006900",
    "hh": "#595159",
    "ih": "#db590c",
    "iy": "#f3823d",
    "jh": "#ffa6ff",
    "k": "#318604",
    "l": "#410075",
    "m": "#00796d",
    "n": "#1cc6aa",
    "ng": "#65f7ba",
    "ow": "#69aaf3",
    "oy": "#aedff7",
    "p": "#61ae18",
    "r": "#860cce",
    "s": "#71696d",
    "sh": "#8e827d",
    "sil": "#9e0871",
    "t": "#8aca20",
    "th": "#a69679",
    "uh": "#fba249",
    "uw": "#fbc661",
    "v": "#c2b29a",
    "w": "#492000",
    "y": "#926100",
    "z": "#e7d2a6",
}
# Index-based lookup for plotting: phone_colors[label_idx] -> hex string
PHONE_COLORS = {i: PHONE_COLORS_HEX[p] for i, p in enumerate(PHONES_39)}

# Typical phone durations (in frames at 10ms)
# Useful for understanding expected semi-CRF behavior
TYPICAL_DURATIONS = {
    "sil": (5, 50),  # Silence: variable
    "aa": (5, 15),  # Vowels: longer
    "iy": (5, 15),
    "p": (2, 8),  # Stops: short
    "t": (2, 8),
    "k": (2, 8),
    "s": (4, 15),  # Fricatives: medium
    "sh": (4, 15),
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PhoneSegment:
    """A phone segment with timing."""

    start_sample: int
    end_sample: int
    phone_61: str
    phone_39: str
    label_idx: int


@dataclass
class Utterance:
    """A TIMIT utterance."""

    utterance_id: str
    speaker_id: str
    dialect_region: str
    wav_path: Path
    phones: list[PhoneSegment]


class SegmentAnnotation(NamedTuple):
    """A segment with label (for metrics)."""

    start: int
    end: int
    label: int


# =============================================================================
# TIMIT Parsing
# =============================================================================


def parse_phn_file(phn_path: Path) -> list[tuple[int, int, str]]:
    """
    Parse a TIMIT .PHN file.

    Returns list of (start_sample, end_sample, phone) tuples.
    """
    phones = []
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = parts
                phones.append((int(start), int(end), phone.lower()))
    return phones


def load_timit_split(
    timit_dir: Path,
    split: Literal["train", "test"],
) -> list[Utterance]:
    """
    Load all utterances from a TIMIT split.
    """
    split_dir = timit_dir / split.upper()
    if not split_dir.exists():
        raise FileNotFoundError(f"TIMIT split directory not found: {split_dir}")

    utterances = []

    # Iterate through dialect regions
    for dr_dir in sorted(split_dir.iterdir()):
        if not dr_dir.is_dir() or not dr_dir.name.startswith("DR"):
            continue

        dialect_region = dr_dir.name

        # Iterate through speakers
        for speaker_dir in sorted(dr_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            # Find all .PHN files
            for phn_path in speaker_dir.glob("*.PHN"):
                # Skip SA sentences (read by all speakers, causes speaker bias)
                if phn_path.stem.upper().startswith("SA"):
                    continue

                wav_path = phn_path.with_suffix(".WAV")
                if not wav_path.exists():
                    # Try lowercase
                    wav_path = phn_path.with_suffix(".wav")

                if not wav_path.exists():
                    logger.warning(f"WAV not found for {phn_path}")
                    continue

                # Parse phone file
                raw_phones = parse_phn_file(phn_path)

                # Convert to PhoneSegments
                phone_segments = []
                for start, end, phone_61 in raw_phones:
                    phone_39 = PHONE_61_TO_39.get(phone_61, "sil")
                    label_idx = PHONE_TO_IDX.get(phone_39, PHONE_TO_IDX["sil"])

                    phone_segments.append(
                        PhoneSegment(
                            start_sample=start,
                            end_sample=end,
                            phone_61=phone_61,
                            phone_39=phone_39,
                            label_idx=label_idx,
                        )
                    )

                utterance_id = f"{dialect_region}_{speaker_id}_{phn_path.stem}"

                utterances.append(
                    Utterance(
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        dialect_region=dialect_region,
                        wav_path=wav_path,
                        phones=phone_segments,
                    )
                )

    logger.info(f"Loaded {len(utterances)} utterances from {split} split")
    return utterances


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_mfcc_features(
    audio_path: Path,
    n_mfcc: int = 13,
    n_fft: int = 400,  # 25ms at 16kHz
    hop_length: int = 160,  # 10ms at 16kHz
    sample_rate: int = 16000,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Returns:
        features: (T, D) where D = n_mfcc * 3 if include_deltas else n_mfcc
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if include_deltas:
        # Add delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])
    else:
        features = mfcc

    # Transpose to (T, D)
    return features.T


def extract_mel_features(
    audio_path: Path,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extract log mel spectrogram features.

    Returns:
        features: (T, n_mels)
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    y, sr = librosa.load(audio_path, sr=sample_rate)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Log compression
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.T


def samples_to_frames(
    sample_idx: int,
    hop_length: int = 160,
) -> int:
    """Convert sample index to frame index."""
    return sample_idx // hop_length


def align_phones_to_frames(
    phones: list[PhoneSegment],
    num_frames: int,
    hop_length: int = 160,
) -> tuple[np.ndarray, list[SegmentAnnotation]]:
    """
    Align phone segments to frame indices.

    Returns:
        labels: (T,) array of phone indices
        segments: list of SegmentAnnotation
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    segments = []

    for phone in phones:
        start_frame = samples_to_frames(phone.start_sample, hop_length)
        end_frame = samples_to_frames(phone.end_sample, hop_length)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))

        labels[start_frame:end_frame] = phone.label_idx
        segments.append(SegmentAnnotation(start_frame, end_frame, phone.label_idx))

    return labels, segments


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_timit(
    timit_dir: Path,
    output_dir: Path,
    feature_type: Literal["mfcc", "mel"] = "mfcc",
    n_mfcc: int = 13,
    n_mels: int = 80,
    hop_length: int = 160,
):
    """
    Preprocess TIMIT dataset into train/test splits.
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        logger.info(f"Processing {split} split...")

        utterances = load_timit_split(timit_dir, split)

        processed = []
        segment_lengths = defaultdict(list)

        for utt in utterances:
            try:
                # Extract features
                if feature_type == "mfcc":
                    features = extract_mfcc_features(
                        utt.wav_path,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                else:
                    features = extract_mel_features(
                        utt.wav_path,
                        n_mels=n_mels,
                        hop_length=hop_length,
                    )

                num_frames = len(features)

                # Align phones to frames
                labels, segments = align_phones_to_frames(utt.phones, num_frames, hop_length)

                # Collect segment statistics
                for seg in segments:
                    segment_lengths[seg.label].append(seg.end - seg.start)

                processed.append(
                    {
                        "utterance_id": utt.utterance_id,
                        "speaker_id": utt.speaker_id,
                        "features": features.tolist(),
                        "labels": labels.tolist(),
                        "segments": [(s.start, s.end, s.label) for s in segments],
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to process {utt.utterance_id}: {e}")
                continue

        # Save processed data
        output_file = output_dir / f"{split}.jsonl"
        logger.info(f"Saving {len(processed)} utterances to {output_file}")

        with open(output_file, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")

        # Save statistics
        stats = {}
        for label_idx, lengths in segment_lengths.items():
            lengths = np.array(lengths)
            phone_name = PHONES_39[label_idx] if label_idx < len(PHONES_39) else "unk"
            stats[phone_name] = {
                "count": len(lengths),
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "p95": float(np.percentile(lengths, 95)),
            }

        stats_file = output_dir / f"{split}_segment_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_file}")

    # Save phone mapping
    mapping_file = output_dir / "phone_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "phones_39": PHONES_39,
                "phone_to_idx": PHONE_TO_IDX,
                "phone_61_to_39": PHONE_61_TO_39,
            },
            f,
            indent=2,
        )

    logger.info("Preprocessing complete!")


# =============================================================================
# Dataset
# =============================================================================


def compute_normalization_stats(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean and std from training data.

    Args:
        data_path: Path to JSONL data file

    Returns:
        mean: (feature_dim,) array of per-dimension means
        std: (feature_dim,) array of per-dimension standard deviations
    """
    all_features = []
    with open(data_path) as f:
        for line in f:
            utt = json.loads(line)
            features = np.array(utt["features"], dtype=np.float32)
            all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    # Avoid division by zero for constant features
    std = np.maximum(std, 1e-8)
    return mean, std


class TIMITDataset(Dataset):
    """Dataset for TIMIT phoneme segmentation."""

    def __init__(
        self,
        data_path: Path,
        max_length: int | None = None,
        normalize: bool = True,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        """
        Initialize TIMIT dataset.

        Args:
            data_path: Path to JSONL data file
            max_length: Optional maximum sequence length (truncates longer sequences)
            normalize: Whether to apply z-score normalization to features
            mean: Pre-computed per-dimension means (if None and normalize=True, computed from data)
            std: Pre-computed per-dimension stds (if None and normalize=True, computed from data)
        """
        self.max_length = max_length
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.utterances = []

        with open(data_path) as f:
            for line in f:
                self.utterances.append(json.loads(line))

        # Compute stats from this dataset if not provided and normalization is requested
        if self.normalize and self.mean is None:
            logger.info("Computing normalization statistics from data...")
            self.mean, self.std = compute_normalization_stats(data_path)
            logger.info(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
            logger.info(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")

        logger.info(f"Loaded {len(self.utterances)} utterances from {data_path}")

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        utt = self.utterances[idx]

        features = np.array(utt["features"], dtype=np.float32)
        labels = np.array(utt["labels"], dtype=np.int64)

        # Apply z-score normalization
        if self.normalize and self.mean is not None:
            features = (features - self.mean) / self.std

        # Truncate if needed
        if self.max_length and len(features) > self.max_length:
            features = features[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "length": torch.tensor(len(features), dtype=torch.long),
            "utterance_id": utt["utterance_id"],
        }


def collate_timit(batch: list[dict], fixed_length: int | None = None) -> dict[str, Tensor]:
    """Collate TIMIT batch with padding.

    Args:
        batch: List of sample dictionaries.
        fixed_length: If provided, force all sequences to this exact length.
            Sequences shorter are padded, longer are truncated. This is useful
            for debugging to eliminate variable-length boundary handling issues.
    """
    if fixed_length is not None:
        max_len = fixed_length
    else:
        max_len = max(b["length"].item() for b in batch)

    features = []
    labels = []
    lengths = []
    utterance_ids = []

    for b in batch:
        feat = b["features"]
        lab = b["labels"]
        seq_len = b["length"].item()

        # Handle fixed length: truncate or pad
        if fixed_length is not None:
            if seq_len > fixed_length:
                # Truncate
                feat = feat[:fixed_length]
                lab = lab[:fixed_length]
                seq_len = fixed_length
            elif seq_len < fixed_length:
                # Pad
                pad_len = fixed_length - seq_len
                feat = F.pad(feat, (0, 0, 0, pad_len))
                lab = F.pad(lab, (0, pad_len), value=0)
            # For fixed length, report the fixed length as the actual length
            lengths.append(torch.tensor(fixed_length, dtype=torch.long))
        else:
            # Variable length: just pad to max_len in batch
            if seq_len < max_len:
                pad_len = max_len - seq_len
                feat = F.pad(feat, (0, 0, 0, pad_len))
                lab = F.pad(lab, (0, pad_len), value=0)
            lengths.append(b["length"])

        features.append(feat)
        labels.append(lab)
        utterance_ids.append(b["utterance_id"])

    return {
        "features": torch.stack(features),
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths),
        "utterance_ids": utterance_ids,
    }


def make_collate_fn(fixed_length: int | None = None):
    """Create a collate function with optional fixed length.

    Args:
        fixed_length: If provided, force all sequences to this length.

    Returns:
        A collate function suitable for DataLoader.
    """

    def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
        return collate_timit(batch, fixed_length=fixed_length)

    return collate_fn
