import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import Dataset

FLIP_COUNTS = [10, 25, 50, 100, 200, 300, 500]
EVAL_PROBABILITIES = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]


class FastCoinFlipDataset(Dataset):
    """
    High-performance pre-generated dataset for coin flips.
    - 50% fair coins (p = 0.5, label = 0)
    - 50% rigged coins (p sampled uniformly from [0.51, 0.95], label = 1)
    - Lengths N sampled uniformly from FLIP_COUNTS
    """
    def __init__(self, num_samples=30000, seed=42):
        rng = np.random.RandomState(seed)
        self.num_samples = num_samples

        # Sample lengths and labels
        self.lengths = rng.choice(FLIP_COUNTS, size=num_samples).astype(np.int64)
        self.is_rigged = (rng.rand(num_samples) < 0.5).astype(np.float32)
        self.p_biases = np.where(
            self.is_rigged == 1.0,
            rng.uniform(0.51, 0.95, size=num_samples),
            0.5
        ).astype(np.float32)

        # Pre-allocate storage
        self.x_last20 = np.zeros((num_samples, 20), dtype=np.float32)
        self.valid_b = np.zeros(num_samples, dtype=np.float32)
        self.head_ratio = np.zeros((num_samples, 1), dtype=np.float32)
        self.total_flips = self.lengths.reshape(-1, 1).astype(np.float32)
        self.labels = self.is_rigged.reshape(-1, 1).astype(np.float32)

        # Store raw flips as list of arrays
        self.flips_list = []

        for i in range(num_samples):
            n = self.lengths[i]
            p = self.p_biases[i]
            flips = rng.binomial(1, p, size=n).astype(np.float32)
            self.flips_list.append(flips)

            # Model B features
            if n >= 20:
                self.x_last20[i] = flips[-20:]
                self.valid_b[i] = 20.0
            else:
                self.x_last20[i, 20 - n:] = flips
                self.valid_b[i] = float(n)

            # Model C features
            self.head_ratio[i, 0] = np.sum(flips) / float(n)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "flips": self.flips_list[idx],
            "n": self.lengths[idx],
            "last20": self.x_last20[idx],
            "valid_b": self.valid_b[idx],
            "head_ratio": self.head_ratio[idx],
            "total_flips": self.total_flips[idx],
            "label": self.labels[idx]
        }


# Backward compatibility alias
CoinFlipDataset = FastCoinFlipDataset


def collate_coin_flips(batch):
    lengths = [item["n"] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Pad Model A sequences to max_len of this batch
    padded_flips = np.zeros((batch_size, max_len), dtype=np.float32)
    for i, item in enumerate(batch):
        l = item["n"]
        padded_flips[i, :l] = item["flips"]

    x_last20 = np.stack([item["last20"] for item in batch], axis=0)
    valid_b = np.array([item["valid_b"] for item in batch], dtype=np.float32)
    head_ratio = np.stack([item["head_ratio"] for item in batch], axis=0)
    total_flips = np.stack([item["total_flips"] for item in batch], axis=0)
    labels = np.stack([item["label"] for item in batch], axis=0)

    return {
        "x_a": torch.from_numpy(padded_flips),
        "lengths_a": torch.tensor(lengths, dtype=torch.long),
        "x_b": torch.from_numpy(x_last20),
        "valid_b": torch.from_numpy(valid_b),
        "head_ratio_c": torch.from_numpy(head_ratio),
        "total_flips_c": torch.from_numpy(total_flips),
        "labels": torch.from_numpy(labels)
    }


def generate_evaluation_batch(p, n, num_trials=500, seed=123):
    """
    Generates a batch of trials for a fixed (p, N) evaluation cell.
    """
    rng = np.random.RandomState(seed)
    flips = rng.binomial(1, p, size=(num_trials, n)).astype(np.float32)

    # Model A:
    x_a = torch.from_numpy(flips)
    lengths_a = torch.full((num_trials,), n, dtype=torch.long)

    # Model B:
    if n >= 20:
        last20 = flips[:, -20:]
        valid_b = np.full((num_trials,), 20.0, dtype=np.float32)
    else:
        last20 = np.pad(flips, ((0, 0), (20 - n, 0)), mode="constant", constant_values=0)
        valid_b = np.full((num_trials,), float(n), dtype=np.float32)

    x_b = torch.from_numpy(last20)
    valid_b = torch.from_numpy(valid_b)

    # Model C:
    k = np.sum(flips, axis=1, keepdims=True)
    head_ratio = k / float(n)
    head_ratio_c = torch.from_numpy(head_ratio)
    total_flips_c = torch.full((num_trials, 1), float(n), dtype=torch.float32)

    labels = torch.full((num_trials, 1), 1.0 if p > 0.5 else 0.0, dtype=torch.float32)

    return {
        "x_a": x_a,
        "lengths_a": lengths_a,
        "x_b": x_b,
        "valid_b": valid_b,
        "head_ratio_c": head_ratio_c,
        "total_flips_c": total_flips_c,
        "labels": labels,
        "k_counts": k.squeeze(-1)
    }
