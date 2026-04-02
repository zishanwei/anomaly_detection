"""
Pre-trained LSTM for time-series anomaly scoring (reconstruction error).

Checkpoint format (torch.save dict):
  state_dict: model weights (required)
  seq_len, hidden_size, num_layers: architecture (defaults from config if missing)

Use scripts/create_dummy_lstm_checkpoint.py to generate a compatible .pt for testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from src.model.torch_device import describe_device, get_torch_device

from config import (
    COUNTING_COLUMN,
    TIME_RANGE_COLUMN,
    CATEGORY_COLUMN,
    LSTM_PRETRAINED_PATH,
    LSTM_SEQ_LEN,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_ANOMALY_QUANTILE,
)


class LSTMAutoencoder(nn.Module):
    """Many-to-many LSTM reconstructing the input sequence (one feature: count)."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


def _load_checkpoint(path: str) -> dict:
    dev = get_torch_device()
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    return {"state_dict": ckpt}


def build_model_from_checkpoint(ckpt: dict) -> tuple[LSTMAutoencoder, dict]:
    seq_len = int(ckpt.get("seq_len", LSTM_SEQ_LEN))
    hidden = int(ckpt.get("hidden_size", LSTM_HIDDEN_SIZE))
    num_layers = int(ckpt.get("num_layers", LSTM_NUM_LAYERS))
    model = LSTMAutoencoder(hidden_size=hidden, num_layers=num_layers)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    meta = {"seq_len": seq_len, "hidden_size": hidden, "num_layers": num_layers}
    return model, meta


def load_pretrained_lstm(path: str | None = None) -> dict:
    """Load model + metadata (inference only; no training)."""
    path = path or LSTM_PRETRAINED_PATH
    if not path:
        raise ValueError("Set LSTM_PRETRAINED_PATH in config or pass path=")
    ckpt = _load_checkpoint(path)
    model, meta = build_model_from_checkpoint(ckpt)
    model.eval()
    return {
        "model": model,
        "path": path,
        "seq_len": meta["seq_len"],
    }


def _make_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Sliding windows (n - seq_len + 1, seq_len, 1)."""
    n = len(values)
    if n < seq_len:
        return np.empty((0, seq_len, 1), dtype=np.float32)
    rows = n - seq_len + 1
    out = np.zeros((rows, seq_len, 1), dtype=np.float32)
    for i in range(rows):
        out[i, :, 0] = values[i : i + seq_len]
    return out


def _score_sequences(model: nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-window reconstruction MSE (mean over time steps)."""
    if len(sequences) == 0:
        return np.array([])
    with torch.no_grad():
        x = torch.from_numpy(sequences).to(device)
        recon = model(x)
        err = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return err


def detect_with_lstm(
    bundle: dict,
    df: pd.DataFrame,
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row reconstruction-error scores; labels via quantile threshold on scored windows.
    Rows without a full window keep score 0 and label 0.
    group_column: one series per distinct value (default category).
    """
    gc = group_column or CATEGORY_COLUMN
    device = get_torch_device()
    print(f"[LSTM] device: {describe_device()} (group by {gc})", flush=True)
    model = bundle["model"].to(device)
    seq_len = bundle["seq_len"]
    n = len(df)
    scores = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int32)
    iloc_of = {ix: i for i, ix in enumerate(df.index)}

    collected: list[tuple[int, float]] = []

    for _, group in df.groupby(gc, sort=False):
        group = group.sort_values(TIME_RANGE_COLUMN)
        idx_order = group.index.tolist()
        counts = group[COUNTING_COLUMN].astype(float).values.reshape(-1, 1)
        scaler = MinMaxScaler()
        counts_n = scaler.fit_transform(counts)

        seqs = _make_sequences(counts_n.ravel(), seq_len)
        if len(seqs) == 0:
            continue
        errs = _score_sequences(model, seqs, device)
        for w, err in enumerate(errs):
            row_in_group = w + seq_len - 1
            orig_idx = idx_order[row_in_group]
            pos = iloc_of[orig_idx]
            scores[pos] = float(err)
            collected.append((pos, float(err)))

    if not collected:
        return scores, labels

    err_vals = [c[1] for c in collected]
    thresh = float(np.quantile(err_vals, LSTM_ANOMALY_QUANTILE))
    if thresh <= 0:
        thresh = max(err_vals) + 1e-8

    for pos, err in collected:
        if err >= thresh:
            labels[pos] = 1

    return scores, labels


def train_lstm_placeholder(_df: pd.DataFrame) -> dict:
    """Load pre-trained weights only (no training in this pipeline)."""
    return load_pretrained_lstm()
