"""
Write a minimal compatible LSTM checkpoint for local testing (random weights).

Usage:
  python scripts/create_dummy_lstm_checkpoint.py models/lstm_traffic.pt

Match config: LSTM_SEQ_LEN, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.lstm_pretrained import LSTMAutoencoder
from config import LSTM_SEQ_LEN, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/lstm_traffic.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    m = LSTMAutoencoder(hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS)
    ckpt = {
        "state_dict": m.state_dict(),
        "seq_len": LSTM_SEQ_LEN,
        "hidden_size": LSTM_HIDDEN_SIZE,
        "num_layers": LSTM_NUM_LAYERS,
    }
    torch.save(ckpt, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
