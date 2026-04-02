"""
Pretrained time-series model from Hugging Face (Amazon Chronos family).

Anomaly score = |actual[t] - median forecast[t]| using context [t-L : t].
Not an LSTM: Chronos uses Transformer / encoder architectures.

Other OSS options for "deep" anomaly detection (rarely drop-in pretrained LSTM):
  PyOD, Aeon, sktime — mostly train-from-scratch; Merlion (Salesforce) has some pipelines.

Requires: pip install chronos-forecasting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.model.torch_device import (
    describe_device,
    get_hf_device_map,
    load_intel_extension_for_pytorch,
)

from config import (
    COUNTING_COLUMN,
    TIME_RANGE_COLUMN,
    CATEGORY_COLUMN,
    SERIES_RESAMPLE_FREQ,
    CHRONOS_MODEL_ID,
    CHRONOS_CONTEXT_LENGTH,
    CHRONOS_ANOMALY_QUANTILE,
)

def _log(msg: str) -> None:
    """Console progress (Chronos has no training step; messages state inference-only)."""
    print(f"[Chronos] {msg}", flush=True)


def _require_chronos():
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            "Install Chronos: pip install chronos-forecasting"
        ) from e
    return Chronos2Pipeline


def load_chronos_pipeline(model_id: str | None = None, device_map: str | None = None) -> dict:
    """Load pretrained Chronos-2 from Hugging Face (inference only). Uses Intel Extension for PyTorch on XPU."""
    Chronos2Pipeline = _require_chronos()
    mid = model_id or CHRONOS_MODEL_ID
    device_map = device_map or get_hf_device_map()
    ipex = None
    on_xpu = False

    if device_map == "xpu":
        ipex = load_intel_extension_for_pytorch()
        if ipex is None:
            _log(
                "intel_extension_for_pytorch not installed — XPU requires IPEX. "
                "pip install intel-extension-for-pytorch (match your torch build). Falling back to CPU."
            )
            device_map = "cpu"
        else:
            _log("Intel Extension for PyTorch (IPEX) loaded for Chronos on Intel XPU.")
            on_xpu = True

    _log(
        "Loading pretrained weights from Hugging Face (no training — Chronos is used for inference only). "
        "First download can take several minutes."
    )
    _log(f"Model id: {mid}, device_map={device_map} ({describe_device()})")
    try:
        pipeline = Chronos2Pipeline.from_pretrained(mid, device_map=device_map)
    except Exception as e:
        if device_map != "cpu":
            _log(f"Load failed on device_map={device_map!r}: {e}")
            _log("Retrying with device_map='cpu'.")
            pipeline = Chronos2Pipeline.from_pretrained(mid, device_map="cpu")
            device_map = "cpu"
            on_xpu = False
            ipex = None
        else:
            raise

    if on_xpu and device_map == "xpu" and ipex is not None:
        try:
            m = pipeline.model
            m.eval()
            pipeline.model = ipex.optimize(m, dtype=torch.float32)
            _log("Applied ipex.optimize() to Chronos2Model for XPU inference.")
        except Exception as ex:
            _log(f"ipex.optimize skipped (non-fatal): {ex}")
    q = getattr(pipeline, "quantiles", None)
    if q is None:
        median_idx = 0
    else:
        q = np.asarray(q, dtype=float)
        median_idx = int(np.argmin(np.abs(q - 0.5)))
    cl = min(CHRONOS_CONTEXT_LENGTH, getattr(pipeline, "model_context_length", CHRONOS_CONTEXT_LENGTH))
    _log(
        f"Ready. context_length={cl} (config CHRONOS_CONTEXT_LENGTH={CHRONOS_CONTEXT_LENGTH}), "
        f"median_quantile_index={median_idx}"
    )
    return {
        "pipeline": pipeline,
        "model_id": mid,
        "context_length": cl,
        "median_quantile_index": median_idx,
    }


def _aggregate_series(group: pd.DataFrame) -> pd.Series:
    g = group.groupby(TIME_RANGE_COLUMN, as_index=True)[COUNTING_COLUMN].sum().sort_index()
    s = g.astype(float)
    s.index = pd.to_datetime(s.index)
    if SERIES_RESAMPLE_FREQ:
        s = s.resample(SERIES_RESAMPLE_FREQ).sum().fillna(0)
    return s


def _floor_time(ts, freq: str | None) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.floor(freq) if freq else t


def detect_with_chronos(
    bundle: dict,
    df: pd.DataFrame,
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step-ahead forecast error per aggregated (group_key, time) bucket;
    scores/labels replicated to all rows sharing that bucket.
    group_column: series are split by this column (default: category); use direction for corridor-level detection.
    """
    gc = group_column or CATEGORY_COLUMN
    pipeline = bundle["pipeline"]
    cl = int(bundle["context_length"])
    med_idx = bundle["median_quantile_index"]

    n_grp = df[gc].nunique()
    _log(
        f"Anomaly scoring: {len(df)} rows, {n_grp} groups ({gc}) — running one-step forecasts "
        f"(not training)."
    )

    scores = np.zeros(len(df), dtype=np.float64)
    labels = np.zeros(len(df), dtype=np.int32)

    iloc_of = {ix: i for i, ix in enumerate(df.index)}
    bucket_err: dict[tuple, float] = {}

    for g_idx, (gkey, group) in enumerate(df.groupby(gc, sort=False), start=1):
        s = _aggregate_series(group)
        vals = s.values.astype(np.float32)
        times = s.index.to_numpy()
        if len(vals) <= cl:
            _log(
                f"  group {g_idx}/{n_grp} {gkey!r}: skip (only {len(vals)} points after resample; "
                f"need > context_length={cl})"
            )
            continue

        contexts: list[np.ndarray] = []
        keys: list[tuple] = []
        for i in range(cl, len(vals)):
            ctx = vals[i - cl : i]
            contexts.append(ctx)
            keys.append((gkey, _floor_time(times[i], SERIES_RESAMPLE_FREQ)))

        n_win = len(contexts)
        _log(
            f"  group {g_idx}/{n_grp} {gkey!r}: {len(vals)} resampled points, {n_win} forecast windows "
            f"(batch inference)"
        )

        chunk = 512
        preds_flat: list[float] = []
        n_batches = (n_win + chunk - 1) // chunk
        for b_idx, start in enumerate(range(0, len(contexts), chunk), start=1):
            batch = contexts[start : start + chunk]
            _log(f"    predict batch {b_idx}/{n_batches} ({len(batch)} windows)")
            out = pipeline.predict(
                batch,
                prediction_length=1,
                batch_size=min(256, len(batch)),
                context_length=cl,
            )
            for t in out:
                arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
                if arr.ndim == 3:
                    pred = float(arr[0, med_idx, 0])
                elif arr.ndim == 2:
                    pred = float(arr[med_idx, 0])
                else:
                    pred = float(arr.flat[0])
                preds_flat.append(pred)

        if len(preds_flat) != len(keys):
            _log(
                f"    warning: got {len(preds_flat)} predictions for {len(keys)} windows — "
                "truncating to min length"
            )
            m = min(len(preds_flat), len(keys))
            preds_flat = preds_flat[:m]
            keys = keys[:m]

        for j, (k, pred) in enumerate(zip(keys, preds_flat)):
            i = cl + j
            actual = float(vals[i])
            err = abs(actual - pred)
            bucket_err[k] = err

    if not bucket_err:
        _log("No buckets scored (all groups too short or empty).")
        return scores, labels

    err_list = list(bucket_err.values())
    thresh = float(np.quantile(err_list, CHRONOS_ANOMALY_QUANTILE))
    if thresh <= 0:
        thresh = max(err_list) + 1e-8

    _log(
        f"Thresholding: anomaly if forecast error >= {thresh:.6g} "
        f"(quantile={CHRONOS_ANOMALY_QUANTILE} over {len(bucket_err)} time buckets)"
    )

    tcol = TIME_RANGE_COLUMN
    if SERIES_RESAMPLE_FREQ:
        floor_t = pd.to_datetime(df[tcol]).dt.floor(SERIES_RESAMPLE_FREQ)
    else:
        floor_t = pd.to_datetime(df[tcol])

    for k, err in bucket_err.items():
        gkey, ts = k
        m = (df[gc] == gkey) & (floor_t == ts)
        for ix in df.index[m]:
            pos = iloc_of[ix]
            scores[pos] = err
            if err >= thresh:
                labels[pos] = 1

    n_flag = int(labels.sum())
    _log(f"Done. Rows labeled anomalous: {n_flag} / {len(df)}")
    return scores, labels


def train_chronos_placeholder(_df: pd.DataFrame) -> dict:
    _log(
        "train_detector step: Chronos does not train on your data — only loads pretrained weights, "
        f"then detect_anomalies runs batched forecasts (see messages below)."
    )
    return load_chronos_pipeline()
