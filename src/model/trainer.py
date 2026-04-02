"""Train outlier detection model on traffic counting data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    COUNTING_COLUMN,
    TIME_RANGE_COLUMN,
    CATEGORY_COLUMN,
    MODEL_TYPE,
    SEASONAL_FREQ,
    SERIES_RESAMPLE_FREQ,
)


def train_detector(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_type: str | None = None,
    group_column: str | None = None,
):
    """
    Train outlier detection model on counting data.

    model_type: "seasonal" (ADTK), "isolation_forest", "lstm" (local .pt), "chronos" (HF Chronos-2).
    group_column: for seasonal, one detector per distinct value (default: category).
    """
    model_type = model_type or MODEL_TYPE
    if model_type == "seasonal":
        return _train_seasonal(df, group_column=group_column)
    if model_type == "lstm":
        from src.model.lstm_pretrained import train_lstm_placeholder

        return train_lstm_placeholder(df)
    if model_type == "chronos":
        from src.model.chronos_hf import train_chronos_placeholder

        return train_chronos_placeholder(df)
    return _train_isolation_forest(df, feature_columns or [COUNTING_COLUMN])


def detect_anomalies(
    clf,
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_type: str | None = None,
    group_column: str | None = None,
):
    """Run trained detector and return anomaly scores and labels."""
    model_type = model_type or MODEL_TYPE
    if model_type == "seasonal":
        return _detect_seasonal(clf, df, group_column=group_column)
    if model_type == "lstm":
        from src.model.lstm_pretrained import detect_with_lstm

        return detect_with_lstm(clf, df, group_column=group_column)
    if model_type == "chronos":
        from src.model.chronos_hf import detect_with_chronos

        return detect_with_chronos(clf, df, group_column=group_column)
    return _detect_isolation_forest(clf, df, feature_columns or [COUNTING_COLUMN])


def _regularize_series(ts: pd.Series, freq: str) -> pd.Series:
    """
    Build a strictly regular time index so ADTK can infer frequency.
    Sums duplicate timestamps, resamples to fixed step (e.g. 15min), fills gaps with 0.
    Values are float64: int counts from SQL would otherwise make ADTK SeasonalAD fail
    (internal seasonal means are floats; assigning into int64 raises LossySetitemError).
    """
    ts = ts.sort_index()
    if ts.index.duplicated().any():
        ts = ts.groupby(ts.index).sum()
    out = ts.resample(freq).sum()
    return out.fillna(0).astype(np.float64)


def _train_seasonal(df: pd.DataFrame, group_column: str | None = None):
    """
    Train ADTK SeasonalAD - traffic-specific model that detects violations
    of weekly and daily patterns (validated on NYC taxi data).
    """
    from adtk.detector import SeasonalAD
    from adtk.data import validate_series

    gc = group_column or CATEGORY_COLUMN
    freq = SERIES_RESAMPLE_FREQ
    detectors = {}

    for key, group in df.groupby(gc):
        ts = group.set_index(TIME_RANGE_COLUMN)[COUNTING_COLUMN]
        ts = _regularize_series(ts, freq)
        ts = validate_series(ts)
        detector = SeasonalAD(freq=SEASONAL_FREQ)
        detector.fit(ts)
        detectors[key] = detector
    return detectors


def _detect_seasonal(detectors: dict, df: pd.DataFrame, group_column: str | None = None):
    """Run SeasonalAD per group key (category or direction) and merge results."""
    from adtk.data import validate_series

    gc = group_column or CATEGORY_COLUMN
    freq = SERIES_RESAMPLE_FREQ
    scores = np.zeros(len(df))
    labels = np.zeros(len(df), dtype=np.int32)

    for key, detector in detectors.items():
        mask = df[gc] == key
        if not mask.any():
            continue
        group = df.loc[mask]
        ts = group.set_index(TIME_RANGE_COLUMN)[COUNTING_COLUMN]
        ts = _regularize_series(ts, freq)
        ts = validate_series(ts)
        anomalies = detector.detect(ts)
        if anomalies is None or anomalies.empty:
            continue
        anomaly_mask = anomalies.fillna(False).astype(bool)
        for idx in group.index:
            ts_val = pd.Timestamp(group.loc[idx, TIME_RANGE_COLUMN]).floor(freq)
            if ts_val in anomaly_mask.index and bool(anomaly_mask.loc[ts_val]):
                scores[df.index.get_loc(idx)] = 1.0
                labels[df.index.get_loc(idx)] = 1

    return scores, labels


def _train_isolation_forest(df: pd.DataFrame, feature_columns: list[str]):
    """Train IsolationForest for generic tabular outlier detection."""
    X = df[feature_columns].fillna(0).values
    clf = IsolationForest(random_state=42, contamination=0.05)
    clf.fit(X)
    return clf


def _detect_isolation_forest(clf, df: pd.DataFrame, feature_columns: list[str]):
    """Run IsolationForest and return scores and labels."""
    X = df[feature_columns].fillna(0).values
    scores = -clf.score_samples(X)
    pred = clf.predict(X)
    labels = (pred == -1).astype(np.int32)
    return scores, labels
