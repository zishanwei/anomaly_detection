"""Identify reasons behind detected anomalies (data lack, weather, weekday)."""

import pandas as pd
import numpy as np


REASON_DATA_LACK = "data_lack"
REASON_WEEKDAY = "weekday"
REASON_WEATHER = "weather"
REASON_UNKNOWN = "unknown"


def identify_reasons(
    df: pd.DataFrame,
    anomaly_mask: np.ndarray,
    counting_col: str = "counting",
) -> list[str]:
    """
    Identify likely reasons for each anomaly.

    Reasons: data_lack (missing/low counts), weekday (unusual day pattern),
    weather (placeholder - extend with weather data when available).
    """
    reasons = []
    for i in np.where(anomaly_mask)[0]:
        row = df.iloc[i]
        r = _reason_for_row(row, counting_col)
        reasons.append(r)
    return reasons


def _reason_for_row(row: pd.Series, counting_col: str) -> str:
    """Determine most likely reason for a single anomaly."""
    count = row.get(counting_col, 0)
    weekday = row.get("weekday", None)

    if pd.isna(count) or count == 0:
        return REASON_DATA_LACK

    if weekday is not None:
        if weekday >= 5:
            return REASON_WEEKDAY
        if count < 10:
            return REASON_DATA_LACK

    return REASON_UNKNOWN


def analyze_anomalies(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    counting_col: str = "counting",
) -> pd.DataFrame:
    """
    Build analysis report: anomalies with scores and identified reasons.
    """
    anomaly_mask = labels == 1
    anomaly_df = df[anomaly_mask].copy()
    anomaly_df["anomaly_score"] = scores[anomaly_mask]
    anomaly_df["reason"] = identify_reasons(df, anomaly_mask, counting_col)
    return anomaly_df
