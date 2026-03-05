"""Model training block - outlier detection for counting data."""

from .trainer import train_detector, detect_anomalies

__all__ = ["train_detector", "detect_anomalies"]
