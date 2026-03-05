"""
Anomaly detection for traffic counting data.

Blocks: data process -> model training -> data analyze
"""

from src.data_process import load_traffic_data, prepare_features
from src.data_process.sample_data import create_sample_table
from src.model import train_detector, detect_anomalies
from src.analyze import analyze_anomalies
from config import COUNTING_COLUMN, MODEL_TYPE


def run():
    """Run full pipeline: load, train, detect, analyze."""
    create_sample_table()

    df = load_traffic_data()
    df = prepare_features(df)

    feature_cols = [COUNTING_COLUMN]
    clf = train_detector(df, feature_cols, model_type=MODEL_TYPE)
    scores, labels = detect_anomalies(clf, df, feature_cols, model_type=MODEL_TYPE)

    report = analyze_anomalies(df, scores, labels, COUNTING_COLUMN)
    print(f"Detected {len(report)} anomalies")
    if not report.empty:
        print(report[[COUNTING_COLUMN, "anomaly_score", "reason"]].head(10))
    return report


if __name__ == "__main__":
    run()
