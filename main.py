"""
Anomaly detection for traffic counting data.

Blocks: data process -> model training -> data analyze
"""

from src.data_process import load_traffic_data, prepare_features
from src.data_process.visualize_data import (
    plot_original_series,
    plot_original_and_anomalies_combined,
)
from src.model import train_detector, detect_anomalies
from src.analyze import analyze_anomalies
from config import COUNTING_COLUMN, MODEL_TYPE


def run(show_visualize: bool = True):
    """Run full pipeline: load, visualize (original), train, detect, visualize (anomalies), analyze."""
    df = load_traffic_data()
    df = prepare_features(df)


    feature_cols = [COUNTING_COLUMN]
    clf = train_detector(df, feature_cols, model_type=MODEL_TYPE)
    scores, labels = detect_anomalies(clf, df, feature_cols, model_type=MODEL_TYPE)

    if show_visualize:
        print("Visualization: data with anomaly detection (after training)")
        plot_original_and_anomalies_combined(df, labels, show=True)

    report = analyze_anomalies(df, scores, labels, COUNTING_COLUMN)

    print(f"Detected {len(report)} anomalies")
    if not report.empty:
        print(report[[COUNTING_COLUMN, "anomaly_score", "reason"]].head(10))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly detection pipeline")
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip matplotlib figures (original before train, anomalies after)",
    )
    args = parser.parse_args()
    run(show_visualize=not args.no_visualize)
