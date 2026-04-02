"""
Anomaly detection for traffic counting data.

Blocks: data process -> model training -> data analyze
"""

from src.data_process import load_traffic_data, prepare_features
from src.data_process.visualize_data import (
    anomaly_group_modes,
    plot_original_and_anomalies_combined,
)
from src.model import train_detector, detect_anomalies
from src.analyze import analyze_anomalies
from config import CATEGORY_COLUMN, COUNTING_COLUMN, MODEL_TYPE


def run(show_visualize: bool = True):
    """Run full pipeline: load, train, detect (by category and/or direction), visualize, analyze."""
    df = load_traffic_data()
    df = prepare_features(df)

    feature_cols = [COUNTING_COLUMN]
    modes = anomaly_group_modes()
    clf = None
    report = None

    for col, glabel in modes:
        if MODEL_TYPE == "seasonal":
            clf = train_detector(df, feature_cols, model_type=MODEL_TYPE, group_column=col)
        else:
            if clf is None:
                clf = train_detector(df, feature_cols, model_type=MODEL_TYPE)
        scores, labels = detect_anomalies(
            clf, df, feature_cols, model_type=MODEL_TYPE, group_column=col,
        )

        if show_visualize:
            print(f"Visualization: anomaly detection grouped by {glabel}")
            plot_original_and_anomalies_combined(
                df, labels, show=True, facet_col=col, group_by_label=glabel,
            )

        rep = analyze_anomalies(df, scores, labels, COUNTING_COLUMN)
        print(f"--- By {glabel}: {len(rep)} anomalous rows in report ---")
        if not rep.empty:
            print(rep[[COUNTING_COLUMN, "anomaly_score", "reason"]].head(10))
        if glabel == "category":
            report = rep

    if report is None:
        report = rep

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
