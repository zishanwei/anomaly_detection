"""
Visualize traffic count data from the target database.

Data format: direction, category, start_time, end_time, interval, device_id, count
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import TARGET_SERVER, TARGET_DATABASE, TARGET_USER, TARGET_PASSWORD, TARGET_DRIVER
from src.data_process.feed_data import (
    build_sqlserver_url,
    get_table_name,
    get_engine,
)


def load_traffic_data(
    table_name: str | None = None,
    target_server: str | None = None,
    target_database: str | None = None,
    target_user: str | None = None,
    target_password: str | None = None,
    target_driver: str | None = None,
) -> pd.DataFrame:
    """Load traffic count data from target database."""
    engine = get_engine(build_sqlserver_url(
        target_server or TARGET_SERVER,
        target_database or TARGET_DATABASE,
        target_user or TARGET_USER,
        target_password or TARGET_PASSWORD,
        target_driver or TARGET_DRIVER,
    ))
    tbl = table_name or get_table_name(target_database or TARGET_DATABASE)
    df = pd.read_sql(f"SELECT * FROM [{tbl}]", engine)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def plot_counts_over_time(df: pd.DataFrame, top_n: int = 5, output_path: str | None = None, show: bool = True):
    """Plot aggregated count over time, with top directions as separate lines."""
    if df.empty:
        print("No data to plot")
        return

    agg = df.groupby(["start_time"])["count"].sum().reset_index()
    agg = agg.sort_values("start_time")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(agg["start_time"], agg["count"], color="steelblue", alpha=0.8, label="Total")

    top_dirs = df.groupby("direction")["count"].sum().nlargest(top_n).index
    for d in top_dirs:
        sub = df[df["direction"] == d].groupby("start_time")["count"].sum().reset_index()
        sub = sub.sort_values("start_time")
        ax.plot(sub["start_time"], sub["count"], alpha=0.7, label=d[:30] + ("..." if len(d) > 30 else ""))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_title("Traffic counts over time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_category_distribution(df: pd.DataFrame, output_path: str | None = None, show: bool = True):
    """Bar chart of counts by category."""
    if df.empty:
        print("No data to plot")
        return

    agg = df.groupby("category")["count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    agg.plot(kind="bar", ax=ax, color="steelblue", width=0.7)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_title("Traffic counts by category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_hourly_heatmap(df: pd.DataFrame, output_path: str | None = None, show: bool = True):
    """Heatmap of counts by hour and weekday."""
    if df.empty:
        print("No data to plot")
        return

    df = df.copy()
    df["hour"] = df["start_time"].dt.hour
    df["weekday"] = df["start_time"].dt.weekday
    pivot = df.pivot_table(index="weekday", columns="hour", values="count", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    ax.set_title("Traffic counts by hour and weekday")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_top_directions(df: pd.DataFrame, top_n: int = 10, output_path: str | None = None, show: bool = True):
    """Plot top directions by total count."""
    if df.empty:
        print("No data to plot")
        return

    agg = df.groupby("direction")["count"].sum().nlargest(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    agg.plot(kind="barh", ax=ax, color="steelblue", width=0.7)
    ax.set_xlabel("Count")
    ax.set_ylabel("Direction")
    ax.set_title(f"Top {top_n} directions by traffic count")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def visualize(
    output_dir: str | None = None,
    target_server: str | None = None,
    target_database: str | None = None,
    target_user: str | None = None,
    target_password: str | None = None,
    target_driver: str | None = None,
    table_name: str | None = None,
    show: bool = True,
):
    """
    Load data and generate all visualizations.

    output_dir: if set, save plots to files instead of showing.
    show: if True and no output_dir, display plots interactively.
    """
    df = load_traffic_data(
        table_name=table_name,
        target_server=target_server,
        target_database=target_database,
        target_user=target_user,
        target_password=target_password,
        target_driver=target_driver,
    )
    if df.empty:
        print("No data loaded")
        return

    print(f"Loaded {len(df)} rows")
    out = Path(output_dir) if output_dir else None

    if out:
        out.mkdir(parents=True, exist_ok=True)

    plot_counts_over_time(df, output_path=str(out / "counts_over_time.png") if out else None, show=show)
    plot_category_distribution(df, output_path=str(out / "category_distribution.png") if out else None, show=show)
    plot_hourly_heatmap(df, output_path=str(out / "hourly_heatmap.png") if out else None, show=show)
    plot_top_directions(df, output_path=str(out / "top_directions.png") if out else None, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize traffic count data")
    parser.add_argument("--output-dir", "-o", help="Directory to save plots")

    parser.add_argument("--target-server", help="Target SQL Server host")
    parser.add_argument("--target-database", help="Target database name")
    parser.add_argument("--target-user", help="Target username")
    parser.add_argument("--target-password", help="Target password")
    parser.add_argument("--table", help="Table name (default: traffic_counts_from_<database>)")

    parser.add_argument("--no-show", action="store_true", help="Do not display plots (only save when -o is set)")
    args = parser.parse_args()

    visualize(
        output_dir=args.output_dir,
        target_server=args.target_server,
        target_database=args.target_database,
        target_user=args.target_user,
        target_password=args.target_password,
        table_name=args.table,
        show=not args.no_show,
    )
