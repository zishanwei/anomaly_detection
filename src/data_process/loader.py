"""Load traffic data from database and prepare for model training."""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import DATABASE_URL, TIME_RANGE_COLUMN, COUNTING_COLUMN


def get_engine() -> Engine:
    """Create database engine from config."""
    return create_engine(DATABASE_URL)


def load_traffic_data(
    table_name: str = "traffic_counts",
    vehicle_category: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """
    Load traffic counting data from database.

    Expected columns: vehicle_category, time_range, counting
    """
    engine = get_engine()
    query = f"SELECT * FROM {table_name}"
    conditions = []
    params = {}

    if vehicle_category:
        conditions.append("vehicle_category = :vehicle_category")
        params["vehicle_category"] = vehicle_category
    if start_time:
        conditions.append("time_range >= :start_time")
        params["start_time"] = start_time
    if end_time:
        conditions.append("time_range <= :end_time")
        params["end_time"] = end_time

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql(query, engine, params=params if params else None)
    if TIME_RANGE_COLUMN in df.columns:
        df[TIME_RANGE_COLUMN] = pd.to_datetime(df[TIME_RANGE_COLUMN])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for anomaly detection.

    Extracts time-based features (weekday, hour, etc.) for reason identification.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    if TIME_RANGE_COLUMN in out.columns:
        out["weekday"] = out[TIME_RANGE_COLUMN].dt.weekday
        out["hour"] = out[TIME_RANGE_COLUMN].dt.hour
        out["date"] = out[TIME_RANGE_COLUMN].dt.date
    return out
