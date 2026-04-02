"""Load traffic data from database and prepare for model training."""

import pandas as pd
from sqlalchemy.engine import Engine

from config import (
    TARGET_SERVER,
    TARGET_DATABASE,
    TARGET_USER,
    TARGET_PASSWORD,
    TARGET_DRIVER,
    TABLE_NAME,
    TIME_RANGE_COLUMN,
    COUNTING_COLUMN,
    CATEGORY_COLUMN,
    DIRECTION_COLUMN,
)
from src.data_process.feed_data import build_sqlserver_url, get_table_name, get_engine as sql_engine_from_url


def get_engine() -> Engine:
    """Create SQL Server engine from target config (same as visualize_data)."""
    url = build_sqlserver_url(
        TARGET_SERVER,
        TARGET_DATABASE,
        TARGET_USER,
        TARGET_PASSWORD,
        TARGET_DRIVER,
    )
    return sql_engine_from_url(url)


def load_traffic_data(
    table_name: str | None = None,
    vehicle_category: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    target_server: str | None = None,
    target_database: str | None = None,
    target_user: str | None = None,
    target_password: str | None = None,
    target_driver: str | None = None,
) -> pd.DataFrame:
    """
    Load traffic counting data from target SQL Server.

    Expected columns match feed_data: category, start_time, count, ...
    Filters use category and TIME_RANGE_COLUMN (start_time).
    """
    engine = sql_engine_from_url(
        build_sqlserver_url(
            target_server or TARGET_SERVER,
            target_database or TARGET_DATABASE,
            target_user or TARGET_USER,
            target_password or TARGET_PASSWORD,
            target_driver or TARGET_DRIVER,
        )
    )

    tbl = table_name or get_table_name(target_database or TARGET_DATABASE)
    query = f"SELECT * FROM [{tbl}]"
    conditions = []
    params = {}

    if vehicle_category:
        conditions.append(f"[{CATEGORY_COLUMN}] = :vehicle_category")
        params["vehicle_category"] = vehicle_category
    if start_time:
        conditions.append(f"[{TIME_RANGE_COLUMN}] >= :start_time")
        params["start_time"] = start_time
    if end_time:
        conditions.append(f"[{TIME_RANGE_COLUMN}] <= :end_time")
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
    if DIRECTION_COLUMN not in out.columns:
        out[DIRECTION_COLUMN] = "unknown"
    else:
        out[DIRECTION_COLUMN] = out[DIRECTION_COLUMN].fillna("unknown").astype(str)
        out[DIRECTION_COLUMN] = out[DIRECTION_COLUMN].mask(out[DIRECTION_COLUMN] == "", "unknown")
    return out
