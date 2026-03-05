"""
Feed data from source database (od_matrix_view, te_data) to target database.

Target format: direction, category, start_time, end_time, interval, device_id, count
Table name is derived from the target database name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config import (
    SOURCE_SERVER, SOURCE_DATABASE, SOURCE_USER, SOURCE_PASSWORD, SOURCE_DRIVER,
    TARGET_SERVER, TARGET_DATABASE, TARGET_USER, TARGET_PASSWORD, TARGET_DRIVER,
)

# Fixed 15-min interval: start_time/end_time always on :00, :15, :30, :45 boundaries
INTERVAL_MINUTES = 15


def build_sqlserver_url(
    server: str, database: str, user: str, password: str, driver: str,
    encrypt: str = "yes", trust_server_certificate: str = "no",
) -> str:
    """Build SQL Server connection URL. For Azure SQL, use encrypt=yes, trust_server_certificate=no."""
    conn_str = (
        f"driver={{{driver}}};Server={server};Database={database};UID={user};PWD={password};"
        f"Encrypt={encrypt};TrustServerCertificate={trust_server_certificate}"
    )
    return f"mssql+pyodbc://?odbc_connect={urllib.parse.quote_plus(conn_str)}"


def get_table_name(database: str) -> str:
    """Use database name as table name."""
    table_name = "traffic_counts_from_" + database.replace(" ", "_").replace("-", "_")
    return table_name


def get_engine(url: str) -> Engine:
    return create_engine(url)


def _floor_15min(ts: pd.Series) -> pd.Series:
    """Floor timestamps to 15-min boundaries (00:00, 00:15, 00:30, 00:45)."""
    return pd.to_datetime(ts).dt.floor(f"{INTERVAL_MINUTES}min")


def transform_od_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform od_matrix_view to target format. Fixed 15-min intervals, all categories."""
    if df.empty:
        return pd.DataFrame(columns=["direction", "category", "start_time", "end_time", "interval", "device_id", "count"])

    df = df.copy()
    df["direction"] = df["Origin"].astype(str) + " - " + df["Destination"].astype(str)
    df["category"] = df["Category"]
    df["start_time"] = _floor_15min(df["StartTime"])
    df["end_time"] = df["start_time"] + pd.Timedelta(minutes=INTERVAL_MINUTES)
    df["device_id"] = df["DeviceID"].astype(str)
    df["count"] = df["Count"].astype(int)

    agg = df.groupby(["direction", "category", "start_time", "end_time", "device_id"])["count"].sum().reset_index()
    agg["interval"] = INTERVAL_MINUTES
    return agg[["direction", "category", "start_time", "end_time", "interval", "device_id", "count"]]


def transform_te_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform gate_data to target format.
    Fixed 15-min intervals (00:00-00:15, 00:15-00:30, etc), all categories.
    """
    if df.empty:
        return pd.DataFrame(columns=["direction", "category", "start_time", "end_time", "interval", "device_id", "count"])

    df = df.copy()
    df["trajectory_start_time"] = pd.to_datetime(df["trajectory_start_time"])
    df["start_time"] = _floor_15min(df["trajectory_start_time"])
    df["end_time"] = df["start_time"] + pd.Timedelta(minutes=INTERVAL_MINUTES)
    df["direction"] = df["name"].astype(str)
    df["device_id"] = df["device_id"].astype(str)

    agg = df.groupby(["direction", "category", "start_time", "end_time", "device_id"]).size().reset_index(name="count")
    agg["interval"] = INTERVAL_MINUTES
    return agg[["direction", "category", "start_time", "end_time", "interval", "device_id", "count"]]


def _rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns to expected names (handles case differences from SQL Server)."""
    rename = {}
    for c in df.columns:
        key = c.lower().replace(" ", "_")
        if key in mapping:
            rename[c] = mapping[key]
    return df.rename(columns=rename) if rename else df


def fetch_source_views(source_engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch od_matrix_view and te_data from source database."""
    od_df = pd.read_sql("SELECT * FROM od_matrix_view", source_engine)
    te_df = pd.read_sql("SELECT * FROM gate_data", source_engine)

    od_map = {"origin": "Origin", "destination": "Destination", "category": "Category",
              "starttime": "StartTime", "endtime": "EndTime", "deviceid": "DeviceID", "count": "Count"}
    te_map = {"category": "category", "trajectory_start_time": "trajectory_start_time",
              "trajectory_end_time": "trajectory_end_time", "name": "name", "device_id": "device_id"}
    od_df = _rename_columns(od_df, od_map)
    te_df = _rename_columns(te_df, te_map)
    return od_df, te_df


def _create_table_sqlserver(conn, table_name: str, replace: bool):
    """Create table in SQL Server."""
    if replace:
        conn.execute(text(f"IF OBJECT_ID('[{table_name}]', 'U') IS NOT NULL DROP TABLE [{table_name}]"))
        conn.commit()
    conn.execute(text(f"""
        IF OBJECT_ID('[{table_name}]', 'U') IS NULL
        CREATE TABLE [{table_name}] (
            id INT IDENTITY(1,1) PRIMARY KEY,
            direction NVARCHAR(255),
            category NVARCHAR(100),
            start_time DATETIME2,
            end_time DATETIME2,
            interval FLOAT,
            device_id NVARCHAR(100),
            count INT
        )
    """))
    conn.commit()


def feed(
    source_server: str | None = None,
    source_database: str | None = None,
    source_user: str | None = None,
    source_password: str | None = None,
    source_driver: str | None = None,
    target_server: str | None = None,
    target_database: str | None = None,
    target_user: str | None = None,
    target_password: str | None = None,
    target_driver: str | None = None,
    replace: bool = False,
):
    """
    Feed data from source DB to target DB (both SQL Server).

    Override config by passing server, database, user, password, driver as arguments.
    replace: if True, replace target table; else append.
    """
    source_engine = get_engine(build_sqlserver_url(
        source_server or SOURCE_SERVER,
        source_database or SOURCE_DATABASE,
        source_user or SOURCE_USER,
        source_password or SOURCE_PASSWORD,
        source_driver or SOURCE_DRIVER,
    ))
    target_engine = get_engine(build_sqlserver_url(
        target_server or TARGET_SERVER,
        target_database or TARGET_DATABASE,
        target_user or TARGET_USER,
        target_password or TARGET_PASSWORD,
        target_driver or TARGET_DRIVER,
    ))
    table_name = get_table_name(target_database or TARGET_DATABASE)

    od_df, te_df = fetch_source_views(source_engine)
    od_transformed = transform_od_matrix(od_df)
    te_transformed = transform_te_data(te_df)

    combined = pd.concat([od_transformed, te_transformed], ignore_index=True)

    with target_engine.connect() as conn:
        _create_table_sqlserver(conn, table_name, replace)

    combined.to_sql(table_name, target_engine, if_exists="append", index=False)
    return len(combined)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-server", help="Source SQL Server host")
    parser.add_argument("--source-database", help="Source database name")
    parser.add_argument("--source-user", help="Source username")
    parser.add_argument("--source-password", help="Source password")
    parser.add_argument("--target-server", help="Target SQL Server host")
    parser.add_argument("--target-database", help="Target database name")
    parser.add_argument("--target-user", help="Target username")
    parser.add_argument("--target-password", help="Target password")
    parser.add_argument("--replace", action="store_true", help="Replace target table instead of append")
    args = parser.parse_args()

    n = feed(
        source_server=args.source_server,
        source_database=args.source_database,
        source_user=args.source_user,
        source_password=args.source_password,
        target_server=args.target_server,
        target_database=args.target_database,
        target_user=args.target_user,
        target_password=args.target_password,
        replace=args.replace,
    )
    print(f"Fed {n} rows to target database")
