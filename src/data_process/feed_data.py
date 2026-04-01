"""
Feed data from source database (od_matrix_view, te_data) to target database.

Target format: direction, category, start_time, end_time, interval, device_id, count
Device identity is device_id:port (e.g. 192.168.1.1:8080).
Table name: config TABLE_NAME, optional per-source table_name, or traffic_counts_from_<source database>.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config import (
    TARGET_SERVER, TARGET_DATABASE, TARGET_USER, TARGET_PASSWORD, TARGET_DRIVER,
    SOURCE_DATABASES, SQL_DRIVER, TABLE_NAME,
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


def get_table_name(database: str, table_name: str | None = None) -> str:
    """
    Resolve target table name: explicit table_name, then config TABLE_NAME,
    else traffic_counts_from_<database>.
    """
    if table_name and str(table_name).strip():
        return str(table_name).strip()
    if TABLE_NAME and str(TABLE_NAME).strip():
        return str(TABLE_NAME).strip()
    if not database:
        raise ValueError("database name is required for table name")
    return "traffic_counts_from_" + database.replace(" ", "_").replace("-", "_")


def get_engine(url: str) -> Engine:
    return create_engine(url)


def get_source_config(name: str | None = None) -> dict:
    """Get source config from SOURCE_DATABASES by name. Default: first entry (Test DB)."""
    if name:
        for db in SOURCE_DATABASES:
            if db.get("name") == name:
                db_name = db.get("database")
                if not db_name:
                    raise ValueError(f"SOURCE_DATABASES entry '{name}' has no 'database'")
                return {
                    "server": db["server"],
                    "database": db_name,
                    "user": db["user"],
                    "password": db.get("password", ""),
                    "driver": db.get("driver", SQL_DRIVER),
                    "table_name": db.get("table_name"),
                }
        raise ValueError(f"Unknown database name: {name}")
    if not SOURCE_DATABASES:
        raise ValueError("SOURCE_DATABASES is empty")
    db = SOURCE_DATABASES[0]
    db_name = db.get("database")
    if not db_name:
        raise ValueError(f"SOURCE_DATABASES entry '{db.get('name', '?')}' has no 'database'")
    return {
        "server": db["server"],
        "database": db_name,
        "user": db["user"],
        "password": db.get("password", ""),
        "driver": db.get("driver", SQL_DRIVER),
        "table_name": db.get("table_name"),
    }


def _floor_15min(ts: pd.Series) -> pd.Series:
    """Floor timestamps to 15-min boundaries (00:00, 00:15, 00:30, 00:45)."""
    return pd.to_datetime(ts).dt.floor(f"{INTERVAL_MINUTES}min")


def _device_identity(device_id: str, port: str | None = None) -> str:
    """Build device identity as device_id:port. If port empty, return device_id as is."""
    did = str(device_id).strip()
    p = str(port).strip() if port is not None and str(port) != "nan" else ""
    return f"{did}:{p}" if p else did


def transform_od_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform od_matrix_view to target format. Fixed 15-min intervals, all categories."""
    if df.empty:
        return pd.DataFrame(columns=["direction", "category", "start_time", "end_time", "interval", "device_id", "count"])

    df = df.copy()
    df["direction"] = df["Origin"].astype(str) + " - " + df["Destination"].astype(str)
    df["category"] = df["Category"]
    df["start_time"] = _floor_15min(df["StartTime"])
    df["end_time"] = df["start_time"] + pd.Timedelta(minutes=INTERVAL_MINUTES)
    has_port = "port" in df.columns
    df["device_id"] = df.apply(
        lambda r: _device_identity(r["DeviceID"], r.get("port") if has_port else None),
        axis=1
    )
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
    has_port = "port" in df.columns
    df["device_id"] = df.apply(
        lambda r: _device_identity(r["device_id"], r.get("port") if has_port else None),
        axis=1
    )

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
    """Fetch od_matrix_view and gate_data from source database."""
    od_df = pd.read_sql("SELECT * FROM od_matrix_view", source_engine)
    te_df = pd.read_sql("SELECT * FROM gate_data", source_engine)

    od_map = {"origin": "Origin", "destination": "Destination", "category": "Category",
              "starttime": "StartTime", "endtime": "EndTime", "deviceid": "DeviceID", "count": "Count",
              "port": "port"}
    te_map = {"category": "category", "trajectory_start_time": "trajectory_start_time",
              "trajectory_end_time": "trajectory_end_time", "name": "name", "device_id": "device_id",
              "port": "port"}
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
            device_id NVARCHAR(150),
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
    source_name: str | None = None,
    table_name: str | None = None,
    target_server: str | None = None,
    target_database: str | None = None,
    target_user: str | None = None,
    target_password: str | None = None,
    target_driver: str | None = None,
    replace: bool = False,
):
    """
    Feed data from source DB to target DB (both SQL Server).

    Use source_name to pick from config.SOURCE_DATABASES, or override with source_* args.
    table_name: override target table (else config TABLE_NAME, else per-source table_name, else derived).
    replace: if True, replace target table; else append.
    """
    src = get_source_config(source_name)
    source_server = source_server or src["server"]
    source_database = source_database or src["database"]
    source_user = source_user or src["user"]
    source_password = source_password or src["password"]
    source_driver = source_driver or src["driver"]

    if not source_database:
        raise ValueError("source_database is required; check SOURCE_DATABASES config or pass --source-database")

    resolved_table = table_name or src.get("table_name")

    source_engine = get_engine(build_sqlserver_url(
        source_server,
        source_database,
        source_user,
        source_password,
        source_driver,
    ))
    target_engine = get_engine(build_sqlserver_url(
        target_server or TARGET_SERVER,
        target_database or TARGET_DATABASE,
        target_user or TARGET_USER,
        target_password or TARGET_PASSWORD,
        target_driver or TARGET_DRIVER,
    ))
    target_table = get_table_name(source_database, table_name=resolved_table)

    od_df, te_df = fetch_source_views(source_engine)
    od_transformed = transform_od_matrix(od_df)
    te_transformed = transform_te_data(te_df)

    combined = pd.concat([od_transformed, te_transformed], ignore_index=True)

    with target_engine.connect() as conn:
        _create_table_sqlserver(conn, target_table, replace)

    combined.to_sql(target_table, target_engine, if_exists="append", index=False)
    return len(combined)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-name", help="Source from config.SOURCE_DATABASES by name (e.g. 'Test DB', 'St Etienne', 'Sterela', 'Nevers'). Default: first entry.")
    parser.add_argument("--source-server", help="Source SQL Server host (overrides --source-name)")
    parser.add_argument("--source-database", help="Source database name")
    parser.add_argument("--source-user", help="Source username")
    parser.add_argument("--source-password", help="Source password")
    parser.add_argument("--target-server", help="Target SQL Server host")
    parser.add_argument("--target-database", help="Target database name")
    parser.add_argument("--target-user", help="Target username")
    parser.add_argument("--target-password", help="Target password")
    parser.add_argument("--replace", action="store_true", help="Replace target table instead of append")
    parser.add_argument("--table-name", help="Target table name (overrides config TABLE_NAME and per-source table_name)")
    parser.add_argument("--all", action="store_true", dest="feed_all", help="Feed from all config.SOURCE_DATABASES")
    args = parser.parse_args()

    if args.feed_all:
        total = 0
        for db in SOURCE_DATABASES:
            name = db.get("name")
            try:
                n = feed(source_name=name, table_name=args.table_name, replace=args.replace)
                total += n
                print(f"[{name}] Fed {n} rows")
            except Exception as e:
                print(f"[{name}] Error: {e}")
        print(f"Total: {total} rows")
    else:
        n = feed(
            source_name=args.source_name,
            source_server=args.source_server,
            source_database=args.source_database,
            source_user=args.source_user,
            source_password=args.source_password,
            table_name=args.table_name,
            target_server=args.target_server,
            target_database=args.target_database,
            target_user=args.target_user,
            target_password=args.target_password,
            replace=args.replace,
        )
        print(f"Fed {n} rows to target database")
