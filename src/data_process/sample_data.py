"""Create sample traffic data table for testing when DB is empty."""

import pandas as pd
from sqlalchemy import create_engine, text
from config import DATABASE_URL


def create_sample_table(engine=None):
    """Create traffic_counts table with sample data if it does not exist."""
    engine = engine or create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS traffic_counts (
                id INTEGER PRIMARY KEY,
                vehicle_category TEXT,
                time_range TIMESTAMP,
                counting INTEGER
            )
        """))
        conn.commit()

        result = conn.execute(text("SELECT COUNT(*) FROM traffic_counts"))
        if result.scalar() == 0:
            dates = pd.date_range("2024-01-01", "2024-01-31", freq="h")
            categories = ["car", "truck", "bicycle", "pedestrian"]
            rows = []
            for i, ts in enumerate(dates):
                for cat in categories:
                    base = 100 + hash(cat) % 50
                    count = base + (i % 24) * 2
                    rows.append({"vehicle_category": cat, "time_range": ts, "counting": count})
            df = pd.DataFrame(rows)
            df.to_sql("traffic_counts", engine, if_exists="append", index=False)
            conn.commit()
