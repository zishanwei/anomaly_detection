"""Configuration for anomaly detection system."""

# Local database (loader, main pipeline)
DATABASE_URL = "sqlite:///traffic_data.db"

# SQL Server driver (shared)
SQL_DRIVER = "ODBC Driver 18 for SQL Server"

# Target database (where feed_data writes)
TARGET_SERVER = "newscoutswisstrafficai.database.windows.net"
TARGET_DATABASE = "anomaly_detection_data"
TARGET_USER = "newscoutadmin"
TARGET_PASSWORD = "NewScout123"
TARGET_DRIVER = SQL_DRIVER

# Target table for feed_data (None = traffic_counts_from_<source database name>)
TABLE_NAME = "traffic_counts_from_sterela"

# Source databases (where feed_data reads from)
# Default: used when --source-name not specified

# Named sources: use with --source-name "St Etienne", etc.
# Optional per entry: "table_name": "my_table" (overrides global TABLE_NAME for that source)
SOURCE_DATABASES = [
    {
        "name": "Test DB",
        "server": "mobility-online-dbsvr-tests-and-uat.database.windows.net",
        "database": "automated.tests.mobility.online",
        "user": "newscoutadmin",
        "password": "NewScout123",
        "driver": SQL_DRIVER,
    },
    {
        "name": "St Etienne",
        "server": "newscoutswisstrafficai.database.windows.net",
        "database": "saint-etienne",
        "user": "newscoutadmin",
        "password": "NewScout123",
    },
    {
        "name": "Sterela",
        "server": "mobility-online-dbsvr-fr-sterela.database.windows.net",
        "database": "sterela",
        "user": "mobility_online_db_admin",
        "password": "NewScout123",
    },
    {
        "name": "Nevers",
        "server": "mobility-online-dbsvr-fr-nevers.database.windows.net",
        "database": "nevers",
        "user": "mobility_online_db_admin",
        "password": "NewScout123",
    },
]

# Traffic data columns (feed_data / SQL Server target table)
TIME_RANGE_COLUMN = "start_time"
COUNTING_COLUMN = "count"
CATEGORY_COLUMN = "category"

# Model
MODEL_TYPE = "seasonal"
# Resample series to this pandas offset before ADTK (must match feed_data interval, e.g. 15min)
SERIES_RESAMPLE_FREQ = "15min"
# SeasonalAD period in number of steps: 96 = one day at 15-min resolution (24 * 4)
SEASONAL_FREQ = 96
