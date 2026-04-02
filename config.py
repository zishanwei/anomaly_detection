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
DIRECTION_COLUMN = "direction"
ANOMALY_GROUP_BY = "category"
# Model: "seasonal" (ADTK), "isolation_forest", "lstm" (local .pt), "chronos" (HF Chronos-2)
MODEL_TYPE = "chronos"
TORCH_DEVICE = "auto"

# Seasonal (ADTK)
SERIES_RESAMPLE_FREQ = "15min"
SEASONAL_FREQ = 96  # one day in 15-min steps (24 * 4)

# Pre-trained LSTM (local checkpoint; scripts/create_dummy_lstm_checkpoint.py)
LSTM_PRETRAINED_PATH = "models/lstm_traffic.pt"
LSTM_SEQ_LEN = 96
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_ANOMALY_QUANTILE = 0.99

# Hugging Face Chronos (pretrained forecasting; anomaly = forecast error). pip install chronos-forecasting
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_CONTEXT_LENGTH = 256
CHRONOS_ANOMALY_QUANTILE = 0.99
