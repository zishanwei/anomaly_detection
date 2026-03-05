"""Configuration for anomaly detection system."""

# Database connection (adjust for your DB - PostgreSQL, MySQL, SQLite, etc.)
DATABASE_URL = "sqlite:///traffic_data.db"

# Feed data: SQL Server connection (change these for your environment)
SOURCE_SERVER = 'newscoutswisstrafficai.database.windows.net'
SOURCE_DATABASE = 'automated.tests.mobility.online'
SOURCE_USER = 'newscoutadmin'
SOURCE_PASSWORD = 'NewScout123'
SOURCE_DRIVER = 'ODBC Driver 18 for SQL Server'

TARGET_SERVER = 'newscoutswisstrafficai.database.windows.net'
TARGET_DATABASE = 'anomaly_detection_data'
TARGET_USER = 'newscoutadmin'
TARGET_PASSWORD = 'NewScout123'
TARGET_DRIVER = 'ODBC Driver 18 for SQL Server'

# Traffic data attributes
# VEHICLE_CATEGORIES = ["car", "truck", "bicycle", "pedestrian"]
TIME_RANGE_COLUMN = "time_range"
COUNTING_COLUMN = "counting"

# Model: "seasonal" (ADTK - traffic-specific), "isolation_forest" (generic)
MODEL_TYPE = "seasonal"

# For seasonal model: period length (e.g. 24=hourly/daily, 48=30min/daily, 168=hourly/weekly)
SEASONAL_FREQ = 24
