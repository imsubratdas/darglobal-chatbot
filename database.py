
from pathlib import Path

import os
import shutil

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv


# Get the current project directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# Create data directory if it doesn't exist
DATA_DIR_SETTING = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR = (DATA_DIR_SETTING if DATA_DIR_SETTING.is_absolute() else BASE_DIR / DATA_DIR_SETTING).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Preserve the packaged SQLite records when a fresh persistent volume is first mounted.
DATABASE_URL_SETTING = os.getenv("DATABASE_URL")
DB_FILE = DATA_DIR / "propintel.db"
PACKAGED_DB_FILE = BASE_DIR / "data" / "propintel.db"
LEGACY_DB_FILE = Path(os.getenv("LEGACY_DATABASE_FILE", "")).expanduser() if os.getenv("LEGACY_DATABASE_FILE") else None
if not DATABASE_URL_SETTING and not DB_FILE.exists():
    if LEGACY_DB_FILE and LEGACY_DB_FILE.is_file():
        shutil.copy2(LEGACY_DB_FILE, DB_FILE)
    elif (DATA_DIR != (BASE_DIR / "data").resolve() and PACKAGED_DB_FILE.exists()):
        shutil.copy2(PACKAGED_DB_FILE, DB_FILE)

# SQLite database location
DATABASE_URL = DATABASE_URL_SETTING or f"sqlite:///{DB_FILE}"


# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if DATABASE_URL.startswith("sqlite") else {}
)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Create database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Base class for database models
Base = declarative_base()


# Database dependency for FastAPI
def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
