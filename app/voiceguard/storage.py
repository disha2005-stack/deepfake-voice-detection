import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def connect(database):
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database):
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    with connect(database) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                real_probability REAL NOT NULL,
                fake_probability REAL NOT NULL,
                processing_seconds REAL NOT NULL,
                segment_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def add_analysis(database, filename, result):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(database) as connection:
        cursor = connection.execute(
            """INSERT INTO analyses (
                filename, prediction, confidence, real_probability,
                fake_probability, processing_seconds, segment_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename,
                result["prediction"],
                result["confidence"],
                result["real_probability"],
                result["fake_probability"],
                result["processing_seconds"],
                result["segment_count"],
                created_at,
            ),
        )
        connection.execute(
            "DELETE FROM analyses WHERE id NOT IN (SELECT id FROM analyses ORDER BY id DESC LIMIT 20)"
        )
        return {"id": cursor.lastrowid, "created_at": created_at}


def get_history(database):
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return [dict(row) for row in rows]


def clear_history(database):
    with connect(database) as connection:
        connection.execute("DELETE FROM analyses")