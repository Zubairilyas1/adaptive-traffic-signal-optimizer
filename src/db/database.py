"""Zero-cost local SQLite telemetry and audit store.

Provides high-performance persistent logging for:
- Second-by-second traffic queues & flowing throughput
- Dynamic closed-loop actuation decisions & switch events
- Red-light violation enforcement logs with image evidence links
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from src import config


def _resolve_path(db_path: Path | str | None) -> Path:
    if db_path is None:
        return config.TELEMETRY_DB
    return Path(db_path)


def init_db(db_path: Path | str | None = None) -> Path:
    """Initialize SQLite database with required tables, indexes, and WAL mode."""
    target = _resolve_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(target)) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS intersections (
                id TEXT PRIMARY KEY,
                name TEXT,
                topology_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intersection_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                approach TEXT NOT NULL,
                halted_pcu REAL NOT NULL,
                flowing_pcu REAL NOT NULL,
                queue_length_m REAL NOT NULL,
                active_phase TEXT,
                active_aspect TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS actuation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intersection_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                from_phase TEXT NOT NULL,
                to_phase TEXT NOT NULL,
                aspect TEXT NOT NULL,
                reason TEXT NOT NULL,
                message TEXT,
                waiting_queue_pcu REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intersection_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_class TEXT NOT NULL,
                speed_km_h REAL NOT NULL,
                timestamp REAL NOT NULL,
                approach TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Performance indexes for historical queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_lookup
            ON telemetry_logs (intersection_id, timestamp);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_actuation_lookup
            ON actuation_events (intersection_id, timestamp);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_lookup
            ON violations (intersection_id, timestamp);
        """)

    return target


@contextlib.contextmanager
def get_db_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager providing a SQLite connection with Row factory."""
    target = _resolve_path(db_path)
    if not target.exists():
        init_db(target)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def log_telemetry(
    intersection_id: str,
    timestamp: float,
    approach: str,
    halted_pcu: float,
    flowing_pcu: float,
    queue_length_m: float,
    active_phase: str = "",
    active_aspect: str = "",
    db_path: Path | str | None = None,
) -> None:
    """Record an instantaneous queue and flow measurement."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO telemetry_logs (
                intersection_id, timestamp, approach, halted_pcu,
                flowing_pcu, queue_length_m, active_phase, active_aspect
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                intersection_id,
                round(float(timestamp), 2),
                str(approach),
                round(float(halted_pcu), 2),
                round(float(flowing_pcu), 2),
                round(float(queue_length_m), 2),
                str(active_phase),
                str(active_aspect),
            ),
        )
        conn.commit()


def log_telemetry_batch(
    records: list[dict[str, Any]],
    db_path: Path | str | None = None,
) -> None:
    """Insert a list of telemetry rows efficiently in a single transaction."""
    if not records:
        return
    rows = [
        (
            r["intersection_id"],
            round(float(r["timestamp"]), 2),
            str(r["approach"]),
            round(float(r["halted_pcu"]), 2),
            round(float(r["flowing_pcu"]), 2),
            round(float(r["queue_length_m"]), 2),
            str(r.get("active_phase", "")),
            str(r.get("active_aspect", "")),
        )
        for r in records
    ]
    with get_db_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO telemetry_logs (
                intersection_id, timestamp, approach, halted_pcu,
                flowing_pcu, queue_length_m, active_phase, active_aspect
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()


def log_actuation_event(
    intersection_id: str,
    timestamp: float,
    from_phase: str,
    to_phase: str,
    aspect: str,
    reason: str,
    message: str,
    waiting_queue_pcu: float,
    db_path: Path | str | None = None,
) -> None:
    """Record an autonomous controller state switch event."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO actuation_events (
                intersection_id, timestamp, from_phase, to_phase,
                aspect, reason, message, waiting_queue_pcu
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                intersection_id,
                round(float(timestamp), 2),
                str(from_phase),
                str(to_phase),
                str(aspect),
                str(reason),
                str(message),
                round(float(waiting_queue_pcu), 2),
            ),
        )
        conn.commit()


def log_violation(
    intersection_id: str,
    track_id: int,
    vehicle_class: str,
    speed_km_h: float,
    timestamp: float,
    approach: str,
    image_path: str,
    db_path: Path | str | None = None,
) -> None:
    """Record a red-light crossing infraction with photo evidence path."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO violations (
                intersection_id, track_id, vehicle_class, speed_km_h,
                timestamp, approach, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                intersection_id,
                int(track_id),
                str(vehicle_class),
                round(float(speed_km_h), 1),
                round(float(timestamp), 2),
                str(approach),
                str(image_path),
            ),
        )
        conn.commit()


def query_telemetry(
    intersection_id: str | None = None,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent telemetry records ordered by timestamp descending."""
    with get_db_connection(db_path) as conn:
        if intersection_id:
            cursor = conn.execute(
                """
                SELECT * FROM telemetry_logs
                WHERE intersection_id = ?
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (intersection_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM telemetry_logs
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]


def query_actuation_events(
    intersection_id: str | None = None,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent signal controller decisions."""
    with get_db_connection(db_path) as conn:
        if intersection_id:
            cursor = conn.execute(
                """
                SELECT * FROM actuation_events
                WHERE intersection_id = ?
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (intersection_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM actuation_events
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]


def query_violations(
    intersection_id: str | None = None,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent red-light infractions."""
    with get_db_connection(db_path) as conn:
        if intersection_id:
            cursor = conn.execute(
                """
                SELECT * FROM violations
                WHERE intersection_id = ?
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (intersection_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM violations
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]


def get_summary_stats(
    intersection_id: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute aggregate operations statistics."""
    with get_db_connection(db_path) as conn:
        where_clause = "WHERE intersection_id = ?" if intersection_id else ""
        params = (intersection_id,) if intersection_id else ()

        telemetry_count = conn.execute(
            f"SELECT COUNT(*) FROM telemetry_logs {where_clause};", params
        ).fetchone()[0]

        max_queue = conn.execute(
            f"SELECT COALESCE(MAX(queue_length_m), 0.0) FROM telemetry_logs {where_clause};", params
        ).fetchone()[0]

        avg_halted_pcu = conn.execute(
            f"SELECT COALESCE(AVG(halted_pcu), 0.0) FROM telemetry_logs {where_clause};", params
        ).fetchone()[0]

        actuation_count = conn.execute(
            f"SELECT COUNT(*) FROM actuation_events {where_clause};", params
        ).fetchone()[0]

        gap_outs = conn.execute(
            f"SELECT COUNT(*) FROM actuation_events WHERE reason = 'GAP_OUT' {'AND intersection_id = ?' if intersection_id else ''};",
            params,
        ).fetchone()[0]

        violation_count = conn.execute(
            f"SELECT COUNT(*) FROM violations {where_clause};", params
        ).fetchone()[0]

        return {
            "total_telemetry_samples": telemetry_count,
            "max_queue_length_m": round(float(max_queue), 1),
            "avg_halted_pcu": round(float(avg_halted_pcu), 2),
            "total_actuations": actuation_count,
            "gap_out_triggers": gap_outs,
            "wasted_green_saved_s": round(gap_outs * 8.5, 1),
            "total_violations": violation_count,
        }


def purge_all_data(db_path: Path | str | None = None) -> None:
    """Purge all recorded telemetry, actuations, and violations (for tests and resets)."""
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM telemetry_logs;")
        conn.execute("DELETE FROM actuation_events;")
        conn.execute("DELETE FROM violations;")
        conn.execute("DELETE FROM intersections;")
        conn.commit()
