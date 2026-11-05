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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS digital_twin_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                sim_time REAL NOT NULL,
                q_north_cam REAL NOT NULL,
                q_south_cam REAL NOT NULL,
                q_east_cam REAL NOT NULL,
                q_west_cam REAL NOT NULL,
                q_north_sumo REAL NOT NULL,
                q_south_sumo REAL NOT NULL,
                q_east_sumo REAL NOT NULL,
                q_west_sumo REAL NOT NULL,
                divergence_mae REAL NOT NULL,
                active_phase INTEGER NOT NULL,
                active_aspect TEXT NOT NULL,
                controller_type TEXT NOT NULL,
                action_taken INTEGER NOT NULL,
                step_reward REAL NOT NULL,
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_digital_twin_lookup
            ON digital_twin_logs (session_id, timestamp);
        """)

    return target


_INITIALIZED_DBS: set[Path] = set()


@contextlib.contextmanager
def get_db_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager providing a SQLite connection with Row factory."""
    target = _resolve_path(db_path)
    if target not in _INITIALIZED_DBS or not target.exists():
        init_db(target)
        _INITIALIZED_DBS.add(target)
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


def log_digital_twin_step(
    session_id: str,
    timestamp: float,
    sim_time: float,
    queues_cam: dict[str, float] | Any,
    queues_sumo: dict[str, float] | Any,
    divergence_mae: float,
    active_phase: int,
    active_aspect: str,
    controller_type: str,
    action_taken: int,
    step_reward: float,
    db_path: Path | str | None = None,
) -> None:
    """Record an instantaneous digital twin perception-vs-simulation synchronization step."""
    def _extract_q(q_input: Any) -> tuple[float, float, float, float]:
        if isinstance(q_input, dict):
            return (
                float(q_input.get("N", q_input.get("North", 0.0))),
                float(q_input.get("S", q_input.get("South", 0.0))),
                float(q_input.get("E", q_input.get("East", 0.0))),
                float(q_input.get("W", q_input.get("West", 0.0))),
            )
        vals = [float(x) for x in list(q_input)[:4]]
        while len(vals) < 4:
            vals.append(0.0)
        return (vals[0], vals[1], vals[2], vals[3])

    qn_cam, qs_cam, qe_cam, qw_cam = _extract_q(queues_cam)
    qn_sumo, qs_sumo, qe_sumo, qw_sumo = _extract_q(queues_sumo)

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO digital_twin_logs (
                session_id, timestamp, sim_time,
                q_north_cam, q_south_cam, q_east_cam, q_west_cam,
                q_north_sumo, q_south_sumo, q_east_sumo, q_west_sumo,
                divergence_mae, active_phase, active_aspect,
                controller_type, action_taken, step_reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(session_id),
                round(float(timestamp), 2),
                round(float(sim_time), 2),
                round(qn_cam, 2),
                round(qs_cam, 2),
                round(qe_cam, 2),
                round(qw_cam, 2),
                round(qn_sumo, 2),
                round(qs_sumo, 2),
                round(qe_sumo, 2),
                round(qw_sumo, 2),
                round(float(divergence_mae), 2),
                int(active_phase),
                str(active_aspect),
                str(controller_type),
                int(action_taken),
                round(float(step_reward), 2),
            ),
        )
        conn.commit()


def query_digital_twin_logs(
    session_id: str | None = None,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent digital twin synchronization records."""
    with get_db_connection(db_path) as conn:
        if session_id:
            cursor = conn.execute(
                """
                SELECT * FROM digital_twin_logs
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (session_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM digital_twin_logs
                ORDER BY timestamp DESC, id DESC LIMIT ?;
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]


def purge_all_data(db_path: Path | str | None = None) -> None:
    """Purge all recorded telemetry, actuations, and violations (for tests and resets)."""
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM telemetry_logs;")
        conn.execute("DELETE FROM actuation_events;")
        conn.execute("DELETE FROM violations;")
        conn.execute("DELETE FROM digital_twin_logs;")
        conn.execute("DELETE FROM intersections;")
        conn.commit()
