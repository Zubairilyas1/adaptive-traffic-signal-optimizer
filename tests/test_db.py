"""Unit tests for SQLite persistent telemetry, actuation, and violation logging."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.db.database import (
    get_summary_stats,
    init_db,
    log_actuation_event,
    log_telemetry,
    log_telemetry_batch,
    log_violation,
    purge_all_data,
    query_actuation_events,
    query_telemetry,
    query_violations,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_telemetry.db"
    init_db(db_file)
    return db_file


def test_init_db(temp_db: Path):
    assert temp_db.exists()
    assert temp_db.stat().st_size > 0


def test_log_and_query_telemetry(temp_db: Path):
    log_telemetry(
        intersection_id="inter_test",
        timestamp=10.5,
        approach="North",
        halted_pcu=4.2,
        flowing_pcu=1.5,
        queue_length_m=28.5,
        active_phase="P1",
        active_aspect="GREEN",
        db_path=temp_db,
    )

    records = query_telemetry("inter_test", limit=10, db_path=temp_db)
    assert len(records) == 1
    row = records[0]
    assert row["intersection_id"] == "inter_test"
    assert row["timestamp"] == 10.5
    assert row["approach"] == "North"
    assert row["halted_pcu"] == 4.2
    assert row["flowing_pcu"] == 1.5
    assert row["queue_length_m"] == 28.5
    assert row["active_phase"] == "P1"
    assert row["active_aspect"] == "GREEN"


def test_log_telemetry_batch(temp_db: Path):
    batch = [
        {
            "intersection_id": "inter_test",
            "timestamp": float(i),
            "approach": "East",
            "halted_pcu": float(i * 0.5),
            "flowing_pcu": 1.0,
            "queue_length_m": float(i * 5.0),
            "active_phase": "P2",
            "active_aspect": "RED",
        }
        for i in range(5)
    ]
    log_telemetry_batch(batch, db_path=temp_db)

    records = query_telemetry("inter_test", limit=10, db_path=temp_db)
    assert len(records) == 5
    # Should be ordered descending by timestamp
    assert records[0]["timestamp"] == 4.0
    assert records[-1]["timestamp"] == 0.0


def test_log_and_query_actuation(temp_db: Path):
    log_actuation_event(
        intersection_id="inter_test",
        timestamp=12.0,
        from_phase="P1",
        to_phase="P2",
        aspect="YELLOW",
        reason="GAP_OUT",
        message="Vehicle headway > 3.0s: dynamic green terminated",
        waiting_queue_pcu=6.5,
        db_path=temp_db,
    )

    events = query_actuation_events("inter_test", db_path=temp_db)
    assert len(events) == 1
    e = events[0]
    assert e["from_phase"] == "P1"
    assert e["to_phase"] == "P2"
    assert e["aspect"] == "YELLOW"
    assert e["reason"] == "GAP_OUT"
    assert e["waiting_queue_pcu"] == 6.5


def test_log_and_query_violations(temp_db: Path):
    log_violation(
        intersection_id="inter_test",
        track_id=42,
        vehicle_class="motorcycle",
        speed_km_h=45.2,
        timestamp=15.4,
        approach="South",
        image_path="/data/violations/15.4_42.jpg",
        db_path=temp_db,
    )

    viols = query_violations("inter_test", db_path=temp_db)
    assert len(viols) == 1
    v = viols[0]
    assert v["track_id"] == 42
    assert v["vehicle_class"] == "motorcycle"
    assert v["speed_km_h"] == 45.2
    assert v["approach"] == "South"
    assert v["image_path"] == "/data/violations/15.4_42.jpg"


def test_summary_stats_and_purge(temp_db: Path):
    log_telemetry(
        intersection_id="inter_stats",
        timestamp=1.0,
        approach="North",
        halted_pcu=2.0,
        flowing_pcu=1.0,
        queue_length_m=15.0,
        db_path=temp_db,
    )
    log_telemetry(
        intersection_id="inter_stats",
        timestamp=2.0,
        approach="North",
        halted_pcu=6.0,
        flowing_pcu=0.5,
        queue_length_m=42.0,
        db_path=temp_db,
    )
    log_actuation_event(
        intersection_id="inter_stats",
        timestamp=2.0,
        from_phase="P1",
        to_phase="P2",
        aspect="YELLOW",
        reason="GAP_OUT",
        message="Terminated",
        waiting_queue_pcu=6.0,
        db_path=temp_db,
    )
    log_violation(
        intersection_id="inter_stats",
        track_id=10,
        vehicle_class="car",
        speed_km_h=55.0,
        timestamp=2.5,
        approach="North",
        image_path="test.jpg",
        db_path=temp_db,
    )

    stats = get_summary_stats("inter_stats", db_path=temp_db)
    assert stats["total_telemetry_samples"] == 2
    assert stats["max_queue_length_m"] == 42.0
    assert stats["avg_halted_pcu"] == 4.0
    assert stats["gap_out_triggers"] == 1
    assert stats["wasted_green_saved_s"] == 8.5
    assert stats["total_violations"] == 1

    purge_all_data(db_path=temp_db)
    cleared_stats = get_summary_stats("inter_stats", db_path=temp_db)
    assert cleared_stats["total_telemetry_samples"] == 0
    assert cleared_stats["total_violations"] == 0
