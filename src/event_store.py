from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.fault_mapping import (
    _clean_event_payload,
    _next_event_id,
    get_event_by_id,
    load_events,
    normalize_events_dataframe,
    summarize_fault,
)

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - used only when PostgreSQL is enabled.
    psycopg = None
    sql = None
    dict_row = None
    Jsonb = None


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def postgres_driver_available() -> bool:
    """Return whether the optional psycopg driver is installed."""
    return psycopg is not None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _display_event_id(event_id: Any) -> Any:
    text = str(event_id)
    return int(text) if text.isdigit() else text


class PostgresEventStore:
    """Small PostgreSQL repository for event registration and similarity reads."""

    def __init__(
        self,
        database_url: str,
        table_name: str = "maintenance_events",
        seed_from_csv: bool = True,
        csv_path: str | Path = "data/banner.csv",
    ) -> None:
        if not _IDENTIFIER_RE.match(table_name):
            raise ValueError("Invalid PostgreSQL table name.")
        self.database_url = database_url
        self.table_name = table_name
        self.seed_from_csv = seed_from_csv
        self.csv_path = Path(csv_path)
        self._ready = False

    def ensure_ready(self) -> None:
        """Create the events table and optionally seed it from banner.csv."""
        if self._ready:
            return
        self._require_driver()
        with self._connect() as conn:
            self._create_table(conn)
            if self.seed_from_csv:
                self._seed_from_csv_if_empty(conn)
            conn.commit()
        self._ready = True

    def load_events(self) -> pd.DataFrame:
        """Load all events from PostgreSQL as the same flat shape used by CSV."""
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute(
                sql.SQL(
                    """
                    SELECT event_id, created_at, fault, payload
                    FROM {table}
                    ORDER BY created_at NULLS LAST, event_id
                    """
                ).format(table=sql.Identifier(self.table_name))
            ).fetchall()

        records = [self._row_to_event(row) for row in rows]
        if not records:
            return normalize_events_dataframe(pd.DataFrame(columns=["id", "created_at", "fault"]))
        return normalize_events_dataframe(pd.DataFrame(records))

    def get_event(self, event_id: int | str) -> dict[str, Any]:
        """Return one event by id."""
        return get_event_by_id(self.load_events(), event_id)

    def upsert_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Insert or update an event using event id as the natural key."""
        self.ensure_ready()
        payload, ignored_fields = _clean_event_payload(event, None)

        event_id = payload.get("id")
        with self._connect() as conn:
            if event_id is None or str(event_id).strip() == "":
                event_id = self._next_event_id(conn)
                payload["id"] = event_id

            event_id_text = str(event_id)
            existing = self._fetch_payload(conn, event_id_text)
            action = "updated" if existing is not None else "created"

            merged_payload = dict(existing or {})
            merged_payload.update(payload)
            merged_payload["id"] = _display_event_id(event_id_text)

            if not str(merged_payload.get("fault", "")).strip():
                raise ValueError("Novos eventos precisam informar o campo 'fault'.")

            if not str(merged_payload.get("created_at", "")).strip():
                merged_payload["created_at"] = datetime.now(timezone.utc).isoformat()

            summary = summarize_fault(merged_payload.get("fault"))
            merged_payload = _json_ready(merged_payload)

            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO {table}
                        (
                            event_id,
                            created_at,
                            fault,
                            fault_normalized,
                            fault_is_operational_state,
                            payload,
                            updated_at
                        )
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (event_id) DO UPDATE SET
                        created_at = EXCLUDED.created_at,
                        fault = EXCLUDED.fault,
                        fault_normalized = EXCLUDED.fault_normalized,
                        fault_is_operational_state = EXCLUDED.fault_is_operational_state,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    """
                ).format(table=sql.Identifier(self.table_name)),
                (
                    event_id_text,
                    merged_payload.get("created_at"),
                    summary.fault_raw,
                    summary.fault_normalized,
                    summary.is_operational_state,
                    Jsonb(merged_payload),
                ),
            )
            conn.commit()

        saved_event = self.get_event(event_id_text)
        return {
            "action": action,
            "id": saved_event["id"],
            "event": saved_event,
            "ignored_fields": ignored_fields,
            "storage": "postgresql",
        }

    def _require_driver(self) -> None:
        if psycopg is None:
            raise RuntimeError(
                "psycopg não está instalado. Rode `pip install -r requirements.txt` "
                "para usar PostgreSQL."
            )

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _create_table(self, conn) -> None:
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    event_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ,
                    fault TEXT NOT NULL,
                    fault_normalized TEXT NOT NULL,
                    fault_is_operational_state BOOLEAN NOT NULL DEFAULT false,
                    payload JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(table=sql.Identifier(self.table_name))
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} (fault_normalized)").format(
                index=sql.Identifier(f"{self.table_name}_fault_normalized_idx"),
                table=sql.Identifier(self.table_name),
            )
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} (created_at)").format(
                index=sql.Identifier(f"{self.table_name}_created_at_idx"),
                table=sql.Identifier(self.table_name),
            )
        )

    def _seed_from_csv_if_empty(self, conn) -> None:
        count = conn.execute(
            sql.SQL("SELECT COUNT(*) AS total FROM {table}").format(
                table=sql.Identifier(self.table_name)
            )
        ).fetchone()["total"]
        if count or not self.csv_path.exists():
            return

        df = load_events(self.csv_path)
        for row in df.to_dict(orient="records"):
            payload, _ignored_fields = _clean_event_payload(row, None)
            if payload.get("id") is None or not str(payload.get("fault", "")).strip():
                continue
            summary = summarize_fault(payload.get("fault"))
            payload = _json_ready(payload)
            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO {table}
                        (
                            event_id,
                            created_at,
                            fault,
                            fault_normalized,
                            fault_is_operational_state,
                            payload,
                            updated_at
                        )
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (event_id) DO NOTHING
                    """
                ).format(table=sql.Identifier(self.table_name)),
                (
                    str(payload["id"]),
                    payload.get("created_at"),
                    summary.fault_raw,
                    summary.fault_normalized,
                    summary.is_operational_state,
                    Jsonb(payload),
                ),
            )

    def _next_event_id(self, conn) -> int:
        row = conn.execute(
            sql.SQL(
                """
                SELECT COALESCE(MAX(event_id::bigint), 0) + 1 AS next_id
                FROM {table}
                WHERE event_id ~ '^[0-9]+$'
                """
            ).format(table=sql.Identifier(self.table_name))
        ).fetchone()
        return int(row["next_id"])

    def _fetch_payload(self, conn, event_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            sql.SQL("SELECT payload FROM {table} WHERE event_id = %s").format(
                table=sql.Identifier(self.table_name)
            ),
            (event_id,),
        ).fetchone()
        return dict(row["payload"]) if row else None

    def _row_to_event(self, row: dict[str, Any]) -> dict[str, Any]:
        event = dict(row["payload"] or {})
        event["id"] = _display_event_id(row["event_id"])
        if row.get("created_at") is not None:
            event["created_at"] = row["created_at"].isoformat()
        event["fault"] = row.get("fault") or event.get("fault", "")
        return event
