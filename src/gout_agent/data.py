from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DATABASE_NAME = "gout_management.db"
DEFAULT_USER_ID = 1
DEFAULT_USER = {
    "name": "Demo User",
    "gender": "unknown",
    "birth_date": None,
    "height_cm": None,
    "baseline_weight_kg": None,
}
DEFAULT_PROFILE = {
    "has_gout_diagnosis": 1,
    "has_hyperuricemia": 1,
    "has_ckd": 0,
    "has_hypertension": 0,
    "has_diabetes": 0,
    "target_uric_acid": 360.0,
    "allergy_notes": "",
    "doctor_advice": "",
}


def _db_path(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / DATABASE_NAME


def get_connection(root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(root))
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        gender TEXT,
        birth_date TEXT,
        height_cm REAL,
        baseline_weight_kg REAL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_health_profile (
        user_id INTEGER PRIMARY KEY,
        has_gout_diagnosis INTEGER DEFAULT 0,
        has_hyperuricemia INTEGER DEFAULT 0,
        has_ckd INTEGER DEFAULT 0,
        has_hypertension INTEGER DEFAULT 0,
        has_diabetes INTEGER DEFAULT 0,
        target_uric_acid REAL,
        allergy_notes TEXT,
        doctor_advice TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        log_date TEXT NOT NULL,
        weight_kg REAL,
        water_ml REAL,
        steps INTEGER,
        exercise_minutes INTEGER,
        sleep_hours REAL,
        alcohol_intake TEXT,
        diet_notes TEXT,
        symptom_notes TEXT,
        pain_score INTEGER,
        joint_pain_flag INTEGER DEFAULT 0,
        medication_taken_flag INTEGER DEFAULT 0,
        free_text TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lab_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        test_date TEXT NOT NULL,
        uric_acid REAL,
        creatinine REAL,
        egfr REAL,
        crp REAL,
        esr REAL,
        ast REAL,
        alt REAL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gout_attacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        attack_date TEXT NOT NULL,
        joint_site TEXT,
        pain_score INTEGER,
        swelling_flag INTEGER DEFAULT 0,
        redness_flag INTEGER DEFAULT 0,
        duration_hours REAL,
        suspected_trigger TEXT,
        resolved_flag INTEGER DEFAULT 0,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        medication_name TEXT NOT NULL,
        dose TEXT,
        frequency TEXT,
        start_date TEXT,
        end_date TEXT,
        purpose TEXT,
        active_flag INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS medication_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        medication_id INTEGER NOT NULL,
        scheduled_time TEXT,
        taken_time TEXT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(medication_id) REFERENCES medications(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        reminder_type TEXT NOT NULL,
        title TEXT NOT NULL,
        schedule_rule TEXT,
        next_trigger_at TEXT,
        active_flag INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        snapshot_date TEXT NOT NULL,
        uric_acid_risk_level TEXT,
        attack_risk_level TEXT,
        overall_risk_score REAL,
        top_risk_factors TEXT,
        trend_direction TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        report_type TEXT NOT NULL,
        period_start TEXT,
        period_end TEXT,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        memory_type TEXT NOT NULL,
        memory_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
]


def init_db(root: Path) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.execute(
            """
            INSERT OR IGNORE INTO users (id, name, gender, birth_date, height_cm, baseline_weight_kg, created_at)
            VALUES (:id, :name, :gender, :birth_date, :height_cm, :baseline_weight_kg, :created_at)
            """,
            {"id": DEFAULT_USER_ID, **DEFAULT_USER, "created_at": now},
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO user_health_profile (
                user_id, has_gout_diagnosis, has_hyperuricemia, has_ckd, has_hypertension,
                has_diabetes, target_uric_acid, allergy_notes, doctor_advice, updated_at
            )
            VALUES (
                :user_id, :has_gout_diagnosis, :has_hyperuricemia, :has_ckd, :has_hypertension,
                :has_diabetes, :target_uric_acid, :allergy_notes, :doctor_advice, :updated_at
            )
            """,
            {"user_id": DEFAULT_USER_ID, **DEFAULT_PROFILE, "updated_at": now},
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _frame_from_query(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def get_user_profile(root: Path, user_id: int = DEFAULT_USER_ID) -> dict[str, Any]:
    init_db(root)
    with get_connection(root) as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        profile = conn.execute("SELECT * FROM user_health_profile WHERE user_id = ?", (user_id,)).fetchone()
    payload = _row_to_dict(user) or {}
    payload.update(_row_to_dict(profile) or {})
    return payload


def update_user_profile(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> dict[str, Any]:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    user_fields = {key: payload.get(key) for key in ["name", "gender", "birth_date", "height_cm", "baseline_weight_kg"] if key in payload}
    profile_fields = {
        key: payload.get(key)
        for key in [
            "has_gout_diagnosis",
            "has_hyperuricemia",
            "has_ckd",
            "has_hypertension",
            "has_diabetes",
            "target_uric_acid",
            "allergy_notes",
            "doctor_advice",
        ]
        if key in payload
    }
    with get_connection(root) as conn:
        if user_fields:
            assignments = ", ".join(f"{field} = :{field}" for field in user_fields)
            conn.execute(f"UPDATE users SET {assignments} WHERE id = :user_id", {**user_fields, "user_id": user_id})
        if profile_fields:
            assignments = ", ".join(f"{field} = :{field}" for field in profile_fields)
            conn.execute(
                f"UPDATE user_health_profile SET {assignments}, updated_at = :updated_at WHERE user_id = :user_id",
                {**profile_fields, "updated_at": now, "user_id": user_id},
            )
        conn.commit()
    return get_user_profile(root, user_id)


def log_daily_health_entry(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "user_id": user_id,
        "log_date": payload.get("log_date", str(date.today())),
        "weight_kg": payload.get("weight_kg"),
        "water_ml": payload.get("water_ml"),
        "steps": payload.get("steps"),
        "exercise_minutes": payload.get("exercise_minutes"),
        "sleep_hours": payload.get("sleep_hours"),
        "alcohol_intake": payload.get("alcohol_intake"),
        "diet_notes": payload.get("diet_notes"),
        "symptom_notes": payload.get("symptom_notes"),
        "pain_score": payload.get("pain_score", 0),
        "joint_pain_flag": int(bool(payload.get("joint_pain_flag"))),
        "medication_taken_flag": int(bool(payload.get("medication_taken_flag"))),
        "free_text": payload.get("free_text"),
        "created_at": now,
    }
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO daily_logs (
                user_id, log_date, weight_kg, water_ml, steps, exercise_minutes, sleep_hours,
                alcohol_intake, diet_notes, symptom_notes, pain_score, joint_pain_flag,
                medication_taken_flag, free_text, created_at
            ) VALUES (
                :user_id, :log_date, :weight_kg, :water_ml, :steps, :exercise_minutes, :sleep_hours,
                :alcohol_intake, :diet_notes, :symptom_notes, :pain_score, :joint_pain_flag,
                :medication_taken_flag, :free_text, :created_at
            )
            """,
            record,
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_health_entries(root: Path, days: int = 30, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    cutoff = (date.today() - timedelta(days=max(days - 1, 0))).isoformat()
    with get_connection(root) as conn:
        frame = _frame_from_query(
            conn,
            """
            SELECT * FROM daily_logs
            WHERE user_id = ? AND log_date >= ?
            ORDER BY log_date ASC, id ASC
            """,
            (user_id, cutoff),
        )
    return frame


def log_lab_result(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "user_id": user_id,
        "test_date": payload.get("test_date", str(date.today())),
        "uric_acid": payload.get("uric_acid"),
        "creatinine": payload.get("creatinine"),
        "egfr": payload.get("egfr"),
        "crp": payload.get("crp"),
        "esr": payload.get("esr"),
        "ast": payload.get("ast"),
        "alt": payload.get("alt"),
        "notes": payload.get("notes"),
        "created_at": now,
    }
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO lab_results (
                user_id, test_date, uric_acid, creatinine, egfr, crp, esr, ast, alt, notes, created_at
            ) VALUES (
                :user_id, :test_date, :uric_acid, :creatinine, :egfr, :crp, :esr, :ast, :alt, :notes, :created_at
            )
            """,
            record,
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_lab_history(root: Path, metric_name: str | None = None, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    with get_connection(root) as conn:
        frame = _frame_from_query(
            conn,
            "SELECT * FROM lab_results WHERE user_id = ? ORDER BY test_date ASC, id ASC",
            (user_id,),
        )
    if metric_name and metric_name in frame.columns:
        keep = ["id", "test_date", metric_name, "notes"]
        return frame[keep].copy()
    return frame


def log_gout_attack(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "user_id": user_id,
        "attack_date": payload.get("attack_date", str(date.today())),
        "joint_site": payload.get("joint_site"),
        "pain_score": payload.get("pain_score", 0),
        "swelling_flag": int(bool(payload.get("swelling_flag"))),
        "redness_flag": int(bool(payload.get("redness_flag"))),
        "duration_hours": payload.get("duration_hours"),
        "suspected_trigger": payload.get("suspected_trigger"),
        "resolved_flag": int(bool(payload.get("resolved_flag"))),
        "notes": payload.get("notes"),
        "created_at": now,
    }
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO gout_attacks (
                user_id, attack_date, joint_site, pain_score, swelling_flag, redness_flag,
                duration_hours, suspected_trigger, resolved_flag, notes, created_at
            ) VALUES (
                :user_id, :attack_date, :joint_site, :pain_score, :swelling_flag, :redness_flag,
                :duration_hours, :suspected_trigger, :resolved_flag, :notes, :created_at
            )
            """,
            record,
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_attack_history(root: Path, days: int = 180, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    cutoff = (date.today() - timedelta(days=max(days - 1, 0))).isoformat()
    with get_connection(root) as conn:
        return _frame_from_query(
            conn,
            "SELECT * FROM gout_attacks WHERE user_id = ? AND attack_date >= ? ORDER BY attack_date DESC, id DESC",
            (user_id, cutoff),
        )


def add_medication(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "user_id": user_id,
        "medication_name": payload.get("medication_name"),
        "dose": payload.get("dose"),
        "frequency": payload.get("frequency"),
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
        "purpose": payload.get("purpose"),
        "active_flag": int(bool(payload.get("active_flag", True))),
        "created_at": now,
    }
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO medications (
                user_id, medication_name, dose, frequency, start_date, end_date, purpose, active_flag, created_at
            ) VALUES (
                :user_id, :medication_name, :dose, :frequency, :start_date, :end_date, :purpose, :active_flag, :created_at
            )
            """,
            record,
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_medications(root: Path, active_only: bool = False, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    query = "SELECT * FROM medications WHERE user_id = ?"
    params: tuple[Any, ...] = (user_id,)
    if active_only:
        query += " AND active_flag = 1"
    query += " ORDER BY active_flag DESC, created_at DESC"
    with get_connection(root) as conn:
        return _frame_from_query(conn, query, params)


def log_medication_taken(
    root: Path,
    medication_id: int,
    status: str,
    scheduled_time: str | None = None,
    taken_time: str | None = None,
    user_id: int = DEFAULT_USER_ID,
) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO medication_logs (user_id, medication_id, scheduled_time, taken_time, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, medication_id, scheduled_time, taken_time, status, now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_medication_adherence(root: Path, days: int = 30, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    cutoff = datetime.now() - timedelta(days=max(days - 1, 0))
    with get_connection(root) as conn:
        return _frame_from_query(
            conn,
            """
            SELECT ml.*, m.medication_name
            FROM medication_logs ml
            JOIN medications m ON m.id = ml.medication_id
            WHERE ml.user_id = ? AND ml.created_at >= ?
            ORDER BY ml.created_at DESC, ml.id DESC
            """,
            (user_id, cutoff.isoformat(timespec="seconds")),
        )


def create_reminder(root: Path, reminder_type: str, title: str, schedule_rule: str, next_trigger_at: str, user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reminders (user_id, reminder_type, title, schedule_rule, next_trigger_at, active_flag, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (user_id, reminder_type, title, schedule_rule, next_trigger_at, now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_active_reminders(root: Path, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    with get_connection(root) as conn:
        return _frame_from_query(
            conn,
            "SELECT * FROM reminders WHERE user_id = ? AND active_flag = 1 ORDER BY next_trigger_at ASC, id ASC",
            (user_id,),
        )


def save_risk_snapshot(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    factors = payload.get("top_risk_factors") or []
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO risk_snapshots (
                user_id, snapshot_date, uric_acid_risk_level, attack_risk_level,
                overall_risk_score, top_risk_factors, trend_direction, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                payload.get("snapshot_date", str(date.today())),
                payload.get("uric_acid_risk_level"),
                payload.get("attack_risk_level"),
                payload.get("overall_risk_score"),
                json.dumps(factors, ensure_ascii=True),
                payload.get("trend_direction"),
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_risk_snapshots(root: Path, days: int = 30, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    cutoff = (date.today() - timedelta(days=max(days - 1, 0))).isoformat()
    with get_connection(root) as conn:
        return _frame_from_query(
            conn,
            "SELECT * FROM risk_snapshots WHERE user_id = ? AND snapshot_date >= ? ORDER BY snapshot_date ASC, id ASC",
            (user_id, cutoff),
        )


def save_report(root: Path, report_type: str, report: dict[str, Any], period_start: str | None, period_end: str | None, user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports (user_id, report_type, period_start, period_end, report_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, report_type, period_start, period_end, json.dumps(report, ensure_ascii=True, indent=2), now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_reports(root: Path, report_type: str | None = None, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    query = "SELECT * FROM reports WHERE user_id = ?"
    params: list[Any] = [user_id]
    if report_type:
        query += " AND report_type = ?"
        params.append(report_type)
    query += " ORDER BY created_at DESC, id DESC"
    with get_connection(root) as conn:
        return _frame_from_query(conn, query, tuple(params))


def save_session_memory(
    root: Path,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    user_id: int = DEFAULT_USER_ID,
) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO session_memories (user_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, content, json.dumps(metadata or {}, ensure_ascii=True), now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_session_memories(root: Path, limit: int = 20, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    with get_connection(root) as conn:
        frame = _frame_from_query(
            conn,
            """
            SELECT * FROM session_memories
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, max(limit, 1)),
        )
    if "metadata_json" in frame.columns:
        frame["metadata"] = frame["metadata_json"].map(lambda value: json.loads(value) if value else {})
    return frame


def save_memory_snapshot(
    root: Path,
    memory_type: str,
    payload: dict[str, Any],
    user_id: int = DEFAULT_USER_ID,
) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO memory_snapshots (user_id, memory_type, memory_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, memory_type, json.dumps(payload, ensure_ascii=True), now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_memory_snapshots(
    root: Path,
    memory_type: str | None = None,
    limit: int = 20,
    user_id: int = DEFAULT_USER_ID,
) -> pd.DataFrame:
    init_db(root)
    query = "SELECT * FROM memory_snapshots WHERE user_id = ?"
    params: list[Any] = [user_id]
    if memory_type:
        query += " AND memory_type = ?"
        params.append(memory_type)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(max(limit, 1))
    with get_connection(root) as conn:
        frame = _frame_from_query(conn, query, tuple(params))
    if "memory_json" in frame.columns:
        frame["memory_payload"] = frame["memory_json"].map(lambda value: json.loads(value) if value else {})
    return frame


def get_latest_memory_snapshot(
    root: Path,
    memory_type: str,
    user_id: int = DEFAULT_USER_ID,
) -> dict[str, Any] | None:
    snapshots = get_memory_snapshots(root, memory_type=memory_type, limit=1, user_id=user_id)
    if snapshots.empty:
        return None
    row = snapshots.iloc[0].to_dict()
    payload = row.get("memory_payload")
    return payload if isinstance(payload, dict) else None
