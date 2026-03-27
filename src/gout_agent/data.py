from __future__ import annotations

import json
import hashlib
import hmac
import random
import sqlite3
import secrets
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DATABASE_NAME = "gout_management.db"
DEFAULT_USER_ID = 1
DEFAULT_DEMO_USERNAME = "demo"
DEFAULT_DEMO_PASSWORD = "demo123"
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
    CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        display_name TEXT NOT NULL,
        active_flag INTEGER DEFAULT 1,
        last_login_at TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
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
    CREATE TABLE IF NOT EXISTS joint_symptom_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        log_date TEXT NOT NULL,
        body_site TEXT,
        pain_score INTEGER,
        swelling_flag INTEGER DEFAULT 0,
        redness_flag INTEGER DEFAULT 0,
        stiffness_flag INTEGER DEFAULT 0,
        symptom_notes TEXT,
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
    CREATE TABLE IF NOT EXISTS digital_twin_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        snapshot_date TEXT NOT NULL,
        summary TEXT,
        profile_json TEXT NOT NULL,
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


def _hash_password(password: str, salt: str | None = None, iterations: int = 120000) -> str:
    actual_salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), actual_salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${actual_salt}${derived.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = _hash_password(password, salt=salt, iterations=int(iterations_text))
        return hmac.compare_digest(candidate, encoded)
    except Exception:
        return False


def _seed_demo_user_data(root: Path, user_id: int = DEFAULT_USER_ID) -> None:
    init_db(root)
    with get_connection(root) as conn:
        existing_logs = conn.execute("SELECT COUNT(1) FROM daily_logs WHERE user_id = ?", (user_id,)).fetchone()[0]
        existing_symptoms = conn.execute("SELECT COUNT(1) FROM joint_symptom_logs WHERE user_id = ?", (user_id,)).fetchone()[0]
        existing_attacks = conn.execute("SELECT COUNT(1) FROM gout_attacks WHERE user_id = ?", (user_id,)).fetchone()[0]
        existing_medications = conn.execute("SELECT COUNT(1) FROM medications WHERE user_id = ?", (user_id,)).fetchone()[0]
        if any(count > 0 for count in [existing_logs, existing_symptoms, existing_attacks, existing_medications]):
            return

        rng = random.Random(20260326)
        now = datetime.now().isoformat(timespec="seconds")
        today = date.today()

        conn.execute(
            """
            UPDATE users
            SET name = ?, gender = ?, birth_date = ?, height_cm = ?, baseline_weight_kg = ?
            WHERE id = ?
            """,
            ("演示管理员", "male", "1989-08-16", 174.0, 76.5, user_id),
        )
        conn.execute(
            """
            UPDATE accounts
            SET display_name = ?
            WHERE user_id = ?
            """,
            ("演示管理员", user_id),
        )
        conn.execute(
            """
            UPDATE user_health_profile
            SET has_gout_diagnosis = 1,
                has_hyperuricemia = 1,
                has_ckd = 0,
                has_hypertension = 1,
                has_diabetes = 0,
                target_uric_acid = 360.0,
                allergy_notes = ?,
                doctor_advice = ?,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                "对秋水仙碱轻度胃肠道不耐受。",
                "优先围绕饮水、规律服药和晚间饮酒控制进行长期管理。",
                now,
                user_id,
            ),
        )

        diet_pool = ["海鲜", "火锅", "烧烤", "啤酒", "红肉", "清淡饮食", "家常菜", "外卖", "夜宵"]
        symptom_pool = ["右脚大脚趾轻微酸痛", "左脚踝发紧", "晨起无明显不适", "晚间走路时脚趾更敏感", "今天状态平稳"]
        alcohol_pool = ["none", "none", "none", "beer", "wine", "spirits"]
        for offset in range(59, -1, -1):
            log_day = today - timedelta(days=offset)
            water_ml = max(700, int(rng.gauss(1650, 380)))
            pain_score = 0
            if offset % 11 == 0:
                pain_score = rng.randint(4, 7)
            elif offset % 5 == 0:
                pain_score = rng.randint(1, 3)
            alcohol_intake = rng.choice(alcohol_pool)
            diet_notes = "、".join(rng.sample(diet_pool, k=2))
            symptom_notes = rng.choice(symptom_pool)
            joint_pain_flag = 1 if pain_score >= 2 else 0
            medication_taken_flag = 0 if offset % 9 == 0 else 1
            conn.execute(
                """
                INSERT INTO daily_logs (
                    user_id, log_date, weight_kg, water_ml, steps, exercise_minutes, sleep_hours,
                    alcohol_intake, diet_notes, symptom_notes, pain_score, joint_pain_flag,
                    medication_taken_flag, free_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    log_day.isoformat(),
                    round(76.5 + rng.uniform(-1.2, 1.0), 1),
                    water_ml,
                    max(1200, int(rng.gauss(5600, 1800))),
                    max(0, int(rng.gauss(22, 14))),
                    round(max(4.5, rng.gauss(6.8, 0.9)), 1),
                    alcohol_intake,
                    diet_notes,
                    symptom_notes,
                    pain_score,
                    joint_pain_flag,
                    medication_taken_flag,
                    "演示数据，用于展示数字分身与风险变化。",
                    f"{log_day.isoformat()}T20:00:00",
                ),
            )

        symptom_sites = ["右脚大脚趾", "左脚大脚趾", "右脚踝", "左脚踝", "右足背"]
        for offset in [2, 4, 7, 11, 16, 22, 29, 37, 45, 54]:
            log_day = today - timedelta(days=offset)
            site = rng.choice(symptom_sites)
            pain_score = rng.randint(3, 8)
            conn.execute(
                """
                INSERT INTO joint_symptom_logs (
                    user_id, log_date, body_site, pain_score, swelling_flag, redness_flag,
                    stiffness_flag, symptom_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    log_day.isoformat(),
                    site,
                    pain_score,
                    1 if pain_score >= 5 else 0,
                    1 if pain_score >= 6 else 0,
                    1 if pain_score >= 4 else 0,
                    f"{site}在晚间更明显，步行后加重。",
                    f"{log_day.isoformat()}T21:00:00",
                ),
            )

        attack_samples = [
            (12, "右脚大脚趾", 7, 30, "啤酒、饮水不足"),
            (33, "左脚大脚趾", 6, 20, "火锅、夜宵"),
            (68, "右脚踝", 8, 42, "海鲜、未按时服药"),
            (104, "左脚踝", 5, 16, "饮水不足"),
            (151, "右脚大脚趾", 7, 36, "啤酒、烧烤"),
        ]
        for offset, site, pain_score, duration_hours, trigger in attack_samples:
            attack_day = today - timedelta(days=offset)
            conn.execute(
                """
                INSERT INTO gout_attacks (
                    user_id, attack_date, joint_site, pain_score, swelling_flag, redness_flag,
                    duration_hours, suspected_trigger, resolved_flag, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    attack_day.isoformat(),
                    site,
                    pain_score,
                    1,
                    1 if pain_score >= 7 else 0,
                    duration_hours,
                    trigger,
                    1,
                    f"{site}发作，休息和补水后逐渐缓解。",
                    f"{attack_day.isoformat()}T22:00:00",
                ),
            )

        lab_samples = [
            (12, 498.0, 82.0, 96.0),
            (46, 522.0, 88.0, 91.0),
            (93, 476.0, 84.0, 95.0),
        ]
        for offset, uric_acid, creatinine, egfr in lab_samples:
            test_day = today - timedelta(days=offset)
            conn.execute(
                """
                INSERT INTO lab_results (
                    user_id, test_date, uric_acid, creatinine, egfr, crp, esr, ast, alt, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    test_day.isoformat(),
                    uric_acid,
                    creatinine,
                    egfr,
                    None,
                    None,
                    None,
                    None,
                    "演示化验数据。",
                    f"{test_day.isoformat()}T10:00:00",
                ),
            )

        medications = [
            ("非布司他", "40mg", "每日一次", "降尿酸"),
            ("秋水仙碱", "0.5mg", "急性期按需", "缓解发作"),
        ]
        medication_ids: list[int] = []
        for name, dose, frequency, purpose in medications:
            cursor = conn.execute(
                """
                INSERT INTO medications (
                    user_id, medication_name, dose, frequency, start_date, end_date, purpose, active_flag, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    name,
                    dose,
                    frequency,
                    (today - timedelta(days=120)).isoformat(),
                    None,
                    purpose,
                    1,
                    now,
                ),
            )
            medication_ids.append(int(cursor.lastrowid))

        for offset in range(29, -1, -1):
            created_at = datetime.combine(today - timedelta(days=offset), datetime.min.time()).replace(hour=8).isoformat(timespec="seconds")
            status = "missed" if offset in {3, 9, 18, 24} else "taken"
            conn.execute(
                """
                INSERT INTO medication_logs (user_id, medication_id, scheduled_time, taken_time, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    medication_ids[0],
                    created_at,
                    created_at if status == "taken" else None,
                    status,
                    created_at,
                ),
            )

        conn.commit()


def init_db(root: Path) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    demo_password_hash = _hash_password(DEFAULT_DEMO_PASSWORD)
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
            INSERT OR IGNORE INTO accounts (
                user_id, username, password_hash, display_name, active_flag, last_login_at, created_at
            )
            VALUES (?, ?, ?, ?, 1, NULL, ?)
            """,
            (DEFAULT_USER_ID, DEFAULT_DEMO_USERNAME, demo_password_hash, DEFAULT_USER["name"], now),
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
        conn.execute(
            """
            UPDATE accounts
            SET display_name = COALESCE(NULLIF(display_name, ''), ?),
                password_hash = COALESCE(NULLIF(password_hash, ''), ?)
            WHERE user_id = ?
            """,
            (DEFAULT_USER["name"], demo_password_hash, DEFAULT_USER_ID),
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _frame_from_query(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def get_account_by_username(root: Path, username: str) -> dict[str, Any] | None:
    init_db(root)
    with get_connection(root) as conn:
        row = conn.execute(
            """
            SELECT a.*, u.name
            FROM accounts a
            JOIN users u ON u.id = a.user_id
            WHERE lower(a.username) = lower(?)
            """,
            (username.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def get_account_by_user_id(root: Path, user_id: int) -> dict[str, Any] | None:
    init_db(root)
    with get_connection(root) as conn:
        row = conn.execute(
            """
            SELECT a.*, u.name
            FROM accounts a
            JOIN users u ON u.id = a.user_id
            WHERE a.user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return _row_to_dict(row)


def create_account(root: Path, username: str, password: str, display_name: str) -> dict[str, Any]:
    init_db(root)
    normalized_username = username.strip()
    normalized_display_name = display_name.strip() or normalized_username
    if not normalized_username:
        raise ValueError("用户名不能为空。")
    if len(password) < 6:
        raise ValueError("密码长度至少需要 6 位。")
    if get_account_by_username(root, normalized_username):
        raise ValueError("该用户名已存在。")

    now = datetime.now().isoformat(timespec="seconds")
    password_hash = _hash_password(password)
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, gender, birth_date, height_cm, baseline_weight_kg, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (normalized_display_name, DEFAULT_USER["gender"], DEFAULT_USER["birth_date"], DEFAULT_USER["height_cm"], DEFAULT_USER["baseline_weight_kg"], now),
        )
        user_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO accounts (user_id, username, password_hash, display_name, active_flag, last_login_at, created_at)
            VALUES (?, ?, ?, ?, 1, NULL, ?)
            """,
            (user_id, normalized_username, password_hash, normalized_display_name, now),
        )
        conn.execute(
            """
            INSERT INTO user_health_profile (
                user_id, has_gout_diagnosis, has_hyperuricemia, has_ckd, has_hypertension,
                has_diabetes, target_uric_acid, allergy_notes, doctor_advice, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                DEFAULT_PROFILE["has_gout_diagnosis"],
                DEFAULT_PROFILE["has_hyperuricemia"],
                DEFAULT_PROFILE["has_ckd"],
                DEFAULT_PROFILE["has_hypertension"],
                DEFAULT_PROFILE["has_diabetes"],
                DEFAULT_PROFILE["target_uric_acid"],
                DEFAULT_PROFILE["allergy_notes"],
                DEFAULT_PROFILE["doctor_advice"],
                now,
            ),
        )
        conn.commit()
    return {"user_id": user_id, "username": normalized_username, "display_name": normalized_display_name}


def authenticate_user(root: Path, username: str, password: str) -> dict[str, Any] | None:
    account = get_account_by_username(root, username)
    if not account or not int(account.get("active_flag") or 0):
        return None
    if not _verify_password(password, str(account.get("password_hash") or "")):
        return None
    if str(account.get("username")) == DEFAULT_DEMO_USERNAME and int(account.get("user_id") or 0) == DEFAULT_USER_ID:
        _seed_demo_user_data(root, user_id=DEFAULT_USER_ID)
        account = get_account_by_username(root, username) or account

    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        conn.execute("UPDATE accounts SET last_login_at = ? WHERE id = ?", (now, account["id"]))
        conn.commit()
    return {
        "user_id": int(account["user_id"]),
        "username": str(account["username"]),
        "display_name": str(account.get("display_name") or account.get("name") or account["username"]),
    }


def update_account_password(root: Path, user_id: int, current_password: str, new_password: str) -> None:
    account = get_account_by_user_id(root, user_id)
    if not account:
        raise ValueError("当前账号不存在。")
    if not _verify_password(current_password, str(account.get("password_hash") or "")):
        raise ValueError("当前密码不正确。")
    if len(new_password) < 6:
        raise ValueError("新密码长度至少需要 6 位。")

    encoded = _hash_password(new_password)
    with get_connection(root) as conn:
        conn.execute("UPDATE accounts SET password_hash = ? WHERE user_id = ?", (encoded, user_id))
        conn.commit()


def deactivate_account(root: Path, user_id: int, current_password: str) -> None:
    account = get_account_by_user_id(root, user_id)
    if not account:
        raise ValueError("当前账号不存在。")
    if int(account.get("user_id") or 0) == DEFAULT_USER_ID:
        raise ValueError("演示账号不支持注销。")
    if not _verify_password(current_password, str(account.get("password_hash") or "")):
        raise ValueError("当前密码不正确。")

    with get_connection(root) as conn:
        conn.execute(
            "UPDATE accounts SET active_flag = 0 WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()


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
            if "name" in user_fields:
                conn.execute(
                    "UPDATE accounts SET display_name = ? WHERE user_id = ?",
                    (user_fields["name"], user_id),
                )
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


def log_joint_symptom(root: Path, payload: dict[str, Any], user_id: int = DEFAULT_USER_ID) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "user_id": user_id,
        "log_date": payload.get("log_date", str(date.today())),
        "body_site": payload.get("body_site"),
        "pain_score": payload.get("pain_score", 0),
        "swelling_flag": int(bool(payload.get("swelling_flag"))),
        "redness_flag": int(bool(payload.get("redness_flag"))),
        "stiffness_flag": int(bool(payload.get("stiffness_flag"))),
        "symptom_notes": payload.get("symptom_notes"),
        "created_at": now,
    }
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO joint_symptom_logs (
                user_id, log_date, body_site, pain_score, swelling_flag, redness_flag,
                stiffness_flag, symptom_notes, created_at
            ) VALUES (
                :user_id, :log_date, :body_site, :pain_score, :swelling_flag, :redness_flag,
                :stiffness_flag, :symptom_notes, :created_at
            )
            """,
            record,
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_joint_symptoms(root: Path, days: int = 90, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    cutoff = (date.today() - timedelta(days=max(days - 1, 0))).isoformat()
    with get_connection(root) as conn:
        frame = _frame_from_query(
            conn,
            """
            SELECT * FROM joint_symptom_logs
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


def save_digital_twin_profile(
    root: Path,
    profile_payload: dict[str, Any],
    snapshot_date: str | None = None,
    user_id: int = DEFAULT_USER_ID,
) -> int:
    init_db(root)
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(root) as conn:
        cursor = conn.execute(
            """
            INSERT INTO digital_twin_profiles (user_id, snapshot_date, summary, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                snapshot_date or str(date.today()),
                profile_payload.get("summary"),
                json.dumps(profile_payload, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_digital_twin_profiles(root: Path, limit: int = 20, user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    init_db(root)
    with get_connection(root) as conn:
        frame = _frame_from_query(
            conn,
            """
            SELECT * FROM digital_twin_profiles
            WHERE user_id = ?
            ORDER BY snapshot_date DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
    if "profile_json" in frame.columns:
        frame["profile_payload"] = frame["profile_json"].map(lambda value: json.loads(value) if value else {})
    return frame


def get_latest_digital_twin_profile(root: Path, user_id: int = DEFAULT_USER_ID) -> dict[str, Any] | None:
    profiles = get_digital_twin_profiles(root, limit=1, user_id=user_id)
    if profiles.empty:
        return None
    payload = profiles.iloc[0].get("profile_payload")
    return payload if isinstance(payload, dict) else None


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
