from __future__ import annotations

import sqlite3
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent import data


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / "tests_tmp" / ("auth_" + uuid.uuid4().hex)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_create_account_and_authenticate(self) -> None:
        account = data.create_account(self.temp_root, "alice", "secret123", "Alice")
        self.assertGreater(account["user_id"], 1)

        authenticated = data.authenticate_user(self.temp_root, "alice", "secret123")
        self.assertIsNotNone(authenticated)
        assert authenticated is not None
        self.assertEqual(authenticated["username"], "alice")
        self.assertEqual(authenticated["display_name"], "Alice")

    def test_user_data_is_isolated_by_user_id(self) -> None:
        alice = data.create_account(self.temp_root, "alice", "secret123", "Alice")
        bob = data.create_account(self.temp_root, "bob", "secret123", "Bob")

        data.log_daily_health_entry(self.temp_root, {"log_date": "2026-03-20", "water_ml": 1200}, user_id=alice["user_id"])
        data.log_daily_health_entry(self.temp_root, {"log_date": "2026-03-20", "water_ml": 2200}, user_id=bob["user_id"])

        alice_logs = data.get_recent_health_entries(self.temp_root, 30, user_id=alice["user_id"])
        bob_logs = data.get_recent_health_entries(self.temp_root, 30, user_id=bob["user_id"])

        self.assertEqual(len(alice_logs), 1)
        self.assertEqual(len(bob_logs), 1)
        self.assertEqual(float(alice_logs.iloc[0]["water_ml"]), 1200.0)
        self.assertEqual(float(bob_logs.iloc[0]["water_ml"]), 2200.0)

    def test_demo_login_seeds_rich_sample_data(self) -> None:
        authenticated = data.authenticate_user(self.temp_root, data.DEFAULT_DEMO_USERNAME, data.DEFAULT_DEMO_PASSWORD)
        self.assertIsNotNone(authenticated)

        logs = data.get_recent_health_entries(self.temp_root, 90, user_id=data.DEFAULT_USER_ID)
        symptoms = data.get_recent_joint_symptoms(self.temp_root, 90, user_id=data.DEFAULT_USER_ID)
        attacks = data.get_attack_history(self.temp_root, 365, user_id=data.DEFAULT_USER_ID)
        medications = data.get_medications(self.temp_root, user_id=data.DEFAULT_USER_ID)

        self.assertGreaterEqual(len(logs), 30)
        self.assertGreaterEqual(len(symptoms), 5)
        self.assertGreaterEqual(len(attacks), 3)
        self.assertGreaterEqual(len(medications), 2)

    def test_deactivate_account_blocks_future_login(self) -> None:
        account = data.create_account(self.temp_root, "alice", "secret123", "Alice")
        data.deactivate_account(self.temp_root, account["user_id"], "secret123")
        authenticated = data.authenticate_user(self.temp_root, "alice", "secret123")
        self.assertIsNone(authenticated)

    def test_init_db_migrates_legacy_schema_and_creates_backup(self) -> None:
        db_path = self.temp_root / "data" / data.DATABASE_NAME
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    gender TEXT,
                    birth_date TEXT,
                    height_cm REAL,
                    baseline_weight_kg REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE user_health_profile (
                    user_id INTEGER PRIMARY KEY,
                    has_gout_diagnosis INTEGER DEFAULT 0,
                    has_hyperuricemia INTEGER DEFAULT 0,
                    has_ckd INTEGER DEFAULT 0,
                    has_hypertension INTEGER DEFAULT 0,
                    has_diabetes INTEGER DEFAULT 0,
                    target_uric_acid REAL,
                    allergy_notes TEXT,
                    doctor_advice TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE joint_symptom_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    log_date TEXT NOT NULL,
                    body_site TEXT,
                    pain_score INTEGER,
                    swelling_flag INTEGER DEFAULT 0,
                    redness_flag INTEGER DEFAULT 0,
                    symptom_notes TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

        data.init_db(self.temp_root)

        with sqlite3.connect(db_path) as conn:
            account_columns = [row[1] for row in conn.execute("PRAGMA table_info(accounts)").fetchall()]
            symptom_columns = [row[1] for row in conn.execute("PRAGMA table_info(joint_symptom_logs)").fetchall()]
            background_job_columns = [row[1] for row in conn.execute("PRAGMA table_info(background_jobs)").fetchall()]
            care_plan_columns = [row[1] for row in conn.execute("PRAGMA table_info(care_plan_summaries)").fetchall()]
            care_plan_run_columns = [row[1] for row in conn.execute("PRAGMA table_info(care_plan_runs)").fetchall()]
            write_audit_columns = [row[1] for row in conn.execute("PRAGMA table_info(write_audit_logs)").fetchall()]
            user_version = conn.execute("PRAGMA user_version").fetchone()[0]

        self.assertIn("active_flag", account_columns)
        self.assertIn("last_login_at", account_columns)
        self.assertIn("stiffness_flag", symptom_columns)
        self.assertIn("status", background_job_columns)
        self.assertIn("plan_type", care_plan_columns)
        self.assertIn("status", care_plan_run_columns)
        self.assertIn("plan_json", care_plan_run_columns)
        self.assertIn("confirmed_flag", write_audit_columns)
        self.assertEqual(user_version, data.SCHEMA_VERSION)

        backups = list((self.temp_root / "data" / "backups").glob("*.db"))
        self.assertTrue(backups)


if __name__ == "__main__":
    unittest.main()
