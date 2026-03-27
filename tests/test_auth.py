from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
