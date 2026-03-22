from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent import memory


class MemoryTests(unittest.TestCase):
    def test_build_long_term_memory_contains_expected_sections(self) -> None:
        profile = {"target_uric_acid": 360, "doctor_advice": "控制尿酸，规律复查，减少饮酒。"}
        logs = pd.DataFrame(
            [
                {"log_date": "2026-03-20", "water_ml": 1800, "steps": 6000, "exercise_minutes": 20, "sleep_hours": 7, "alcohol_intake": "beer", "diet_notes": "海鲜 火锅", "pain_score": 1, "medication_taken_flag": 1},
                {"log_date": "2026-03-21", "water_ml": 2200, "steps": 8000, "exercise_minutes": 35, "sleep_hours": 7.5, "alcohol_intake": "none", "diet_notes": "鸡蛋 蔬菜", "pain_score": 0, "medication_taken_flag": 1},
            ]
        )
        labs = pd.DataFrame([{"test_date": "2026-03-21", "uric_acid": 510}])
        attacks = pd.DataFrame(
            [
                {"attack_date": "2026-01-01", "joint_site": "右脚大脚趾", "pain_score": 7, "suspected_trigger": "饮酒"},
                {"attack_date": "2026-03-01", "joint_site": "右脚大脚趾", "pain_score": 8, "suspected_trigger": "海鲜"},
            ]
        )

        result = memory.build_long_term_memory(profile, logs, labs, attacks)

        self.assertIn("user_preferences", result)
        self.assertIn("ai_advice_summary", result)
        self.assertIn("attack_patterns", result)
        self.assertIn("behavior_portraits", result)
        self.assertIn("gout_management_twin_profile", result)
        self.assertIn("7d", result["behavior_portraits"])
        self.assertEqual(result["attack_patterns"]["common_joint_site"], "右脚大脚趾")

    def test_behavior_portrait_uses_requested_window(self) -> None:
        logs = pd.DataFrame(
            [
                {"log_date": "2026-03-01", "water_ml": 1200, "steps": 3000, "exercise_minutes": 10, "sleep_hours": 6, "alcohol_intake": "beer", "pain_score": 2, "medication_taken_flag": 0},
                {"log_date": "2026-03-21", "water_ml": 2400, "steps": 9000, "exercise_minutes": 40, "sleep_hours": 8, "alcohol_intake": "none", "pain_score": 0, "medication_taken_flag": 1},
            ]
        )
        labs = pd.DataFrame([{"test_date": "2026-03-21", "uric_acid": 430}])
        attacks = pd.DataFrame([{"attack_date": "2026-03-20", "joint_site": "踝关节", "pain_score": 6, "suspected_trigger": "火锅"}])

        portrait = memory.build_behavior_portrait(logs, labs, attacks, 7)

        self.assertEqual(portrait["window_days"], 7)
        self.assertEqual(portrait["days_with_logs"], 1)
        self.assertEqual(portrait["attack_count"], 1)
        self.assertEqual(portrait["latest_uric_acid"], 430)

    def test_twin_profile_contains_expected_sections(self) -> None:
        profile = {"target_uric_acid": 360}
        logs = pd.DataFrame(
            [
                {"log_date": "2026-03-18", "water_ml": 1200, "steps": 3000, "exercise_minutes": 10, "sleep_hours": 6, "alcohol_intake": "beer", "diet_notes": "海鲜 火锅", "pain_score": 2, "medication_taken_flag": 0},
                {"log_date": "2026-03-19", "water_ml": 1300, "steps": 3500, "exercise_minutes": 10, "sleep_hours": 6.5, "alcohol_intake": "beer", "diet_notes": "烧烤", "pain_score": 3, "medication_taken_flag": 0},
                {"log_date": "2026-03-21", "water_ml": 1800, "steps": 6000, "exercise_minutes": 20, "sleep_hours": 7, "alcohol_intake": "none", "diet_notes": "鸡蛋 蔬菜", "pain_score": 1, "medication_taken_flag": 1},
            ]
        )
        labs = pd.DataFrame([{"test_date": "2026-03-21", "uric_acid": 520}])
        attacks = pd.DataFrame([{"attack_date": "2026-03-20", "joint_site": "右脚大脚趾", "pain_score": 7, "suspected_trigger": "海鲜"}])

        twin = memory.build_gout_management_twin_profile(profile, logs, labs, attacks)

        self.assertIn("summary", twin)
        self.assertIn("top_triggers", twin)
        self.assertIn("trigger_patterns", twin)
        self.assertIn("risk_windows", twin)
        self.assertIn("behavior_patterns", twin)
        self.assertIn("management_stability", twin)
        self.assertIn("current_shortcomings", twin)
        self.assertTrue(twin["top_triggers"])


if __name__ == "__main__":
    unittest.main()
