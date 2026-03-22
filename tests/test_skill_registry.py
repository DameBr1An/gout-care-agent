from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent.skill_registry import load_skill_registry


class SkillRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_skill_registry(PROJECT_ROOT / "skills")

    def test_profile_skill_is_loaded(self) -> None:
        skill = self.registry.get_by_route("profile")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertIn("获取用户档案", skill.recommended_tools)
        self.assertIn("更新用户档案", skill.recommended_tools)

    def test_reporting_skill_extracts_execution_tools(self) -> None:
        skill = self.registry.get_by_route("reporting")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.execution_tools[:3], ["生成周报", "生成月报", "导出报告"])

    def test_match_question_prefers_profile_skill(self) -> None:
        match = self.registry.match_question("帮我修改目标尿酸和 AI 管理意见")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["route_name"], "profile")

    def test_risk_skill_has_decision_prompt(self) -> None:
        skill = self.registry.get_by_route("risk_assessment")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertIn("下一步决策", skill.decision_prompt)


if __name__ == "__main__":
    unittest.main()
