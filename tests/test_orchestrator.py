from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent.risk import RiskResult
from gout_agent.skills.orchestrator import AppOrchestrator


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = PROJECT_ROOT
        self.temp_root = PROJECT_ROOT / "tests_tmp" / ("orchestrator_" + uuid.uuid4().hex)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _build_orchestrator(self) -> AppOrchestrator:
        shutil.copytree(self.repo_root / "skills", self.temp_root / "skills")
        return AppOrchestrator(self.temp_root)

    def test_profile_update_uses_allowed_tools(self) -> None:
        orchestrator = self._build_orchestrator()
        self.assertTrue(orchestrator.is_tool_allowed("profile", "获取用户档案"))
        self.assertTrue(orchestrator.is_tool_allowed("profile", "更新用户档案"))

    def test_langgraph_runtime_is_initialized(self) -> None:
        orchestrator = self._build_orchestrator()
        self.assertIsNotNone(orchestrator.agent_graph)
        self.assertIsNotNone(orchestrator.intake_graph)
        self.assertIsNotNone(orchestrator.profile_graph)
        self.assertIsNotNone(orchestrator.reporting_graph)
        self.assertIsNotNone(orchestrator.medication_graph)
        self.assertIsNotNone(orchestrator.risk_graph)
        self.assertIsNotNone(orchestrator.lifestyle_graph)

    def test_route_specific_graphs_are_selected(self) -> None:
        orchestrator = self._build_orchestrator()
        self.assertIs(orchestrator._select_graph("orchestrator", dry_run=False), orchestrator.agent_graph)
        self.assertIs(orchestrator._select_graph("orchestrator", dry_run=True), orchestrator.preview_graph)
        self.assertIs(orchestrator._select_graph("intake", dry_run=False), orchestrator.intake_graph)
        self.assertIs(orchestrator._select_graph("intake", dry_run=True), orchestrator.preview_intake_graph)
        self.assertIs(orchestrator._select_graph("profile", dry_run=False), orchestrator.profile_graph)
        self.assertIs(orchestrator._select_graph("profile", dry_run=True), orchestrator.preview_profile_graph)
        self.assertIs(orchestrator._select_graph("reporting", dry_run=False), orchestrator.reporting_graph)
        self.assertIs(orchestrator._select_graph("reporting", dry_run=True), orchestrator.preview_reporting_graph)
        self.assertIs(orchestrator._select_graph("medication_followup", dry_run=False), orchestrator.medication_graph)
        self.assertIs(orchestrator._select_graph("medication_followup", dry_run=True), orchestrator.preview_medication_graph)
        self.assertIs(orchestrator._select_graph("risk_assessment", dry_run=False), orchestrator.risk_graph)
        self.assertIs(orchestrator._select_graph("risk_assessment", dry_run=True), orchestrator.preview_risk_graph)
        self.assertIs(orchestrator._select_graph("lifestyle_coach", dry_run=False), orchestrator.lifestyle_graph)
        self.assertIs(orchestrator._select_graph("lifestyle_coach", dry_run=True), orchestrator.preview_lifestyle_graph)

    def test_reporting_plan_comes_from_skill_steps(self) -> None:
        orchestrator = self._build_orchestrator()
        plan = orchestrator.build_execution_plan("reporting", intent="export", period_type="monthly")
        self.assertEqual(plan, ["生成月报", "导出报告", "保存报告"])

    def test_load_context_contains_memory(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        self.assertIn("user_preferences", context.long_term_memory)
        self.assertIn("ai_advice_summary", context.long_term_memory)
        self.assertIn("attack_patterns", context.long_term_memory)
        self.assertIn("90d", context.long_term_memory["behavior_portraits"])

    def test_profile_route_can_answer_read_request(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        result = orchestrator.answer_coach_question("帮我看看当前健康档案", context)
        self.assertEqual(result["skill"], "profile")
        self.assertIn("获取用户档案", result["route_meta"]["allowed_tools"])

    def test_agent_loop_executes_multiple_risk_tools(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        result = orchestrator.run_agent_loop("最近为什么风险升高了？", context)
        self.assertEqual(result["skill"], "risk_assessment")
        self.assertGreaterEqual(len(result["agent_loop"]["completed_tools"]), 2)
        self.assertIn("计算痛风风险", result["agent_loop"]["completed_tools"])

    def test_agent_loop_returns_step_trace(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        result = orchestrator.run_agent_loop("帮我看看当前健康档案", context)
        self.assertTrue(result["agent_loop"]["steps"])
        self.assertEqual(result["agent_loop"]["steps"][-1]["action"], "finish")

    def test_run_agent_loop_persists_session_memories(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        orchestrator.run_agent_loop("帮我看看当前健康档案", context)
        refreshed = orchestrator.load_context()
        self.assertGreaterEqual(len(refreshed.session_memories), 2)
        self.assertEqual(refreshed.session_memories[-2]["role"], "user")
        self.assertEqual(refreshed.session_memories[-1]["role"], "assistant")

    def test_preview_agent_loop_does_not_execute_tools(self) -> None:
        orchestrator = self._build_orchestrator()
        preview = orchestrator.preview_agent_loop("帮我看看当前健康档案", max_steps=5)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["skill"], "profile")
        self.assertTrue(preview["agent_loop"]["planned_tools"])
        self.assertFalse(preview["agent_loop"]["completed_tools"])
        self.assertTrue(preview["agent_loop"]["steps"])
        self.assertEqual(preview["agent_loop"]["steps"][0]["action"], "preview_call_tool")

    def test_preview_agent_loop_contains_decision_path(self) -> None:
        orchestrator = self._build_orchestrator()
        preview = orchestrator.preview_agent_loop("最近为什么风险升高了？", max_steps=5)
        decide_steps = [step for step in preview["agent_loop"]["steps"] if step["action"] == "decide"]
        self.assertTrue(decide_steps)
        self.assertEqual(decide_steps[0]["status"], "preview")
        self.assertEqual(decide_steps[0]["decision"]["source"], "dry_run_preview")
        self.assertIn("candidate_tools", decide_steps[0]["decision"])

    def test_replan_can_add_trend_tool_from_observation(self) -> None:
        orchestrator = self._build_orchestrator()
        risk_result = RiskResult("High", "High", 9, [], [], "up", "高风险", "多喝水", "清淡饮食", "减少剧烈运动", "优先避免诱因")
        replanned = orchestrator._replan_after_observation(
            route_name="risk_assessment",
            question="为什么最近风险升高了",
            remaining_tools=[],
            completed_tools=["计算痛风风险", "识别异常指标"],
            observations={"计算痛风风险": risk_result, "识别异常指标": [object(), object()]},
            period_type="weekly",
            intent="review",
        )
        self.assertIn("预测发作趋势", replanned)

    def test_model_can_choose_next_tool_within_allowed_tools(self) -> None:
        orchestrator = self._build_orchestrator()

        class FakeLLMResult:
            ok = True
            content = '{"continue": true, "next_tool": "识别异常指标", "confidence": 0.86, "reason": "先看异常指标", "refusal_reason": null}'
            used_model = "fake"

        original_call = orchestrator.registry.call

        def fake_call(name, *args, **kwargs):
            if name == "调用本地痛风模型":
                return FakeLLMResult()
            return original_call(name, *args, **kwargs)

        orchestrator.registry.call = fake_call
        decision = orchestrator._decide_next_tools_with_model(
            route_name="risk_assessment",
            question="为什么最近风险升高了",
            candidate_tools=["识别痛风诱因", "识别异常指标"],
            completed_tools=["计算痛风风险"],
            observations={},
            intent="review",
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision["plan"], ["识别异常指标", "识别痛风诱因"])
        self.assertEqual(decision["decision"]["confidence"], 0.86)

    def test_decide_step_contains_confidence_and_reason(self) -> None:
        orchestrator = self._build_orchestrator()
        context = orchestrator.load_context()
        result = orchestrator.run_agent_loop("最近为什么风险升高了？", context)
        decide_steps = [step for step in result["agent_loop"]["steps"] if step["action"] == "decide"]
        self.assertTrue(decide_steps)
        self.assertIn("confidence", decide_steps[0]["decision"])
        self.assertIn("reason", decide_steps[0]["decision"])


if __name__ == "__main__":
    unittest.main()
