from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from gout_agent import data, llm, memory
from gout_agent.skill_registry import SkillRegistry, load_skill_registry
from gout_agent.skills._runtime_loader import load_runtime_module
from gout_agent.toolkit import ToolRegistry, build_default_tool_registry, serialize_tool_result

RISK_LABELS = {"Low": "低", "Moderate": "中", "High": "高", "up": "上升", "down": "下降", "stable": "稳定"}
TRIGGER_LABELS = {"alcohol": "饮酒", "beer": "啤酒", "spirits": "烈酒", "seafood": "海鲜", "shellfish": "贝类", "organ_meat": "动物内脏", "red_meat": "红肉", "hotpot": "火锅", "barbecue": "烧烤", "sugary_drinks": "含糖饮料", "low_hydration": "饮水不足", "missed_medication": "未按时服药"}
ROUTE_FALLBACKS = {"profile": ["档案", "资料", "基本信息", "目标尿酸", "AI建议", "AI 管理意见", "基础病"], "risk_assessment": ["尿酸", "风险", "发作", "诱因", "异常", "趋势"], "lifestyle_coach": ["吃", "喝", "饮食", "食物", "运动", "喝水", "啤酒", "海鲜"], "medication_followup": ["药", "服药", "提醒", "复查", "依从性"], "reporting": ["周报", "月报", "报告", "导出"], "intake": ["记录", "录入", "今天", "昨晚", "刚刚"]}

@dataclass
class AppContext:
    user_journal: dict[str, Any]
    site_history: pd.DataFrame
    risk_overview: dict[str, Any]
    profile: dict[str, Any]
    logs: pd.DataFrame
    symptom_logs: pd.DataFrame
    labs: pd.DataFrame
    attacks: pd.DataFrame
    medications: pd.DataFrame
    reminders: pd.DataFrame
    risk_result: Any
    trigger_summary: list[dict[str, Any]]
    abnormal_items: list[str]
    medication_completion_rate: float | None
    llm_status: dict[str, Any]
    long_term_memory: dict[str, Any]
    session_memories: list[dict[str, Any]]

@dataclass
class AgentLoopStep:
    index: int
    thought: str
    action: str
    tool_name: str | None
    observation: Any | None
    status: str
    decision: dict[str, Any] | None = None


class AgentGraphState(TypedDict):
    question: str
    context: AppContext
    route_name: str
    route_meta: dict[str, Any]
    intent: str
    period_type: str
    planned_tools: list[str]
    remaining_tools: list[str]
    observations: dict[str, Any]
    completed_tools: list[str]
    steps: list[dict[str, Any]]
    max_steps: int
    answer: str | None
    source: str | None
    model: str | None
    error: str | None
    dry_run: bool

class AppOrchestrator:
    def __init__(self, project_root: Path, user_id: int = data.DEFAULT_USER_ID) -> None:
        self.project_root = project_root
        self.user_id = user_id
        self.registry: ToolRegistry = build_default_tool_registry(project_root, user_id=user_id)
        self.skill_registry: SkillRegistry = load_skill_registry(project_root / "skills")
        self.intake_runtime = load_runtime_module("intake-skill")
        self.lifestyle_runtime = load_runtime_module("lifestyle-coach-skill")
        self.medication_runtime = load_runtime_module("medication-followup-skill")
        self.reporting_runtime = load_runtime_module("report-explanation-skill")
        self.lab_report_runtime = load_runtime_module("lab-report-skill")
        self.risk_runtime = load_runtime_module("risk-assessment-skill")
        self.agent_graph = self._build_agent_graph()
        self.preview_graph = self._build_agent_graph(dry_run=True)
        self.intake_graph = self._build_intake_graph()
        self.preview_intake_graph = self._build_intake_graph(dry_run=True)
        self.profile_graph = self._build_profile_graph()
        self.preview_profile_graph = self._build_profile_graph(dry_run=True)
        self.reporting_graph = self._build_reporting_graph()
        self.preview_reporting_graph = self._build_reporting_graph(dry_run=True)
        self.medication_graph = self._build_medication_graph()
        self.preview_medication_graph = self._build_medication_graph(dry_run=True)
        self.risk_graph = self._build_risk_graph()
        self.preview_risk_graph = self._build_risk_graph(dry_run=True)
        self.lifestyle_graph = self._build_lifestyle_graph()
        self.preview_lifestyle_graph = self._build_lifestyle_graph(dry_run=True)

    def load_context(self) -> AppContext:
        trace = self._trace_context("orchestrator", source="context_loader")
        profile = self.registry.call("获取用户档案", _trace_context=trace)
        logs = self.registry.call("获取近期健康记录", 90, _trace_context=trace)
        symptom_logs = self.registry.call("获取部位症状历史", 90, _trace_context=trace)
        labs = self.registry.call("获取化验历史", _trace_context=trace)
        attacks = self.registry.call("获取发作历史", 365, _trace_context=trace)
        medications = self.registry.call("获取药物列表", _trace_context=trace)
        reminders = self.registry.call("获取启用提醒", _trace_context=trace)
        risk_result = self.registry.call("计算痛风风险", profile, logs, labs, attacks, _trace_context=trace)
        trigger_counts = self.registry.call("识别痛风诱因", logs, 14, _trace_context=trace)
        abnormal_metrics = self.registry.call("识别异常指标", profile, labs.iloc[-1].to_dict() if not labs.empty else None, logs.iloc[-1].to_dict() if not logs.empty else None, _trace_context=trace)
        long_term_memory = memory.build_long_term_memory(profile, logs, labs, attacks, symptom_logs=symptom_logs)
        self._sync_long_term_memory(long_term_memory)
        self._sync_digital_twin_profile(long_term_memory.get("gout_management_twin_profile"))
        session_memories = self._load_session_memories(limit=12)
        trigger_summary = [{"name": k, "label": TRIGGER_LABELS.get(k, k), "count": v} for k, v in list(trigger_counts.items())[:5]]
        user_journal = self._build_user_journal(profile, logs)
        site_history = self._build_site_history(symptom_logs, attacks)
        risk_overview = self._build_risk_overview(risk_result, trigger_summary, [item.message for item in abnormal_metrics])
        return AppContext(user_journal=user_journal, site_history=site_history, risk_overview=risk_overview, profile=profile, logs=logs, symptom_logs=symptom_logs, labs=labs, attacks=attacks, medications=medications, reminders=reminders, risk_result=risk_result, trigger_summary=trigger_summary, abnormal_items=[item.message for item in abnormal_metrics], medication_completion_rate=self._medication_completion_rate(logs.tail(7)), llm_status=self.registry.call("获取本地模型状态", _trace_context=trace), long_term_memory=long_term_memory, session_memories=session_memories)

    def get_ui_snapshot(self, context: AppContext) -> dict[str, Any]:
        return {"latest_uric_acid": None, "attack_risk_label": context.risk_overview.get("attack_risk_label"), "uric_acid_risk_label": context.risk_overview.get("uric_acid_risk_label"), "overall_risk_score": context.risk_overview.get("overall_risk_score"), "explanation": context.risk_overview.get("explanation"), "hydration_advice": context.risk_overview.get("hydration_advice"), "diet_advice": context.risk_overview.get("diet_advice"), "exercise_advice": context.risk_overview.get("exercise_advice"), "behavior_goal": context.risk_overview.get("behavior_goal"), "abnormal_items": context.risk_overview.get("abnormal_items", []), "trigger_summary": context.risk_overview.get("trigger_summary", []), "medication_completion_rate": context.medication_completion_rate, "active_reminder_count": len(context.reminders), "active_medication_count": len(context.medications.loc[context.medications["active_flag"] == 1]) if not context.medications.empty and "active_flag" in context.medications.columns else 0, "llm_status": context.llm_status}

    def answer_coach_question(self, question: str, context: AppContext) -> dict[str, Any]:
        return self.run_agent_loop(question, context)

    def preview_agent_loop(self, question: str, max_steps: int = 8) -> dict[str, Any]:
        question = question.strip()
        route_name, route_meta = self._route_question(question)
        intent = self._infer_loop_intent(question, route_name)
        period_type = self._infer_period_type(question)
        planned_tools = self.build_execution_plan(route_name, intent=intent, period_type=period_type)[:max_steps]
        preview_context = self.load_context()
        state: AgentGraphState = {
            "question": question,
            "context": preview_context,
            "route_name": route_name,
            "route_meta": route_meta,
            "intent": intent,
            "period_type": period_type,
            "planned_tools": planned_tools,
            "remaining_tools": list(planned_tools),
            "observations": {},
            "completed_tools": [],
            "steps": [],
            "max_steps": max_steps,
            "answer": None,
            "source": None,
            "model": None,
            "error": None,
            "dry_run": True,
        }
        final_state = self._select_graph(route_name, dry_run=True).invoke(state)
        preview_steps = list(final_state["steps"])
        preview_steps.append(
            asdict(
                AgentLoopStep(
                    index=len(preview_steps) + 1,
                    thought="这是 dry-run 预演，不会真的执行工具，也不会生成最终回答。",
                    action="finish",
                    tool_name=None,
                    observation={"source": "dry_run"},
                    status="preview",
                )
            )
        )
        return {
            "skill": route_name,
            "source": "dry_run",
            "answer": None,
            "model": None,
            "error": None,
            "route_meta": route_meta,
            "dry_run": True,
            "agent_loop": {
                "intent": intent,
                "planned_tools": planned_tools,
                "completed_tools": final_state["completed_tools"],
                "steps": preview_steps,
                "observations": {},
            },
        }

    def run_agent_loop(self, question: str, context: AppContext, max_steps: int = 8) -> dict[str, Any]:
        question = question.strip()
        self._save_session_memory("user", question, {"source": "agent_loop"})
        route_name, route_meta = self._route_question(question)
        intent = self._infer_loop_intent(question, route_name)
        period_type = self._infer_period_type(question)
        planned_tools = self.build_execution_plan(route_name, intent=intent, period_type=period_type)
        state: AgentGraphState = {
            "question": question,
            "context": context,
            "route_name": route_name,
            "route_meta": route_meta,
            "intent": intent,
            "period_type": period_type,
            "planned_tools": planned_tools,
            "remaining_tools": list(planned_tools),
            "observations": {},
            "completed_tools": [],
            "steps": [],
            "max_steps": max_steps,
            "answer": None,
            "source": None,
            "model": None,
            "error": None,
            "dry_run": False,
        }
        final_state = self._select_graph(route_name, dry_run=False).invoke(state)
        observations = final_state["observations"]
        completed_tools = final_state["completed_tools"]
        steps = [AgentLoopStep(**step) for step in final_state["steps"]]
        llm_context = self._build_llm_context(context)
        llm_context.update({"selected_skill": route_name, "route_meta": route_meta, "allowed_tools": route_meta["allowed_tools"], "execution_steps": self.skill_registry.get_execution_steps(route_name), "execution_plan": completed_tools, "agent_loop": {"intent": intent, "completed_tools": completed_tools, "steps": [asdict(step) for step in steps], "observations": {name: serialize_tool_result(value) for name, value in observations.items()}}})
        llm_result = self.registry.call("调用本地痛风模型", question, llm_context, _trace_context=self._trace_context(route_name, source="agent_loop_llm"))
        if llm_result.ok:
            answer, source, model, error = llm_result.content, "local_llm", llm_result.used_model, None
        else:
            answer, source, model, error = self._fallback_answer(question, route_name, context, observations), "rule_fallback", None, llm_result.error_message
        self._save_session_memory("assistant", answer, {"skill": route_name, "source": source})
        steps.append(AgentLoopStep(index=len(steps) + 1, thought="基于已完成的观察结果组织最终回答。", action="finish", tool_name=None, observation={"source": source}, status="completed"))
        return {"skill": route_name, "source": source, "answer": answer, "model": model, "error": error, "route_meta": route_meta, "agent_loop": {"intent": intent, "planned_tools": planned_tools, "completed_tools": completed_tools, "steps": [asdict(step) for step in steps], "observations": {name: serialize_tool_result(value) for name, value in observations.items()}}}

    def _build_agent_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("run_tool", self._graph_preview_tool)
        else:
            graph.add_node("run_tool", self._graph_run_tool)
        graph.add_node("decide_next", self._graph_decide_next)
        graph.add_edge(START, "run_tool")
        graph.add_conditional_edges("run_tool", self._graph_after_run_tool, {"decide_next": "decide_next", "finish": END})
        graph.add_conditional_edges("decide_next", self._graph_after_decide, {"run_tool": "run_tool", "finish": END})
        return graph.compile()

    def _build_profile_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("profile_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("profile_run_tool", self._graph_run_tool)
        graph.add_edge(START, "profile_preview_tool" if dry_run else "profile_run_tool")
        graph.add_edge("profile_preview_tool" if dry_run else "profile_run_tool", END)
        return graph.compile()

    def _build_reporting_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("reporting_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("reporting_run_tool", self._graph_run_tool)
        graph.add_node("reporting_decide_next", self._graph_decide_next)
        graph.add_edge(START, "reporting_preview_tool" if dry_run else "reporting_run_tool")
        graph.add_conditional_edges(
            "reporting_preview_tool" if dry_run else "reporting_run_tool",
            self._graph_after_run_tool,
            {"decide_next": "reporting_decide_next", "finish": END},
        )
        graph.add_conditional_edges("reporting_decide_next", self._graph_after_decide, {"run_tool": "reporting_preview_tool" if dry_run else "reporting_run_tool", "finish": END})
        return graph.compile()

    def _build_medication_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("medication_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("medication_run_tool", self._graph_run_tool)
        graph.add_node("medication_decide_next", self._graph_decide_next)
        graph.add_edge(START, "medication_preview_tool" if dry_run else "medication_run_tool")
        graph.add_conditional_edges(
            "medication_preview_tool" if dry_run else "medication_run_tool",
            self._graph_after_run_tool,
            {"decide_next": "medication_decide_next", "finish": END},
        )
        graph.add_conditional_edges(
            "medication_decide_next",
            self._graph_after_decide,
            {"run_tool": "medication_preview_tool" if dry_run else "medication_run_tool", "finish": END},
        )
        return graph.compile()

    def _build_risk_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("risk_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("risk_run_tool", self._graph_run_tool)
        graph.add_node("risk_decide_next", self._graph_decide_next)
        graph.add_edge(START, "risk_preview_tool" if dry_run else "risk_run_tool")
        graph.add_conditional_edges(
            "risk_preview_tool" if dry_run else "risk_run_tool",
            self._graph_after_run_tool,
            {"decide_next": "risk_decide_next", "finish": END},
        )
        graph.add_conditional_edges(
            "risk_decide_next",
            self._graph_after_decide,
            {"run_tool": "risk_preview_tool" if dry_run else "risk_run_tool", "finish": END},
        )
        return graph.compile()

    def _build_lifestyle_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("lifestyle_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("lifestyle_run_tool", self._graph_run_tool)
        graph.add_node("lifestyle_decide_next", self._graph_decide_next)
        graph.add_edge(START, "lifestyle_preview_tool" if dry_run else "lifestyle_run_tool")
        graph.add_conditional_edges(
            "lifestyle_preview_tool" if dry_run else "lifestyle_run_tool",
            self._graph_after_run_tool,
            {"decide_next": "lifestyle_decide_next", "finish": END},
        )
        graph.add_conditional_edges(
            "lifestyle_decide_next",
            self._graph_after_decide,
            {"run_tool": "lifestyle_preview_tool" if dry_run else "lifestyle_run_tool", "finish": END},
        )
        return graph.compile()

    def _build_intake_graph(self, dry_run: bool = False):
        graph = StateGraph(AgentGraphState)
        if dry_run:
            graph.add_node("intake_preview_tool", self._graph_preview_tool)
        else:
            graph.add_node("intake_run_tool", self._graph_run_tool)
        graph.add_node("intake_decide_next", self._graph_decide_next)
        graph.add_edge(START, "intake_preview_tool" if dry_run else "intake_run_tool")
        graph.add_conditional_edges(
            "intake_preview_tool" if dry_run else "intake_run_tool",
            self._graph_after_run_tool,
            {"decide_next": "intake_decide_next", "finish": END},
        )
        graph.add_conditional_edges(
            "intake_decide_next",
            self._graph_after_decide,
            {"run_tool": "intake_preview_tool" if dry_run else "intake_run_tool", "finish": END},
        )
        return graph.compile()

    def _select_graph(self, route_name: str, dry_run: bool):
        if route_name == "intake":
            return self.preview_intake_graph if dry_run else self.intake_graph
        if route_name == "profile":
            return self.preview_profile_graph if dry_run else self.profile_graph
        if route_name == "reporting":
            return self.preview_reporting_graph if dry_run else self.reporting_graph
        if route_name == "medication_followup":
            return self.preview_medication_graph if dry_run else self.medication_graph
        if route_name == "risk_assessment":
            return self.preview_risk_graph if dry_run else self.risk_graph
        if route_name == "lifestyle_coach":
            return self.preview_lifestyle_graph if dry_run else self.lifestyle_graph
        return self.preview_graph if dry_run else self.agent_graph

    def _graph_run_tool(self, state: AgentGraphState) -> dict[str, Any]:
        remaining_tools = list(state["remaining_tools"])
        if not remaining_tools or len(state["completed_tools"]) >= state["max_steps"]:
            return {"remaining_tools": remaining_tools}

        tool_name = remaining_tools.pop(0)
        observation = self._execute_loop_tool(state["route_name"], tool_name, state["context"], state["observations"])
        observations = dict(state["observations"])
        observations[tool_name] = observation
        completed_tools = list(state["completed_tools"]) + [tool_name]
        steps = list(state["steps"])
        steps.append(
            asdict(
                AgentLoopStep(
                    index=len(steps) + 1,
                    thought=self._build_step_thought(state["route_name"], tool_name, state["question"]),
                    action="call_tool",
                    tool_name=tool_name,
                    observation=serialize_tool_result(observation),
                    status="completed",
                )
            )
        )
        return {
            "remaining_tools": remaining_tools,
            "observations": observations,
            "completed_tools": completed_tools,
            "steps": steps,
        }

    def _graph_preview_tool(self, state: AgentGraphState) -> dict[str, Any]:
        remaining_tools = list(state["remaining_tools"])
        if not remaining_tools or len(state["completed_tools"]) >= state["max_steps"]:
            return {"remaining_tools": remaining_tools}

        tool_name = remaining_tools.pop(0)
        steps = list(state["steps"])
        steps.append(
            asdict(
                AgentLoopStep(
                    index=len(steps) + 1,
                    thought=self._build_step_thought(state["route_name"], tool_name, state["question"]),
                    action="preview_call_tool",
                    tool_name=tool_name,
                    observation={"preview": True},
                    status="preview",
                )
            )
        )
        return {
            "remaining_tools": remaining_tools,
            "completed_tools": list(state["completed_tools"]),
            "steps": steps,
        }

    def _graph_decide_next(self, state: AgentGraphState) -> dict[str, Any]:
        if state.get("dry_run"):
            remaining_tools = list(state["remaining_tools"])
            next_tool = remaining_tools[0] if remaining_tools else None
            decision_meta = {
                "source": "dry_run_preview",
                "continue": bool(next_tool),
                "next_tool": next_tool,
                "confidence": 0.35 if next_tool else 0.8,
                "reason": "根据当前 execution plan 预估下一步顺序，未执行任何工具。",
                "refusal_reason": None if next_tool else "计划中的工具已预演完毕。",
                "candidate_tools": list(remaining_tools),
            }
        else:
            remaining_tools, decision_meta = self._decide_next_tools(
                state["route_name"],
                state["question"],
                list(state["remaining_tools"]),
                list(state["completed_tools"]),
                dict(state["observations"]),
                state["period_type"],
                state["intent"],
            )
        steps = list(state["steps"])
        steps.append(
            asdict(
                AgentLoopStep(
                    index=len(steps) + 1,
                    thought="基于当前技能和执行状态，决定是否继续下一步，以及优先调用哪个工具。",
                    action="decide",
                    tool_name=decision_meta.get("next_tool"),
                    observation={"remaining_tools": remaining_tools},
                    status="preview" if state.get("dry_run") else "completed",
                    decision=decision_meta,
                )
            )
        )
        return {"remaining_tools": remaining_tools, "steps": steps}

    def _graph_after_run_tool(self, state: AgentGraphState) -> str:
        if not state["remaining_tools"] or len(state["completed_tools"]) >= state["max_steps"]:
            return "finish"
        return "decide_next"

    def _graph_after_decide(self, state: AgentGraphState) -> str:
        if not state["remaining_tools"] or len(state["completed_tools"]) >= state["max_steps"]:
            return "finish"
        return "run_tool"
    def explain_report(self, period_type: str, context: AppContext) -> dict[str, Any]:
        plan = self.build_execution_plan("reporting", intent="explain", period_type=period_type)
        report_payload = self._execute_reporting_plan(plan, context, "json")["report_payload"]
        llm_context = self._build_interpretation_context(context, selected_report=report_payload, period_type=period_type)
        llm_context.update({"allowed_tools": self.get_allowed_tools("reporting"), "execution_plan": plan})
        llm_result = self.registry.call("调用本地痛风模型", "请结合这份周期报告、当前健康分身、近期 7/30/90 天行为变化、历史报告摘要和当前风险，用简洁中文解读本期情况，说明现在最重要的变化、主要诱因、今天的重点行动，以及什么情况下要及时就医。", llm_context, _trace_context=self._trace_context("reporting", source="llm_report"))
        return {"report": report_payload, "explanation": llm_result.content if llm_result.ok else self.reporting_runtime.explain_report(report_payload, self.serialize_context(context)), "source": "local_llm" if llm_result.ok else "rule_fallback", "error": None if llm_result.ok else llm_result.error_message}

    def explain_uploaded_lab_reports(self, uploaded_files: list[dict[str, Any]], context: AppContext) -> dict[str, Any]:
        parsed_lab_reports = self.lab_report_runtime.parse_uploaded_lab_files(uploaded_files, self._extract_lab_metrics_with_local_model)
        llm_context = self._build_interpretation_context(context, uploaded_lab_reports=uploaded_files, parsed_lab_reports=parsed_lab_reports)
        llm_result = self.registry.call(
            "调用本地痛风模型",
            "请结合当前健康分身、近期 7/30/90 天行为变化、历史周报/月报摘要，以及新上传化验报告中已识别出的指标值，用简洁中文说明现在最该关注什么、哪些变化值得警惕、接下来建议补充哪些记录或线下检查。如果未成功识别出指标值，请明确按文件信息和当前记录保守解释。",
            llm_context,
            _trace_context=self._trace_context("reporting", source="llm_lab_report"),
        )
        if llm_result.ok:
            return {"answer": llm_result.content, "source": "local_llm", "error": None}
        fallback = self._fallback_answer("请结合当前记录解读我上传的化验报告，并给出后续建议。", "reporting", context)
        return {"answer": fallback, "source": "rule_fallback", "error": llm_result.error_message}

    def export_report(self, period_type: str, format_name: str, context: AppContext) -> Path:
        plan = self.build_execution_plan("reporting", intent="export", period_type=period_type)
        return self._execute_reporting_plan(plan, context, format_name)["path"]

    def parse_intake_text(self, text: str) -> dict[str, Any]: return self.intake_runtime.parse_free_text_entry(text)
    def describe_skills(self) -> list[dict[str, Any]]: return self.skill_registry.describe()
    def describe_tools(self, include_schema: bool = False) -> list[dict[str, Any]]: return self.registry.describe(include_schema=include_schema)
    def get_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]: return self.registry.get_traces(limit=limit)
    def get_allowed_tools(self, route_name: str) -> list[str]: return list(self.skill_registry.get_allowed_tools(route_name))
    def is_tool_allowed(self, route_name: str, tool_name: str) -> bool:
        allowed_tools = self.get_allowed_tools(route_name)
        return route_name == "orchestrator" if not allowed_tools else tool_name in allowed_tools

    def build_execution_plan(self, route_name: str, intent: str | None = None, period_type: str = "weekly") -> list[str]:
        execution_tools = self.skill_registry.get_execution_tools(route_name) or self.get_allowed_tools(route_name)
        allowed_tools = self.get_allowed_tools(route_name)
        if route_name == "reporting":
            plan = ["生成月报" if period_type == "monthly" else "生成周报"]
            if intent == "export":
                for tool_name in execution_tools:
                    if tool_name in {"导出报告", "保存报告"} and tool_name not in plan:
                        plan.append(tool_name)
            return plan
        if route_name == "profile":
            target_tools = ["获取用户档案", "更新用户档案"] if intent == "update" else ["获取用户档案"]
            return [tool for tool in target_tools if tool in allowed_tools]
        if route_name == "risk_assessment":
            plan = [tool for tool in execution_tools if tool in allowed_tools]
            return plan if intent == "trend" else [tool for tool in plan if tool != "预测发作趋势"]
        if route_name == "medication_followup":
            plan = [tool for tool in ["获取药物列表", "获取服药依从性", "获取启用提醒"] if tool in allowed_tools]
            if intent == "add_medication" and "添加药物方案" in allowed_tools: plan.append("添加药物方案")
            if intent == "create_reminder" and "创建提醒" in allowed_tools: plan.append("创建提醒")
            if intent == "log_medication" and "记录服药情况" in allowed_tools: plan.append("记录服药情况")
            return plan
        return [tool for tool in execution_tools if tool in allowed_tools]

    def save_daily_log(self, payload: dict[str, Any]) -> Any: return self._call_skill_tool("intake", "记录日常健康", payload)
    def save_joint_symptom(self, payload: dict[str, Any]) -> Any: return self._call_skill_tool("intake", "记录部位症状", payload)
    def save_lab_result(self, payload: dict[str, Any]) -> Any: return self._call_skill_tool("intake", "记录化验结果", payload)
    def update_profile(self, payload: dict[str, Any]) -> Any:
        if "获取用户档案" in self.build_execution_plan("profile", intent="update"): self._call_skill_tool("profile", "获取用户档案")
        return self._call_skill_tool("profile", "更新用户档案", payload)
    def get_profile(self) -> Any: return self._call_skill_tool("profile", (self.build_execution_plan("profile", intent="read") or ["获取用户档案"])[0])
    def save_attack(self, payload: dict[str, Any]) -> Any: return self._call_skill_tool("intake", "记录痛风发作", payload)
    def add_medication(self, payload: dict[str, Any]) -> Any: return self._call_skill_tool("medication_followup", "添加药物方案", payload)
    def create_reminder(self, reminder_type: str, title: str, schedule_rule: str, next_trigger_at: str) -> Any: return self._call_skill_tool("medication_followup", "创建提醒", reminder_type, title, schedule_rule, next_trigger_at)
    def log_medication_taken(self, medication_id: int, status: str, taken_time: str | None = None) -> Any: return self._call_skill_tool("medication_followup", "记录服药情况", medication_id, status, None, taken_time)

    def sync_daily_snapshot(self, context: AppContext) -> None:
        existing = self.registry.call("获取风险快照", 2, _trace_context=self._trace_context("orchestrator", source="snapshot"))
        today = pd.Timestamp.today().date().isoformat()
        if not existing.empty and today in existing["snapshot_date"].astype(str).tolist(): return
        self.registry.call("保存风险快照", {"snapshot_date": today, "uric_acid_risk_level": context.risk_result.uric_acid_risk_level, "attack_risk_level": context.risk_result.attack_risk_level, "overall_risk_score": context.risk_result.overall_risk_score, "top_risk_factors": context.risk_result.top_risk_factors, "trend_direction": context.risk_result.trend_direction}, _trace_context=self._trace_context("orchestrator", source="snapshot"))

    def serialize_context(self, context: AppContext) -> dict[str, Any]:
        return {"user_journal": context.user_journal, "site_history": context.site_history.tail(20).to_dict(orient="records") if not context.site_history.empty else [], "risk_overview": context.risk_overview, "risk_result": {"uric_acid_risk_level": context.risk_result.uric_acid_risk_level, "attack_risk_level": context.risk_result.attack_risk_level, "uric_acid_risk_level_cn": self.label_risk(context.risk_result.uric_acid_risk_level), "attack_risk_level_cn": self.label_risk(context.risk_result.attack_risk_level), "overall_risk_score": context.risk_result.overall_risk_score, "explanation": context.risk_result.explanation, "hydration_advice": context.risk_result.hydration_advice, "diet_advice": context.risk_result.diet_advice, "exercise_advice": context.risk_result.exercise_advice, "behavior_goal": context.risk_result.behavior_goal}, "trigger_summary": context.trigger_summary, "abnormal_items": context.abnormal_items, "medication_completion_rate": context.medication_completion_rate, "active_reminder_count": len(context.reminders), "long_term_memory": context.long_term_memory, "session_memories": context.session_memories, "recent_symptom_logs": context.symptom_logs.tail(20).to_dict(orient="records") if not context.symptom_logs.empty else []}

    def _build_llm_context(self, context: AppContext) -> dict[str, Any]:
        payload = self.serialize_context(context)
        payload.update({"user_profile": context.user_journal.get("profile", {}), "recent_health_records": context.user_journal.get("recent_health_records", []), "site_history_preview": context.site_history.head(10).to_dict(orient="records") if not context.site_history.empty else [], "latest_daily_log": context.logs.iloc[-1].to_dict() if not context.logs.empty else {}, "recent_symptom_logs": context.symptom_logs.head(10).to_dict(orient="records") if not context.symptom_logs.empty else [], "recent_attack_records": context.attacks.head(5).to_dict(orient="records") if not context.attacks.empty else [], "behavior_portraits": context.long_term_memory.get("behavior_portraits"), "digital_twin_profile": context.long_term_memory.get("gout_management_twin_profile"), "recent_session_memories": context.session_memories[-6:]})
        return payload

    def _build_interpretation_context(
        self,
        context: AppContext,
        selected_report: dict[str, Any] | None = None,
        period_type: str | None = None,
        uploaded_lab_reports: list[dict[str, Any]] | None = None,
        parsed_lab_reports: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_llm_context(context)
        report_history = self.registry.call("获取报告历史", _trace_context=self._trace_context("reporting", source="report_history"))
        history_summaries: list[dict[str, Any]] = []
        if report_history is not None and not report_history.empty:
            for _, row in report_history.head(4).iterrows():
                report_payload = {}
                raw = row.get("report_json")
                if isinstance(raw, str) and raw.strip():
                    try:
                        report_payload = json.loads(raw)
                    except json.JSONDecodeError:
                        report_payload = {}
                history_summaries.append(
                    {
                        "report_type": row.get("report_type"),
                        "period_start": row.get("period_start"),
                        "period_end": row.get("period_end"),
                        "summary": report_payload.get("executive_summary") or report_payload.get("summary"),
                        "action_plan": report_payload.get("action_plan") or [],
                    }
                )
        payload.update(
            {
                "selected_report": selected_report or {},
                "report_period_type": period_type,
                "report_history_summaries": history_summaries,
                "current_risk_overview": context.risk_overview,
                "uploaded_lab_reports": uploaded_lab_reports or [],
                "parsed_lab_reports": parsed_lab_reports or {},
            }
        )
        return payload

    def _extract_lab_metrics_with_local_model(self, uploaded_files: list[dict[str, Any]]) -> dict[str, Any]:
        return llm.ask_local_lab_vision_llm(uploaded_files)

    def _build_user_journal(self, profile: dict[str, Any], logs: pd.DataFrame) -> dict[str, Any]:
        recent_records = logs.tail(20).to_dict(orient="records") if not logs.empty else []
        latest_record = logs.iloc[-1].to_dict() if not logs.empty else {}
        return {"profile": serialize_tool_result(profile), "recent_health_records": serialize_tool_result(recent_records), "latest_record": serialize_tool_result(latest_record)}

    def _build_site_history(self, symptom_logs: pd.DataFrame, attacks: pd.DataFrame) -> pd.DataFrame:
        symptom_frame = symptom_logs.copy()
        attack_frame = attacks.copy()
        normalized_frames: list[pd.DataFrame] = []
        if not symptom_frame.empty:
            symptom_frame = symptom_frame.assign(event_type="symptom", event_date=symptom_frame.get("log_date"), site=symptom_frame.get("body_site"), trigger_notes=None, duration_hours=None, resolved_flag=None)
            normalized_frames.append(symptom_frame[["event_type", "event_date", "site", "pain_score", "swelling_flag", "redness_flag", "stiffness_flag", "symptom_notes", "trigger_notes", "duration_hours", "resolved_flag"]].copy())
        if not attack_frame.empty:
            attack_frame = attack_frame.assign(event_type="attack", event_date=attack_frame.get("attack_date"), site=attack_frame.get("joint_site"), stiffness_flag=None, symptom_notes=attack_frame.get("notes"), trigger_notes=attack_frame.get("suspected_trigger"))
            normalized_frames.append(attack_frame[["event_type", "event_date", "site", "pain_score", "swelling_flag", "redness_flag", "stiffness_flag", "symptom_notes", "trigger_notes", "duration_hours", "resolved_flag"]].copy())
        if not normalized_frames:
            return pd.DataFrame(columns=["event_type", "event_date", "site", "pain_score", "swelling_flag", "redness_flag", "stiffness_flag", "symptom_notes", "trigger_notes", "duration_hours", "resolved_flag"])
        history = pd.concat(normalized_frames, ignore_index=True)
        history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
        history = history.sort_values(["event_date", "event_type"], ascending=[False, True]).reset_index(drop=True)
        return history

    def _build_risk_overview(self, risk_result: Any, trigger_summary: list[dict[str, Any]], abnormal_items: list[str]) -> dict[str, Any]:
        return {"uric_acid_risk_label": self.label_risk(risk_result.uric_acid_risk_level), "attack_risk_label": self.label_risk(risk_result.attack_risk_level), "overall_risk_score": risk_result.overall_risk_score, "explanation": risk_result.explanation, "hydration_advice": risk_result.hydration_advice, "diet_advice": risk_result.diet_advice, "exercise_advice": risk_result.exercise_advice, "behavior_goal": risk_result.behavior_goal, "trigger_summary": trigger_summary, "abnormal_items": abnormal_items}

    def _load_session_memories(self, limit: int = 12) -> list[dict[str, Any]]:
        frame = data.get_session_memories(self.project_root, limit=limit, user_id=self.user_id)
        if frame.empty:
            return []
        records = frame.iloc[::-1].to_dict(orient="records")
        return [serialize_tool_result(record) for record in records]

    def _save_session_memory(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not str(content or "").strip():
            return
        data.save_session_memory(self.project_root, role, content, metadata=metadata or {}, user_id=self.user_id)

    def _sync_long_term_memory(self, long_term_memory: dict[str, Any]) -> None:
        latest = data.get_latest_memory_snapshot(self.project_root, "long_term_memory", user_id=self.user_id)
        latest_clean = dict(latest or {})
        latest_clean.pop("updated_at", None)
        current_clean = dict(long_term_memory)
        current_clean.pop("updated_at", None)
        if latest_clean != current_clean:
            data.save_memory_snapshot(self.project_root, "long_term_memory", long_term_memory, user_id=self.user_id)

    def _sync_digital_twin_profile(self, twin_profile: dict[str, Any] | None) -> None:
        if not isinstance(twin_profile, dict) or not twin_profile:
            return
        latest = data.get_latest_digital_twin_profile(self.project_root, user_id=self.user_id) or {}
        latest_clean = dict(latest)
        latest_clean.pop("updated_at", None)
        current_clean = dict(twin_profile)
        current_clean.pop("updated_at", None)
        if latest_clean != current_clean:
            data.save_digital_twin_profile(self.project_root, twin_profile, user_id=self.user_id)

    def _execute_loop_tool(self, route_name: str, tool_name: str, context: AppContext, observations: dict[str, Any]) -> Any:
        if route_name == "profile": return self._call_skill_tool("profile", tool_name)
        if route_name == "reporting": return self._execute_reporting_loop_tool(tool_name, context, observations)
        if tool_name == "计算痛风风险":
            tool_route = "risk_assessment" if route_name == "risk_assessment" else "lifestyle_coach"
            return self._call_skill_tool(tool_route, tool_name, context.profile, context.logs, context.labs, context.attacks)
        if tool_name == "识别痛风诱因":
            tool_route = "risk_assessment" if route_name == "risk_assessment" else "lifestyle_coach"
            return self._call_skill_tool(tool_route, tool_name, context.logs, 14)
        if tool_name == "识别异常指标": return self._call_skill_tool("risk_assessment", tool_name, context.profile, context.labs.iloc[-1].to_dict() if not context.labs.empty else None, context.logs.iloc[-1].to_dict() if not context.logs.empty else None)
        if tool_name == "预测发作趋势": return self._call_skill_tool("risk_assessment", tool_name, context.logs, context.labs, 7)
        if tool_name == "获取药物列表": return self._call_skill_tool(route_name, tool_name)
        if tool_name == "获取服药依从性": return self._call_skill_tool(route_name, tool_name, 30)
        if tool_name == "获取启用提醒": return self._call_skill_tool(route_name, tool_name)
        return None

    def _execute_reporting_loop_tool(self, tool_name: str, context: AppContext, observations: dict[str, Any]) -> Any:
        if tool_name in {"生成周报", "生成月报"}:
            return self._call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks, context.symptom_logs)
        report_payload = observations.get("生成周报") or observations.get("生成月报")
        if report_payload is None: raise RuntimeError("执行报告工具前缺少报告内容。")
        report_type = "monthly" if "生成月报" in observations else "weekly"
        if tool_name == "导出报告": return self._call_skill_tool("reporting", tool_name, report_payload, report_type, "json")
        if tool_name == "保存报告":
            period_start, period_end = report_payload["period"].split(" 至 ")
            return self._call_skill_tool("reporting", tool_name, report_type, report_payload, period_start, period_end)
        return None

    def _replan_after_observation(self, route_name: str, question: str, remaining_tools: list[str], completed_tools: list[str], observations: dict[str, Any], period_type: str, intent: str) -> list[str]:
        replanned = [tool for tool in remaining_tools if tool not in completed_tools]
        if route_name == "risk_assessment":
            if not self._question_mentions_trend(question): replanned = [tool for tool in replanned if tool != "预测发作趋势"]
            risk_payload = observations.get("计算痛风风险")
            abnormal_payload = observations.get("识别异常指标")
            if risk_payload is not None:
                needs_trend = self._question_mentions_trend(question) or getattr(risk_payload, "overall_risk_score", 0) >= 8 or len(abnormal_payload or []) >= 2
                if needs_trend and "预测发作趋势" not in replanned and "预测发作趋势" not in completed_tools: replanned.append("预测发作趋势")
        if route_name == "lifestyle_coach":
            risk_payload = observations.get("计算痛风风险")
            if risk_payload is not None and getattr(risk_payload, "overall_risk_score", 0) < 4: replanned = [tool for tool in replanned if tool != "识别痛风诱因"]
        if route_name == "medication_followup":
            adherence_frame = observations.get("获取服药依从性")
            if isinstance(adherence_frame, pd.DataFrame) and not adherence_frame.empty:
                missed_count = int((adherence_frame["status"] == "missed").sum()) if "status" in adherence_frame.columns else 0
                if missed_count >= 2 and intent == "review" and "创建提醒" in self.get_allowed_tools(route_name) and "创建提醒" not in replanned and "创建提醒" not in completed_tools: replanned.append("创建提醒")
        if route_name == "reporting":
            primary_tool = "生成月报" if period_type == "monthly" else "生成周报"
            if primary_tool not in completed_tools: replanned = [tool for tool in replanned if tool in {"生成周报", "生成月报"}]
        if route_name == "profile" and "获取用户档案" in completed_tools: replanned = [tool for tool in replanned if tool != "获取用户档案"]
        return replanned

    def _build_preview_steps(self, route_name: str, question: str, planned_tools: list[str]) -> list[AgentLoopStep]:
        steps: list[AgentLoopStep] = []
        for tool_name in planned_tools:
            remaining_after = [tool for tool in planned_tools if tool != tool_name and tool not in [step.tool_name for step in steps if step.action == "preview_call_tool"]]
            steps.append(
                AgentLoopStep(
                    index=len(steps) + 1,
                    thought=self._build_step_thought(route_name, tool_name, question),
                    action="preview_call_tool",
                    tool_name=tool_name,
                    observation={"preview": True},
                    status="preview",
                )
            )
            next_tool = remaining_after[0] if remaining_after else None
            steps.append(
                AgentLoopStep(
                    index=len(steps) + 1,
                    thought="基于当前技能和执行计划，预估下一步可能继续的工具。",
                    action="decide",
                    tool_name=next_tool,
                    observation={"remaining_tools": remaining_after},
                    status="preview",
                    decision={
                        "source": "dry_run_preview",
                        "continue": bool(next_tool),
                        "next_tool": next_tool,
                        "confidence": 0.35 if next_tool else 0.8,
                        "reason": "根据当前 execution plan 预估下一步顺序，未执行任何工具。",
                        "refusal_reason": None if next_tool else "计划中的工具已预演完毕。",
                        "candidate_tools": list(remaining_after),
                    },
                )
            )
        steps.append(
            AgentLoopStep(
                index=len(steps) + 1,
                thought="这是 dry-run 预演，不会真的执行工具，也不会生成最终回答。",
                action="finish",
                tool_name=None,
                observation={"source": "dry_run"},
                status="preview",
            )
        )
        return steps

    def _decide_next_tools(self, route_name: str, question: str, remaining_tools: list[str], completed_tools: list[str], observations: dict[str, Any], period_type: str, intent: str) -> tuple[list[str], dict[str, Any]]:
        rule_plan = self._replan_after_observation(route_name, question, remaining_tools, completed_tools, observations, period_type, intent)
        model_plan = self._decide_next_tools_with_model(route_name, question, rule_plan, completed_tools, observations, intent)
        if model_plan is not None:
            return model_plan["plan"], model_plan["decision"]
        next_tool = rule_plan[0] if rule_plan else None
        return rule_plan, {
            "source": "rule_replan",
            "continue": bool(rule_plan),
            "next_tool": next_tool,
            "confidence": 1.0,
            "reason": "使用规则重排后的结果作为下一步计划。",
            "refusal_reason": None,
            "candidate_tools": list(rule_plan),
        }

    def _decide_next_tools_with_model(self, route_name: str, question: str, candidate_tools: list[str], completed_tools: list[str], observations: dict[str, Any], intent: str) -> dict[str, Any] | None:
        if not candidate_tools:
            return {
                "plan": [],
                "decision": {
                    "source": "rule_replan",
                    "continue": False,
                    "next_tool": None,
                    "confidence": 1.0,
                    "reason": "已经没有可继续执行的候选工具。",
                    "refusal_reason": "候选工具列表为空。",
                    "candidate_tools": [],
                },
            }
        llm_context = {"mode": "tool_decision", "route_name": route_name, "intent": intent, "allowed_tools": self.get_allowed_tools(route_name), "candidate_tools": candidate_tools, "completed_tools": completed_tools, "observations": {name: serialize_tool_result(value) for name, value in observations.items()}}
        llm_result = self.registry.call("调用本地痛风模型", self._build_next_tool_decision_prompt(route_name, question, candidate_tools, completed_tools, observations, intent), llm_context, _trace_context=self._trace_context(route_name, source="agent_loop_decision"))
        if not llm_result.ok:
            return None
        decision = self._parse_next_tool_decision(llm_result.content)
        if decision is None:
            return None
        if not decision.get("continue", True):
            decision.update({"source": "local_llm_decision", "next_tool": None, "candidate_tools": list(candidate_tools)})
            return {"plan": [], "decision": decision}
        next_tool = decision.get("next_tool")
        if next_tool not in candidate_tools or not self.is_tool_allowed(route_name, next_tool):
            return None
        decision.update({"source": "local_llm_decision", "candidate_tools": list(candidate_tools)})
        return {"plan": [next_tool] + [tool for tool in candidate_tools if tool != next_tool], "decision": decision}

    def _build_next_tool_decision_prompt(self, route_name: str, question: str, candidate_tools: list[str], completed_tools: list[str], observations: dict[str, Any], intent: str) -> str:
        decision_prompt = self.skill_registry.get_decision_prompt(route_name) or self.skill_registry.get_decision_prompt("orchestrator")
        return (
            (decision_prompt + "\n\n" if decision_prompt else "")
            + "你是一个受约束的工具决策器。"
            + "你只能从 candidate_tools 中选择下一步工具，或决定停止继续。"
            + "绝不能输出 candidate_tools 之外的工具。"
            "请只输出 JSON，不要输出解释。\n\n"
            f"route_name: {route_name}\n"
            f"intent: {intent}\n"
            f"question: {question}\n"
            f"completed_tools: {completed_tools}\n"
            f"candidate_tools: {candidate_tools}\n"
            f"observation_keys: {list(observations.keys())}\n\n"
            '输出格式: {"continue": true, "next_tool": "工具名", "confidence": 0.0, "reason": "一句话原因", "refusal_reason": null}'
        )

    def _parse_next_tool_decision(self, text: str) -> dict[str, Any] | None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            import json
            payload = json.loads(cleaned[start : end + 1])
        except Exception:
            return None
        continue_flag = bool(payload.get("continue", True))
        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))
        return {
            "continue": continue_flag,
            "next_tool": payload.get("next_tool"),
            "confidence": confidence_value,
            "reason": str(payload.get("reason") or ""),
            "refusal_reason": str(payload.get("refusal_reason")) if payload.get("refusal_reason") not in (None, "") else None,
        }

    def _execute_reporting_plan(self, plan: list[str], context: AppContext, format_name: str) -> dict[str, Any]:
        report_payload, path = None, None
        for tool_name in plan:
            if tool_name in {"生成周报", "生成月报"}:
                report_payload = self._call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks, context.symptom_logs)
            elif tool_name == "导出报告":
                if report_payload is None: raise RuntimeError("导出报告前缺少报告内容。")
                path = self._call_skill_tool("reporting", tool_name, report_payload, "monthly" if "月报" in plan[0] else "weekly", format_name)
            elif tool_name == "保存报告":
                if report_payload is None: raise RuntimeError("保存报告前缺少报告内容。")
                period_start, period_end = report_payload["period"].split(" 至 ")
                self._call_skill_tool("reporting", tool_name, "monthly" if "月报" in plan[0] else "weekly", report_payload, period_start, period_end)
        if report_payload is None: raise RuntimeError("未能生成报告。")
        return {"report_payload": report_payload, "path": path}

    def _call_skill_tool(self, route_name: str, tool_name: str, *args, **kwargs) -> Any:
        if not self.is_tool_allowed(route_name, tool_name): raise PermissionError("技能 %s 不允许调用工具 %s；允许工具为：%s" % (route_name, tool_name, "、".join(self.get_allowed_tools(route_name)) or "无"))
        return self.registry.call(tool_name, *args, _trace_context=self._trace_context(route_name), **kwargs)

    def _trace_context(self, route_name: str, source: str = "skill_tool") -> dict[str, str]:
        skill = self.skill_registry.get_by_route(route_name)
        return {"route_name": route_name, "skill_name": skill.name if skill else route_name, "source": source}

    def _fallback_answer(self, question: str, route_name: str, context: AppContext, observations: dict[str, Any] | None = None) -> str:
        observations = observations or {}
        serialized = self.serialize_context(context)
        if route_name == "profile": return self._summarize_profile(observations.get("获取用户档案") or self.get_profile())
        if route_name == "risk_assessment":
            risk_payload = observations.get("计算痛风风险")
            trigger_payload = observations.get("识别痛风诱因")
            abnormal_payload = observations.get("识别异常指标")
            if risk_payload is not None:
                local_context = {"risk_result": {"uric_acid_risk_level_cn": self.label_risk(risk_payload.uric_acid_risk_level), "attack_risk_level_cn": self.label_risk(risk_payload.attack_risk_level), "overall_risk_score": risk_payload.overall_risk_score, "explanation": risk_payload.explanation}, "trigger_summary": [{"label": TRIGGER_LABELS.get(name, name), "count": count} for name, count in list((trigger_payload or {}).items())[:5]], "abnormal_items": [item.message for item in (abnormal_payload or [])]}
                return self.risk_runtime.summarize_risk(local_context) + "\n" + self.risk_runtime.summarize_triggers(local_context) + "\n" + self.risk_runtime.summarize_abnormal_items(local_context)
            return self.risk_runtime.summarize_risk(serialized) + "\n" + self.risk_runtime.summarize_triggers(serialized)
        if route_name == "lifestyle_coach": return self.lifestyle_runtime.answer_food_question(question, serialized)
        if route_name == "medication_followup": return self.medication_runtime.summarize_medication_and_reminders(serialized)
        if route_name == "reporting":
            return self.reporting_runtime.explain_report(
                observations.get("生成周报")
                or observations.get("生成月报")
                or self._call_skill_tool("reporting", "生成周报", context.profile, context.logs, context.labs, context.attacks, context.symptom_logs),
                serialized,
            )
        return self.risk_runtime.summarize_risk(serialized) + "\n" + self.lifestyle_runtime.build_daily_lifestyle_guidance(serialized)

    def _route_question(self, question: str) -> tuple[str, dict[str, Any]]:
        match = self.skill_registry.match_question(question)
        if match and match["score"] > 0: return match["route_name"], {"source": "skill_registry", "matched_hints": match["matched_hints"], "score": match["score"], "skill_name": match["name"], "allowed_tools": match["allowed_tools"], "execution_tools": match["execution_tools"]}
        route_name = self._legacy_route_question(question)
        return route_name, {"source": "fallback_rules", "matched_hints": ROUTE_FALLBACKS.get(route_name, []), "score": 0, "skill_name": route_name, "allowed_tools": self.get_allowed_tools(route_name), "execution_tools": self.skill_registry.get_execution_tools(route_name)}

    def _legacy_route_question(self, question: str) -> str:
        lower = question.lower()
        if any(keyword in question for keyword in ROUTE_FALLBACKS["profile"]): return "profile"
        if any(keyword in question for keyword in ROUTE_FALLBACKS["risk_assessment"]) or "risk" in lower: return "risk_assessment"
        if any(keyword in question for keyword in ROUTE_FALLBACKS["lifestyle_coach"]): return "lifestyle_coach"
        if any(keyword in question for keyword in ROUTE_FALLBACKS["medication_followup"]): return "medication_followup"
        if any(keyword in question for keyword in ROUTE_FALLBACKS["reporting"]): return "reporting"
        if any(keyword in question for keyword in ROUTE_FALLBACKS["intake"]): return "intake"
        return "orchestrator"

    def _infer_loop_intent(self, question: str, route_name: str) -> str:
        if route_name == "reporting": return "export" if "导出" in question else "explain"
        if route_name == "profile": return "update" if any(hint in question for hint in ["修改", "更新", "补充", "设置"]) else "read"
        if route_name == "medication_followup":
            if "提醒" in question and any(hint in question for hint in ["创建", "设置"]): return "create_reminder"
            if "新增" in question or "添加" in question: return "add_medication"
            if "打卡" in question or "已服药" in question: return "log_medication"
            return "review"
        if route_name == "risk_assessment": return "trend" if self._question_mentions_trend(question) else "review"
        return "default"

    def _infer_period_type(self, question: str) -> str: return "monthly" if ("月报" in question or ("月" in question and "报告" in question)) else "weekly"
    def _question_mentions_trend(self, question: str) -> bool: return any(hint in question for hint in ["趋势", "未来", "接下来", "最近会不会", "未来几天", "预测"])
    def _build_step_thought(self, route_name: str, tool_name: str, question: str) -> str: return "当前技能为 %s。为了回答“%s”，先调用 %s 获取可验证信息。" % (route_name, question, tool_name)
    def _medication_completion_rate(self, frame: pd.DataFrame) -> float | None:
        if frame.empty or "medication_taken_flag" not in frame.columns: return None
        series = pd.to_numeric(frame["medication_taken_flag"], errors="coerce")
        if not series.notna().any(): return None
        return float(series.fillna(0).clip(0, 1).mean() * 100)
    def _summarize_profile(self, profile: dict[str, Any]) -> str:
        parts: list[str] = []
        if profile.get("name"): parts.append("当前档案用户：%s。" % profile["name"])
        if profile.get("target_uric_acid"): parts.append("目标尿酸为 %s。" % profile["target_uric_acid"])
        conditions: list[str] = []
        if profile.get("has_gout_diagnosis"): conditions.append("已确诊痛风")
        if profile.get("has_hyperuricemia"): conditions.append("高尿酸血症")
        if profile.get("has_ckd"): conditions.append("慢性肾病")
        if profile.get("has_hypertension"): conditions.append("高血压")
        if profile.get("has_diabetes"): conditions.append("糖尿病")
        if conditions: parts.append("当前长期健康背景包括：%s。" % "、".join(conditions))
        if profile.get("doctor_advice"): parts.append("AI 管理意见：%s。" % profile["doctor_advice"])
        return " ".join(parts) if parts else "当前还没有完善的长期健康档案，建议先补充目标尿酸、基础病和 AI 管理意见。"
    @staticmethod
    def label_risk(value: str) -> str: return RISK_LABELS.get(value, value)
