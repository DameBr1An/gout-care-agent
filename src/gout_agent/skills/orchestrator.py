from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from gout_agent import data, llm, memory, runtime_fallbacks, runtime_jobs, runtime_state, runtime_tools
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
    twin_state: dict[str, Any]
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
        self.skill_runtimes = {
            "intake": self.intake_runtime,
            "profile": None,
            "risk_assessment": self.risk_runtime,
            "lifestyle_coach": self.lifestyle_runtime,
            "medication_followup": self.medication_runtime,
            "reporting": self.reporting_runtime,
            "lab_report": self.lab_report_runtime,
        }
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
        twin_state = self._build_twin_state(long_term_memory, risk_overview)
        return AppContext(user_journal=user_journal, site_history=site_history, risk_overview=risk_overview, twin_state=twin_state, profile=profile, logs=logs, symptom_logs=symptom_logs, labs=labs, attacks=attacks, medications=medications, reminders=reminders, risk_result=risk_result, trigger_summary=trigger_summary, abnormal_items=[item.message for item in abnormal_metrics], medication_completion_rate=self._medication_completion_rate(logs.tail(7)), llm_status=self.registry.call("获取本地模型状态", _trace_context=trace), long_term_memory=long_term_memory, session_memories=session_memories)

    def get_ui_snapshot(self, context: AppContext) -> dict[str, Any]:
        return {"latest_uric_acid": None, "attack_risk_label": context.risk_overview.get("attack_risk_label"), "uric_acid_risk_label": context.risk_overview.get("uric_acid_risk_label"), "overall_risk_score": context.risk_overview.get("overall_risk_score"), "explanation": context.risk_overview.get("explanation"), "hydration_advice": context.risk_overview.get("hydration_advice"), "diet_advice": context.risk_overview.get("diet_advice"), "exercise_advice": context.risk_overview.get("exercise_advice"), "behavior_goal": context.risk_overview.get("behavior_goal"), "abnormal_items": context.risk_overview.get("abnormal_items", []), "trigger_summary": context.risk_overview.get("trigger_summary", []), "medication_completion_rate": context.medication_completion_rate, "active_reminder_count": len(context.reminders), "active_medication_count": len(context.medications.loc[context.medications["active_flag"] == 1]) if not context.medications.empty and "active_flag" in context.medications.columns else 0, "llm_status": context.llm_status}

    def _build_twin_state(self, long_term_memory: dict[str, Any], risk_overview: dict[str, Any]) -> dict[str, Any]:
        enriched_memory = dict(long_term_memory)
        enriched_memory.setdefault("memory_summary", memory.build_llm_memory_summary(long_term_memory))
        enriched_memory.setdefault("report_memory_summary", memory.build_report_memory_summary(long_term_memory))
        enriched_memory.setdefault("updated_at", long_term_memory.get("updated_at"))
        return runtime_state.build_twin_state(enriched_memory, risk_overview)

    def answer_coach_question(self, question: str, context: AppContext) -> dict[str, Any]:
        return self.run_agent_loop(question, context)

    def _get_skill_runtime(self, route_name: str):
        return self.skill_runtimes.get(route_name)

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
        runtime = self._get_skill_runtime("reporting")
        llm_context = self._build_interpretation_context(context, selected_report=report_payload, period_type=period_type)
        if runtime is not None:
            llm_context = runtime.prepare(llm_context, report_payload=report_payload, twin_state=context.twin_state, period_type=period_type)
        llm_context.update({"allowed_tools": self.get_allowed_tools("reporting"), "execution_plan": plan})
        llm_result = self.registry.call("调用本地痛风模型", "请结合这份周期报告、当前健康分身、近期 7/30/90 天行为变化、历史报告摘要和当前风险，用简洁中文解读本期情况，说明现在最重要的变化、主要诱因、今天的重点行动，以及什么情况下要及时就医。", llm_context, _trace_context=self._trace_context("reporting", source="llm_report"))
        fallback_text = runtime.summarize("explain_report", report_payload, self.serialize_context(context)) if runtime is not None else self.reporting_runtime.explain_report(report_payload, self.serialize_context(context))
        return {"report": report_payload, "explanation": llm_result.content if llm_result.ok else fallback_text, "source": "local_llm" if llm_result.ok else "rule_fallback", "error": None if llm_result.ok else llm_result.error_message}

    def explain_uploaded_lab_reports(self, uploaded_files: list[dict[str, Any]], context: AppContext) -> dict[str, Any]:
        runtime = self._get_skill_runtime("lab_report")
        parsed_lab_reports = runtime.run("parse_uploaded_lab_files", uploaded_files, self._extract_lab_metrics_with_local_model) if runtime is not None else self.lab_report_runtime.parse_uploaded_lab_files(uploaded_files, self._extract_lab_metrics_with_local_model)
        return self.explain_parsed_lab_reports(parsed_lab_reports, context, uploaded_files=uploaded_files)

    def explain_parsed_lab_reports(
        self,
        parsed_lab_reports: dict[str, Any],
        context: AppContext,
        uploaded_files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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

    def parse_intake_text(self, text: str) -> dict[str, Any]:
        runtime = self._get_skill_runtime("intake")
        return runtime.run("parse_free_text_entry", text) if runtime is not None else self.intake_runtime.parse_free_text_entry(text)
    def describe_skills(self) -> list[dict[str, Any]]: return self.skill_registry.describe()
    def describe_tools(self, include_schema: bool = False) -> list[dict[str, Any]]: return self.registry.describe(include_schema=include_schema)
    def get_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]: return self.registry.get_traces(limit=limit)
    def get_allowed_tools(self, route_name: str) -> list[str]: return list(self.skill_registry.get_allowed_tools(route_name))
    def is_tool_allowed(self, route_name: str, tool_name: str) -> bool:
        allowed_tools = self.get_allowed_tools(route_name)
        return route_name == "orchestrator" if not allowed_tools else tool_name in allowed_tools
    def list_background_jobs(self, status: str | None = None, limit: int = 20) -> pd.DataFrame:
        return data.get_background_jobs(self.project_root, status=status, limit=limit, user_id=self.user_id)

    def submit_background_job(self, job_type: str, payload: dict[str, Any] | None = None) -> int:
        payload = payload or {}
        return data.create_background_job(self.project_root, job_type, payload, user_id=self.user_id)

    def run_pending_background_jobs(self, limit: int = 5) -> list[dict[str, Any]]:
        jobs = data.get_pending_background_jobs(self.project_root, limit=limit, user_id=self.user_id)
        results: list[dict[str, Any]] = []
        if jobs.empty:
            return results
        for _, row in jobs.iterrows():
            job_id = int(row["id"])
            payload = row.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            started_at = pd.Timestamp.now().isoformat()
            data.update_background_job(self.project_root, job_id, status="running", started_at=started_at)
            try:
                result_payload = self._execute_background_job(str(row.get("job_type") or ""), payload)
                data.update_background_job(
                    self.project_root,
                    job_id,
                    status="completed",
                    result_payload=result_payload,
                    finished_at=pd.Timestamp.now().isoformat(),
                )
                results.append({"job_id": job_id, "status": "completed", "result": result_payload})
            except Exception as exc:
                data.update_background_job(
                    self.project_root,
                    job_id,
                    status="failed",
                    error_message=str(exc),
                    finished_at=pd.Timestamp.now().isoformat(),
                )
                results.append({"job_id": job_id, "status": "failed", "error": str(exc)})
        return results

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

    def save_daily_log(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("intake", "记录日常健康", payload, _audit_meta=audit_meta)["result"]
    def save_joint_symptom(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("intake", "记录部位症状", payload, _audit_meta=audit_meta)["result"]
    def save_lab_result(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("intake", "记录化验结果", payload, _audit_meta=audit_meta)["result"]
    def update_profile(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any:
        if "获取用户档案" in self.build_execution_plan("profile", intent="update"): self._call_skill_tool("profile", "获取用户档案")
        return self._run_write_action("profile", "更新用户档案", payload, _audit_meta=audit_meta)["result"]
    def get_profile(self) -> Any: return self._call_skill_tool("profile", (self.build_execution_plan("profile", intent="read") or ["获取用户档案"])[0])
    def save_attack(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("intake", "记录痛风发作", payload, _audit_meta=audit_meta)["result"]
    def add_medication(self, payload: dict[str, Any], audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("medication_followup", "添加药物方案", payload, _audit_meta=audit_meta)["result"]
    def create_reminder(self, reminder_type: str, title: str, schedule_rule: str, next_trigger_at: str, audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("medication_followup", "创建提醒", reminder_type, title, schedule_rule, next_trigger_at, _audit_meta=audit_meta)["result"]
    def log_medication_taken(self, medication_id: int, status: str, taken_time: str | None = None, audit_meta: dict[str, Any] | None = None) -> Any: return self._run_write_action("medication_followup", "记录服药情况", medication_id, status, None, taken_time, _audit_meta=audit_meta)["result"]
    def get_write_audit_logs(self, limit: int = 50) -> pd.DataFrame:
        return data.get_write_audit_logs(self.project_root, limit=limit, user_id=self.user_id)

    def _run_write_action(self, route_name: str, tool_name: str, *args, **kwargs) -> dict[str, Any]:
        audit_meta = kwargs.pop("_audit_meta", None) or {}
        tool_spec = self.registry.get_spec(tool_name)
        payload_preview = {"args": serialize_tool_result(list(args)), "kwargs": serialize_tool_result(kwargs)}
        source = str(audit_meta.get("source") or "ui_form")
        confirmed = bool(audit_meta.get("confirmed"))
        if tool_spec and tool_spec.sensitive_write and not confirmed:
            data.log_write_audit(
                self.project_root,
                route_name,
                tool_name,
                payload_preview,
                source=source,
                sensitive_write=True,
                confirmed_flag=False,
                status="blocked",
                error_message="敏感写操作缺少确认",
                user_id=self.user_id,
            )
            raise PermissionError(f"工具 {tool_name} 属于敏感写操作，需要显式确认。")
        try:
            result = self._call_skill_tool(route_name, tool_name, *args, **kwargs)
        except Exception as exc:
            if tool_spec and tool_spec.access_mode == "write":
                data.log_write_audit(
                    self.project_root,
                    route_name,
                    tool_name,
                    payload_preview,
                    source=source,
                    sensitive_write=bool(tool_spec.sensitive_write),
                    confirmed_flag=confirmed,
                    status="failed",
                    error_message=str(exc),
                    user_id=self.user_id,
                )
            raise
        if tool_spec and tool_spec.access_mode == "write":
            data.log_write_audit(
                self.project_root,
                route_name,
                tool_name,
                payload_preview,
                source=source,
                sensitive_write=bool(tool_spec.sensitive_write),
                confirmed_flag=confirmed,
                status="executed",
                user_id=self.user_id,
            )
        refreshed_context = self._refresh_context_state()
        return {"result": result, "context": refreshed_context, "ui_snapshot": self.get_ui_snapshot(refreshed_context)}

    def _refresh_context_state(self) -> AppContext:
        refreshed_context = self.load_context()
        self.sync_daily_snapshot(refreshed_context)
        return refreshed_context

    def _execute_background_job(self, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return runtime_jobs.execute_background_job(
            self.project_root,
            self.user_id,
            job_type,
            payload,
            load_context=self.load_context,
            explain_report=self.explain_report,
            refresh_context_state=self._refresh_context_state,
            get_skill_runtime=self._get_skill_runtime,
            extract_lab_metrics_with_local_model=self._extract_lab_metrics_with_local_model,
        )

    def sync_daily_snapshot(self, context: AppContext) -> None:
        existing = self.registry.call("获取风险快照", 2, _trace_context=self._trace_context("orchestrator", source="snapshot"))
        today = pd.Timestamp.today().date().isoformat()
        if not existing.empty and today in existing["snapshot_date"].astype(str).tolist(): return
        self.registry.call("保存风险快照", {"snapshot_date": today, "uric_acid_risk_level": context.risk_result.uric_acid_risk_level, "attack_risk_level": context.risk_result.attack_risk_level, "overall_risk_score": context.risk_result.overall_risk_score, "top_risk_factors": context.risk_result.top_risk_factors, "trend_direction": context.risk_result.trend_direction}, _trace_context=self._trace_context("orchestrator", source="snapshot"))

    def serialize_context(self, context: AppContext) -> dict[str, Any]:
        return runtime_state.serialize_context_payload(context, self.label_risk)

    def _build_llm_context(self, context: AppContext) -> dict[str, Any]:
        return runtime_state.build_llm_context_payload(context, self.label_risk)

    def _build_interpretation_context(
        self,
        context: AppContext,
        selected_report: dict[str, Any] | None = None,
        period_type: str | None = None,
        uploaded_lab_reports: list[dict[str, Any]] | None = None,
        parsed_lab_reports: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report_history = self.registry.call("获取报告历史", _trace_context=self._trace_context("reporting", source="report_history"))
        return runtime_state.build_interpretation_context_payload(
            context,
            self.label_risk,
            report_history,
            selected_report=selected_report,
            period_type=period_type,
            uploaded_lab_reports=uploaded_lab_reports,
            parsed_lab_reports=parsed_lab_reports,
        )

    def _build_harness_state_summary(self, context: AppContext) -> dict[str, Any]:
        return runtime_state.build_harness_state_summary(context)

    def _extract_lab_metrics_with_local_model(self, uploaded_files: list[dict[str, Any]]) -> dict[str, Any]:
        return llm.ask_local_lab_vision_llm(uploaded_files)

    def _build_user_journal(self, profile: dict[str, Any], logs: pd.DataFrame) -> dict[str, Any]:
        return runtime_state.build_user_journal(profile, logs)

    def _build_site_history(self, symptom_logs: pd.DataFrame, attacks: pd.DataFrame) -> pd.DataFrame:
        return runtime_state.build_site_history(symptom_logs, attacks)

    def _build_risk_overview(self, risk_result: Any, trigger_summary: list[dict[str, Any]], abnormal_items: list[str]) -> dict[str, Any]:
        return runtime_state.build_risk_overview(risk_result, trigger_summary, abnormal_items, self.label_risk)

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
        return runtime_tools.execute_loop_tool(self._call_skill_tool, route_name, tool_name, context, observations)

    def _execute_reporting_loop_tool(self, tool_name: str, context: AppContext, observations: dict[str, Any]) -> Any:
        return runtime_tools.execute_reporting_loop_tool(self._call_skill_tool, tool_name, context, observations)

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
        return runtime_tools.execute_reporting_plan(self._call_skill_tool, plan, context, format_name)

    def _call_skill_tool(self, route_name: str, tool_name: str, *args, **kwargs) -> Any:
        self._assert_skill_tool_permission(route_name, tool_name)
        return self.registry.call(tool_name, *args, _trace_context=self._trace_context(route_name), **kwargs)

    def _assert_skill_tool_permission(self, route_name: str, tool_name: str) -> None:
        if not self.is_tool_allowed(route_name, tool_name):
            raise PermissionError("技能 %s 不允许调用工具 %s；允许工具为：%s" % (route_name, tool_name, "、".join(self.get_allowed_tools(route_name)) or "无"))
        skill = self.skill_registry.get_by_route(route_name)
        tool_spec = self.registry.get_spec(tool_name)
        if skill is None or tool_spec is None:
            return
        if tool_spec.access_mode == "write" and tool_name not in skill.write_permissions:
            raise PermissionError("技能 %s 没有写权限调用工具 %s。" % (route_name, tool_name))

    def _trace_context(self, route_name: str, source: str = "skill_tool") -> dict[str, str]:
        skill = self.skill_registry.get_by_route(route_name)
        return {"route_name": route_name, "skill_name": skill.name if skill else route_name, "source": source}

    def _fallback_answer(self, question: str, route_name: str, context: AppContext, observations: dict[str, Any] | None = None) -> str:
        return runtime_fallbacks.build_fallback_answer(
            route_name,
            question,
            self.serialize_context(context),
            observations or {},
            label_risk=self.label_risk,
            get_profile=self.get_profile,
            call_reporting_report=lambda: self._call_skill_tool("reporting", "生成周报", context.profile, context.logs, context.labs, context.attacks, context.symptom_logs),
            get_skill_runtime=self._get_skill_runtime,
        )

    def _route_question(self, question: str) -> tuple[str, dict[str, Any]]:
        match = self.skill_registry.match_question(question)
        if match and match["score"] > 0: return match["route_name"], {"source": "skill_registry", "matched_hints": match["matched_hints"], "score": match["score"], "skill_name": match["name"], "allowed_tools": match["allowed_tools"], "execution_tools": match["execution_tools"], "write_permissions": match.get("write_permissions", [])}
        route_name = self._legacy_route_question(question)
        skill = self.skill_registry.get_by_route(route_name)
        return route_name, {"source": "fallback_rules", "matched_hints": ROUTE_FALLBACKS.get(route_name, []), "score": 0, "skill_name": route_name, "allowed_tools": self.get_allowed_tools(route_name), "execution_tools": self.skill_registry.get_execution_tools(route_name), "write_permissions": list(skill.write_permissions) if skill else []}

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
    @staticmethod
    def label_risk(value: str) -> str: return RISK_LABELS.get(value, value)
