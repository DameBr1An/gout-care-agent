from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from gout_agent import data, memory
from gout_agent.skill_registry import SkillRegistry, load_skill_registry
from gout_agent.skills import intake, lifestyle, medication, reporting_skill, risk_skill
from gout_agent.toolkit import ToolRegistry, build_default_tool_registry, serialize_tool_result

RISK_LABELS = {"Low": "低", "Moderate": "中", "High": "高", "up": "上升", "down": "下降", "stable": "稳定"}
TRIGGER_LABELS = {"alcohol": "饮酒", "beer": "啤酒", "spirits": "烈酒", "seafood": "海鲜", "shellfish": "贝类", "organ_meat": "动物内脏", "red_meat": "红肉", "hotpot": "火锅", "barbecue": "烧烤", "sugary_drinks": "含糖饮料", "low_hydration": "饮水不足", "missed_medication": "未按时服药"}
ROUTE_FALLBACKS = {"profile": ["档案", "资料", "基本信息", "目标尿酸", "AI建议", "AI 管理意见", "基础病"], "risk_assessment": ["尿酸", "风险", "发作", "诱因", "异常", "趋势"], "lifestyle_coach": ["吃", "喝", "饮食", "食物", "运动", "喝水", "啤酒", "海鲜"], "medication_followup": ["药", "服药", "提醒", "复查", "依从性"], "reporting": ["周报", "月报", "报告", "导出"], "intake": ["记录", "录入", "今天", "昨晚", "刚刚"]}

@dataclass
class AppContext:
    profile: dict[str, Any]
    logs: pd.DataFrame
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
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.registry: ToolRegistry = build_default_tool_registry(project_root)
        self.skill_registry: SkillRegistry = load_skill_registry(project_root / "skills")
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
        labs = self.registry.call("获取化验历史", _trace_context=trace)
        attacks = self.registry.call("获取发作历史", 365, _trace_context=trace)
        medications = self.registry.call("获取药物列表", _trace_context=trace)
        reminders = self.registry.call("获取启用提醒", _trace_context=trace)
        risk_result = self.registry.call("计算痛风风险", profile, logs, labs, attacks, _trace_context=trace)
        trigger_counts = self.registry.call("识别痛风诱因", logs, 14, _trace_context=trace)
        abnormal_metrics = self.registry.call("识别异常指标", profile, labs.iloc[-1].to_dict() if not labs.empty else None, logs.iloc[-1].to_dict() if not logs.empty else None, _trace_context=trace)
        long_term_memory = memory.build_long_term_memory(profile, logs, labs, attacks)
        self._sync_long_term_memory(long_term_memory)
        session_memories = self._load_session_memories(limit=12)
        return AppContext(profile=profile, logs=logs, labs=labs, attacks=attacks, medications=medications, reminders=reminders, risk_result=risk_result, trigger_summary=[{"name": k, "label": TRIGGER_LABELS.get(k, k), "count": v} for k, v in list(trigger_counts.items())[:5]], abnormal_items=[item.message for item in abnormal_metrics], medication_completion_rate=self._medication_completion_rate(logs.tail(7)), llm_status=self.registry.call("获取本地模型状态", _trace_context=trace), long_term_memory=long_term_memory, session_memories=session_memories)

    def get_ui_snapshot(self, context: AppContext) -> dict[str, Any]:
        latest_lab = context.labs.iloc[-1].to_dict() if not context.labs.empty else {}
        return {"latest_uric_acid": latest_lab.get("uric_acid"), "attack_risk_label": self.label_risk(context.risk_result.attack_risk_level), "uric_acid_risk_label": self.label_risk(context.risk_result.uric_acid_risk_level), "overall_risk_score": context.risk_result.overall_risk_score, "explanation": context.risk_result.explanation, "hydration_advice": context.risk_result.hydration_advice, "diet_advice": context.risk_result.diet_advice, "exercise_advice": context.risk_result.exercise_advice, "behavior_goal": context.risk_result.behavior_goal, "abnormal_items": context.abnormal_items, "trigger_summary": context.trigger_summary, "medication_completion_rate": context.medication_completion_rate, "active_reminder_count": len(context.reminders), "active_medication_count": len(context.medications.loc[context.medications["active_flag"] == 1]) if not context.medications.empty and "active_flag" in context.medications.columns else 0, "llm_status": context.llm_status}

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
        llm_context = self._build_llm_context(context)
        llm_context.update({"selected_report": report_payload, "allowed_tools": self.get_allowed_tools("reporting"), "execution_plan": plan})
        llm_result = self.registry.call("调用本地痛风模型", "请用简洁中文总结这份痛风管理报告，说明当前风险、主要诱因、今天的重点行动，以及什么情况下要及时就医。", llm_context, _trace_context=self._trace_context("reporting", source="llm_report"))
        return {"report": report_payload, "explanation": llm_result.content if llm_result.ok else reporting_skill.explain_report(report_payload, self.serialize_context(context)), "source": "local_llm" if llm_result.ok else "rule_fallback", "error": None if llm_result.ok else llm_result.error_message}

    def export_report(self, period_type: str, format_name: str, context: AppContext) -> Path:
        plan = self.build_execution_plan("reporting", intent="export", period_type=period_type)
        return self._execute_reporting_plan(plan, context, format_name)["path"]

    def parse_intake_text(self, text: str) -> dict[str, Any]: return intake.parse_free_text_entry(text)
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
        return {"risk_result": {"uric_acid_risk_level": context.risk_result.uric_acid_risk_level, "attack_risk_level": context.risk_result.attack_risk_level, "uric_acid_risk_level_cn": self.label_risk(context.risk_result.uric_acid_risk_level), "attack_risk_level_cn": self.label_risk(context.risk_result.attack_risk_level), "overall_risk_score": context.risk_result.overall_risk_score, "explanation": context.risk_result.explanation, "hydration_advice": context.risk_result.hydration_advice, "diet_advice": context.risk_result.diet_advice, "exercise_advice": context.risk_result.exercise_advice, "behavior_goal": context.risk_result.behavior_goal}, "trigger_summary": context.trigger_summary, "abnormal_items": context.abnormal_items, "medication_completion_rate": context.medication_completion_rate, "active_reminder_count": len(context.reminders), "long_term_memory": context.long_term_memory, "session_memories": context.session_memories}

    def _build_llm_context(self, context: AppContext) -> dict[str, Any]:
        payload = self.serialize_context(context)
        payload.update({"user_profile": {"name": context.profile.get("name"), "target_uric_acid": context.profile.get("target_uric_acid"), "has_gout_diagnosis": bool(context.profile.get("has_gout_diagnosis")), "has_hyperuricemia": bool(context.profile.get("has_hyperuricemia")), "has_ckd": bool(context.profile.get("has_ckd")), "has_hypertension": bool(context.profile.get("has_hypertension")), "has_diabetes": bool(context.profile.get("has_diabetes")), "ai_advice": context.profile.get("doctor_advice")}, "latest_daily_log": context.logs.iloc[-1].to_dict() if not context.logs.empty else {}, "latest_lab_result": context.labs.iloc[-1].to_dict() if not context.labs.empty else {}, "recent_attack_records": context.attacks.head(5).to_dict(orient="records") if not context.attacks.empty else [], "ai_advice_summary": context.long_term_memory.get("ai_advice_summary") or context.long_term_memory.get("doctor_advice_summary"), "behavior_portraits": context.long_term_memory.get("behavior_portraits"), "attack_patterns": context.long_term_memory.get("attack_patterns"), "user_preferences": context.long_term_memory.get("user_preferences"), "recent_session_memories": context.session_memories[-6:]})
        return payload

    def _load_session_memories(self, limit: int = 12) -> list[dict[str, Any]]:
        frame = data.get_session_memories(self.project_root, limit=limit)
        if frame.empty:
            return []
        records = frame.iloc[::-1].to_dict(orient="records")
        return [serialize_tool_result(record) for record in records]

    def _save_session_memory(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not str(content or "").strip():
            return
        data.save_session_memory(self.project_root, role, content, metadata=metadata or {})

    def _sync_long_term_memory(self, long_term_memory: dict[str, Any]) -> None:
        latest = data.get_latest_memory_snapshot(self.project_root, "long_term_memory")
        latest_clean = dict(latest or {})
        latest_clean.pop("updated_at", None)
        current_clean = dict(long_term_memory)
        current_clean.pop("updated_at", None)
        if latest_clean != current_clean:
            data.save_memory_snapshot(self.project_root, "long_term_memory", long_term_memory)

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
        if tool_name in {"生成周报", "生成月报"}: return self._call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks)
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
            if tool_name in {"生成周报", "生成月报"}: report_payload = self._call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks)
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
                return risk_skill.summarize_risk(local_context) + "\n" + risk_skill.summarize_triggers(local_context) + "\n" + risk_skill.summarize_abnormal_items(local_context)
            return risk_skill.summarize_risk(serialized) + "\n" + risk_skill.summarize_triggers(serialized)
        if route_name == "lifestyle_coach": return lifestyle.answer_food_question(question, serialized)
        if route_name == "medication_followup": return medication.summarize_medication_and_reminders(serialized)
        if route_name == "reporting": return reporting_skill.explain_report(observations.get("生成周报") or observations.get("生成月报") or self._call_skill_tool("reporting", "生成周报", context.profile, context.logs, context.labs, context.attacks), serialized)
        return risk_skill.summarize_risk(serialized) + "\n" + lifestyle.build_daily_lifestyle_guidance(serialized)

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
