from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from gout_agent import data, llm, memory, reporting, risk


ToolHandler = Callable[..., Any]


@dataclass
class ToolParameterSpec:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any | None = None


@dataclass
class ToolExampleSpec:
    description: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultSpec:
    type: str
    description: str


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler
    domain: str
    access_mode: str
    sensitive_write: bool = False
    parameters: list[ToolParameterSpec] = field(default_factory=list)
    returns: ToolResultSpec | None = None
    examples: list[ToolExampleSpec] = field(default_factory=list)


@dataclass
class ToolTraceRecord:
    timestamp: str
    tool_name: str
    success: bool
    route_name: str | None = None
    skill_name: str | None = None
    source: str = "registry"
    args_preview: list[Any] = field(default_factory=list)
    kwargs_preview: dict[str, Any] = field(default_factory=dict)
    result_preview: Any | None = None
    error: str | None = None


class ToolRegistry:
    def __init__(self, max_trace_entries: int = 200) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._traces: deque[ToolTraceRecord] = deque(maxlen=max_trace_entries)

    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        domain: str | None = None,
        access_mode: str | None = None,
        sensitive_write: bool | None = None,
        parameters: list[ToolParameterSpec] | None = None,
        returns: ToolResultSpec | None = None,
        examples: list[ToolExampleSpec] | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            domain=domain or _infer_tool_domain(name),
            access_mode=access_mode or _infer_tool_access_mode(name),
            sensitive_write=_infer_sensitive_write(name) if sensitive_write is None else sensitive_write,
            parameters=parameters or [],
            returns=returns,
            examples=examples or [],
        )

    def call(self, name: str, *args, _trace_context: dict[str, Any] | None = None, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"未知工具：{name}")

        trace_context = dict(_trace_context or {})
        args_preview = [_summarize_for_trace(item) for item in args]
        kwargs_preview = {key: _summarize_for_trace(value) for key, value in kwargs.items()}

        try:
            result = self._tools[name].handler(*args, **kwargs)
            self._append_trace(
                ToolTraceRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    tool_name=name,
                    success=True,
                    route_name=trace_context.get("route_name"),
                    skill_name=trace_context.get("skill_name"),
                    source=trace_context.get("source", "registry"),
                    args_preview=args_preview,
                    kwargs_preview=kwargs_preview,
                    result_preview=_summarize_for_trace(result),
                )
            )
            return result
        except Exception as exc:
            self._append_trace(
                ToolTraceRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    tool_name=name,
                    success=False,
                    route_name=trace_context.get("route_name"),
                    skill_name=trace_context.get("skill_name"),
                    source=trace_context.get("source", "registry"),
                    args_preview=args_preview,
                    kwargs_preview=kwargs_preview,
                    error=str(exc),
                )
            )
            raise

    def describe(self, include_schema: bool = False) -> list[dict[str, Any]]:
        tools = sorted(self._tools.values(), key=lambda item: item.name)
        if not include_schema:
            return [{"name": tool.name, "description": tool.description} for tool in tools]
        return [self._serialize_tool(tool) for tool in tools]

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def get_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        records = list(self._traces)[-max(limit, 0) :]
        return [serialize_tool_result(asdict(item)) for item in reversed(records)]

    def clear_traces(self) -> None:
        self._traces.clear()

    def _append_trace(self, record: ToolTraceRecord) -> None:
        self._traces.append(record)

    def _serialize_tool(self, tool: ToolSpec) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "domain": tool.domain,
            "access_mode": tool.access_mode,
            "sensitive_write": tool.sensitive_write,
            "parameters": [serialize_tool_result(asdict(item)) for item in tool.parameters],
            "returns": serialize_tool_result(asdict(tool.returns)) if tool.returns else None,
            "examples": [serialize_tool_result(asdict(item)) for item in tool.examples],
        }
        payload["required_parameters"] = [item.name for item in tool.parameters if item.required]
        return payload


def serialize_tool_result(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        frame = value.copy().where(pd.notnull(value), None)
        return frame.to_dict(orient="records")
    if isinstance(value, pd.Series):
        series = value.where(pd.notnull(value), None)
        return series.to_dict()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): serialize_tool_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_tool_result(item) for item in value]
    if is_dataclass(value):
        return serialize_tool_result(asdict(value))
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return serialize_tool_result(vars(value))
    return str(value)


def _summarize_for_trace(value: Any) -> Any:
    serialized = serialize_tool_result(value)
    if isinstance(serialized, str):
        return serialized if len(serialized) <= 120 else serialized[:117] + "..."
    if isinstance(serialized, list):
        if len(serialized) <= 3:
            return serialized
        return serialized[:3] + [f"... 共 {len(serialized)} 项"]
    if isinstance(serialized, dict):
        items = list(serialized.items())[:6]
        trimmed = {key: value for key, value in items}
        if len(serialized) > 6:
            trimmed["..."] = f"共 {len(serialized)} 个字段"
        return trimmed
    return serialized


def _param(name: str, type_name: str, description: str, required: bool = True, default: Any | None = None) -> ToolParameterSpec:
    return ToolParameterSpec(name=name, type=type_name, description=description, required=required, default=default)


def _returns(type_name: str, description: str) -> ToolResultSpec:
    return ToolResultSpec(type=type_name, description=description)


def _example(description: str, args: list[Any] | None = None, kwargs: dict[str, Any] | None = None) -> ToolExampleSpec:
    return ToolExampleSpec(description=description, args=args or [], kwargs=kwargs or {})


def _infer_tool_domain(name: str) -> str:
    if any(token in name for token in ("档案", "资料")):
        return "profile"
    if any(token in name for token in ("日常", "部位症状", "化验", "发作")):
        return "record"
    if any(token in name for token in ("药物", "服药", "提醒")):
        return "medication"
    if any(token in name for token in ("风险", "诱因", "异常", "趋势")):
        return "risk"
    if any(token in name for token in ("报告", "周报", "月报")):
        return "report"
    if any(token in name for token in ("分身", "记忆", "画像")):
        return "memory"
    if "模型" in name:
        return "llm"
    return "system"


def _infer_tool_access_mode(name: str) -> str:
    if any(token in name for token in ("记录", "更新", "保存", "创建", "添加", "导出")):
        return "write"
    return "read"


def _infer_sensitive_write(name: str) -> bool:
    return _infer_tool_access_mode(name) == "write" and any(
        token in name for token in ("更新用户档案", "记录化验结果", "保存报告", "创建提醒", "添加药物方案")
    )


def build_default_tool_registry(project_root: Path, user_id: int = data.DEFAULT_USER_ID) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        "获取用户档案",
        "读取当前用户的基础档案和健康档案。",
        lambda: data.get_user_profile(project_root, user_id=user_id),
        returns=_returns("object", "用户基础档案和健康档案。"),
        examples=[_example("读取默认用户档案")],
    )
    registry.register(
        "更新用户档案",
        "更新当前用户的基础档案和健康档案。",
        lambda payload: data.update_user_profile(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "要更新的档案字段。")],
        returns=_returns("object", "更新后的用户档案。"),
        examples=[_example("更新目标尿酸", kwargs={"payload": {"target_uric_acid": 360}})],
    )
    registry.register(
        "记录日常健康",
        "写入一条每日健康记录。",
        lambda payload: data.log_daily_health_entry(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "每日记录内容。")],
        returns=_returns("integer", "新建记录 ID。"),
        examples=[_example("记录今日饮水和疼痛评分", kwargs={"payload": {"log_date": "2026-03-21", "water_ml": 1800, "pain_score": 2}})],
    )
    registry.register(
        "获取近期健康记录",
        "读取最近一段时间的每日健康记录。",
        lambda days=30: data.get_recent_health_entries(project_root, days=days, user_id=user_id),
        parameters=[_param("days", "integer", "向前查询的天数。", required=False, default=30)],
        returns=_returns("array<object>", "每日健康记录列表。"),
        examples=[_example("读取最近 7 天记录", kwargs={"days": 7})],
    )
    registry.register(
        "记录部位症状",
        "写入一条身体部位症状记录。",
        lambda payload: data.log_joint_symptom(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "部位、疼痛、红肿等症状内容。")],
        returns=_returns("integer", "新建部位症状记录 ID。"),
        examples=[_example("记录右脚大脚趾疼痛", kwargs={"payload": {"log_date": "2026-03-21", "body_site": "右脚大脚趾", "pain_score": 5}})],
    )
    registry.register(
        "获取部位症状历史",
        "读取最近一段时间的身体部位症状记录。",
        lambda days=90: data.get_recent_joint_symptoms(project_root, days=days, user_id=user_id),
        parameters=[_param("days", "integer", "向前查询的天数。", required=False, default=90)],
        returns=_returns("array<object>", "部位症状记录列表。"),
        examples=[_example("读取最近 30 天部位症状", kwargs={"days": 30})],
    )
    registry.register(
        "记录化验结果",
        "写入一条化验结果。",
        lambda payload: data.log_lab_result(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "化验结果内容。")],
        returns=_returns("integer", "新建化验记录 ID。"),
        examples=[_example("记录尿酸结果", kwargs={"payload": {"test_date": "2026-03-21", "uric_acid": 510}})],
    )
    registry.register(
        "获取化验历史",
        "读取化验历史。",
        lambda metric_name=None: data.get_lab_history(project_root, metric_name=metric_name, user_id=user_id),
        parameters=[_param("metric_name", "string", "可选，按单个指标过滤。", required=False, default=None)],
        returns=_returns("array<object>", "化验历史列表。"),
        examples=[_example("读取全部化验历史")],
    )
    registry.register(
        "记录痛风发作",
        "写入一条痛风发作记录。",
        lambda payload: data.log_gout_attack(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "发作信息。")],
        returns=_returns("integer", "新建发作记录 ID。"),
        examples=[_example("记录右脚大脚趾发作", kwargs={"payload": {"attack_date": "2026-03-21", "joint_site": "右脚大脚趾", "pain_score": 7}})],
    )
    registry.register(
        "获取发作历史",
        "读取最近一段时间的痛风发作历史。",
        lambda days=180: data.get_attack_history(project_root, days=days, user_id=user_id),
        parameters=[_param("days", "integer", "向前查询的天数。", required=False, default=180)],
        returns=_returns("array<object>", "痛风发作记录列表。"),
        examples=[_example("读取半年发作历史")],
    )
    registry.register(
        "添加药物方案",
        "新增一条药物方案。",
        lambda payload: data.add_medication(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "药物名称、剂量、频率等内容。")],
        returns=_returns("integer", "新建药物方案 ID。"),
        examples=[_example("新增非布司他", kwargs={"payload": {"medication_name": "非布司他", "dose": "40mg", "frequency": "每日一次"}})],
    )
    registry.register(
        "获取药物列表",
        "读取药物方案列表。",
        lambda active_only=False: data.get_medications(project_root, active_only=active_only, user_id=user_id),
        parameters=[_param("active_only", "boolean", "是否只返回启用中的药物。", required=False, default=False)],
        returns=_returns("array<object>", "药物方案列表。"),
        examples=[_example("只看启用中的药物", kwargs={"active_only": True})],
    )
    registry.register(
        "记录服药情况",
        "写入一条服药依从性记录。",
        lambda medication_id, status, scheduled_time=None, taken_time=None: data.log_medication_taken(
            project_root,
            medication_id=medication_id,
            status=status,
            scheduled_time=scheduled_time,
            taken_time=taken_time,
            user_id=user_id,
        ),
        parameters=[
            _param("medication_id", "integer", "药物方案 ID。"),
            _param("status", "string", "服药状态，例如 taken、missed。"),
            _param("scheduled_time", "string", "计划服药时间。", required=False, default=None),
            _param("taken_time", "string", "实际服药时间。", required=False, default=None),
        ],
        returns=_returns("integer", "新建服药记录 ID。"),
        examples=[_example("记录已服药", kwargs={"medication_id": 1, "status": "taken", "taken_time": "2026-03-21T08:00:00"})],
    )
    registry.register(
        "获取服药依从性",
        "读取最近一段时间的服药依从性记录。",
        lambda days=30: data.get_medication_adherence(project_root, days=days, user_id=user_id),
        parameters=[_param("days", "integer", "向前查询的天数。", required=False, default=30)],
        returns=_returns("array<object>", "服药记录列表。"),
        examples=[_example("读取最近 30 天服药记录")],
    )
    registry.register(
        "创建提醒",
        "创建一条提醒记录。",
        lambda reminder_type, title, schedule_rule, next_trigger_at: data.create_reminder(
            project_root,
            reminder_type=reminder_type,
            title=title,
            schedule_rule=schedule_rule,
            next_trigger_at=next_trigger_at,
            user_id=user_id,
        ),
        parameters=[
            _param("reminder_type", "string", "提醒类型。"),
            _param("title", "string", "提醒标题。"),
            _param("schedule_rule", "string", "提醒规则。"),
            _param("next_trigger_at", "string", "下次触发时间。"),
        ],
        returns=_returns("integer", "新建提醒 ID。"),
        examples=[_example("创建饮水提醒", kwargs={"reminder_type": "hydration", "title": "下午补水", "schedule_rule": "每日 15:00", "next_trigger_at": "2026-03-21T15:00:00"})],
    )
    registry.register(
        "获取启用提醒",
        "读取当前启用中的提醒。",
        lambda: data.list_active_reminders(project_root, user_id=user_id),
        returns=_returns("array<object>", "启用中的提醒列表。"),
        examples=[_example("读取全部启用提醒")],
    )
    registry.register(
        "获取风险快照",
        "读取风险快照历史。",
        lambda days=90: data.get_risk_snapshots(project_root, days=days, user_id=user_id),
        parameters=[_param("days", "integer", "向前查询的天数。", required=False, default=90)],
        returns=_returns("array<object>", "风险快照列表。"),
        examples=[_example("读取最近 90 天风险快照")],
    )
    registry.register(
        "保存风险快照",
        "写入一条风险快照。",
        lambda payload: data.save_risk_snapshot(project_root, payload, user_id=user_id),
        parameters=[_param("payload", "object", "风险快照内容。")],
        returns=_returns("integer", "新建风险快照 ID。"),
        examples=[_example("保存今日风险快照", kwargs={"payload": {"snapshot_date": "2026-03-21", "overall_risk_score": 4}})],
    )
    registry.register(
        "获取报告历史",
        "读取已保存的报告。",
        lambda report_type=None: data.get_reports(project_root, report_type=report_type, user_id=user_id),
        parameters=[_param("report_type", "string", "可选，按 weekly 或 monthly 过滤。", required=False, default=None)],
        returns=_returns("array<object>", "历史报告列表。"),
        examples=[_example("读取全部报告")],
    )
    registry.register(
        "生成数字分身",
        "基于用户资料、行为、发作和部位症状生成个人痛风数字分身。",
        lambda profile, logs, labs, attacks, symptom_logs=None: memory.build_gout_management_twin_profile(
            profile,
            logs,
            labs,
            attacks,
            symptom_logs=symptom_logs,
        ),
        parameters=[
            _param("profile", "object", "用户资料。"),
            _param("logs", "array<object>", "日常行为记录。"),
            _param("labs", "array<object>", "化验记录。"),
            _param("attacks", "array<object>", "发作记录。"),
            _param("symptom_logs", "array<object>", "部位症状记录。", required=False, default=None),
        ],
        returns=_returns("object", "数字分身画像。"),
        examples=[_example("生成当前数字分身")],
    )
    registry.register(
        "保存数字分身快照",
        "保存一份数字分身快照。",
        lambda profile_payload, snapshot_date=None: data.save_digital_twin_profile(project_root, profile_payload, snapshot_date=snapshot_date, user_id=user_id),
        parameters=[
            _param("profile_payload", "object", "数字分身画像内容。"),
            _param("snapshot_date", "string", "快照日期。", required=False, default=None),
        ],
        returns=_returns("integer", "新建数字分身快照 ID。"),
        examples=[_example("保存今日数字分身快照")],
    )
    registry.register(
        "获取数字分身历史",
        "读取已保存的数字分身快照。",
        lambda limit=20: data.get_digital_twin_profiles(project_root, limit=limit, user_id=user_id),
        parameters=[_param("limit", "integer", "最多返回多少条快照。", required=False, default=20)],
        returns=_returns("array<object>", "数字分身快照列表。"),
        examples=[_example("读取最近 10 条数字分身快照", kwargs={"limit": 10})],
    )
    registry.register(
        "保存报告",
        "保存一份已生成的报告。",
        lambda report_type, report_payload, period_start, period_end: data.save_report(
            project_root,
            report_type=report_type,
            report=report_payload,
            period_start=period_start,
            period_end=period_end,
            user_id=user_id,
        ),
        parameters=[
            _param("report_type", "string", "报告类型。"),
            _param("report_payload", "object", "完整报告内容。"),
            _param("period_start", "string", "报告开始日期。"),
            _param("period_end", "string", "报告结束日期。"),
        ],
        returns=_returns("integer", "新建报告 ID。"),
        examples=[_example("保存周报", kwargs={"report_type": "weekly", "report_payload": {"summary": "整体稳定"}, "period_start": "2026-03-15", "period_end": "2026-03-21"})],
    )
    registry.register(
        "计算痛风风险",
        "计算当前尿酸风险和发作风险。",
        risk.calculate_gout_risk,
        parameters=[
            _param("profile", "object", "用户档案。"),
            _param("logs", "array<object>", "每日健康记录。"),
            _param("labs", "array<object>", "化验结果。"),
            _param("attacks", "array<object>", "痛风发作记录。"),
            _param("symptom_logs", "array<object>", "部位症状记录。", required=False, default=None),
        ],
        returns=_returns("object", "风险等级、风险分数和解释。"),
        examples=[_example("基于当前上下文计算风险")],
    )
    registry.register(
        "识别痛风诱因",
        "从近期记录中识别诱因。",
        risk.detect_gout_triggers,
        parameters=[
            _param("logs", "array<object>", "每日健康记录。"),
            _param("window_days", "integer", "向前分析的天数。", required=False, default=14),
        ],
        returns=_returns("object", "诱因名称到出现次数的映射。"),
        examples=[_example("分析最近两周诱因", kwargs={"window_days": 14})],
    )
    registry.register(
        "识别异常指标",
        "识别最近一次记录中的异常指标。",
        risk.detect_abnormal_metrics,
        parameters=[
            _param("profile", "object", "用户档案。"),
            _param("latest_lab", "object", "最近一次化验结果。"),
            _param("latest_log", "object", "最近一次每日记录。"),
        ],
        returns=_returns("array<object>", "异常指标列表。"),
        examples=[_example("基于最新记录识别异常")],
    )
    registry.register(
        "预测发作趋势",
        "预测近期痛风发作趋势。",
        risk.predict_attack_trend,
        parameters=[
            _param("logs", "array<object>", "每日健康记录。"),
            _param("labs", "array<object>", "化验结果。"),
            _param("horizon_days", "integer", "向前预测天数。", required=False, default=7),
        ],
        returns=_returns("object", "预测风险等级和分数。"),
        examples=[_example("预测未来 7 天趋势")],
    )
    registry.register(
        "生成周报",
        "生成痛风管理周报。",
        reporting.build_weekly_report,
        parameters=[
            _param("profile", "object", "用户档案。"),
            _param("logs", "array<object>", "每日健康记录。"),
            _param("labs", "array<object>", "化验结果。"),
            _param("attacks", "array<object>", "痛风发作记录。"),
            _param("symptom_logs", "array<object>", "部位症状记录。", required=False, default=None),
        ],
        returns=_returns("object", "结构化周报对象。"),
        examples=[_example("基于当前上下文生成周报")],
    )
    registry.register(
        "生成月报",
        "生成痛风管理月报。",
        reporting.build_monthly_report,
        parameters=[
            _param("profile", "object", "用户档案。"),
            _param("logs", "array<object>", "每日健康记录。"),
            _param("labs", "array<object>", "化验结果。"),
            _param("attacks", "array<object>", "痛风发作记录。"),
        ],
        returns=_returns("object", "结构化月报对象。"),
        examples=[_example("基于当前上下文生成月报")],
    )
    registry.register(
        "导出报告",
        "把报告导出到本地文件。",
        lambda report_payload, report_type, format_name: reporting.export_report(project_root, report_payload, report_type, format_name),
        parameters=[
            _param("report_payload", "object", "待导出的报告内容。"),
            _param("report_type", "string", "报告类型。"),
            _param("format_name", "string", "导出格式，例如 json 或 html。"),
        ],
        returns=_returns("string", "导出文件路径。"),
        examples=[_example("导出 JSON 周报", kwargs={"report_payload": {"summary": "稳定"}, "report_type": "weekly", "format_name": "json"})],
    )
    registry.register(
        "调用本地痛风模型",
        "调用本地 HuatuoGPT 兼容模型。",
        llm.ask_local_gout_llm,
        parameters=[
            _param("question", "string", "用户问题或任务提示。"),
            _param("context", "object", "传给本地模型的结构化上下文。"),
        ],
        returns=_returns("object", "模型调用结果，包含成功状态、回答和错误信息。"),
        examples=[_example("结合上下文询问食物建议", kwargs={"question": "今晚能不能吃海鲜？", "context": {"risk_result": {"attack_risk_level": "Moderate"}}})],
    )
    registry.register(
        "获取本地模型状态",
        "查看本地模型接口配置。",
        llm.get_local_llm_status,
        returns=_returns("object", "本地模型的接口地址、模型名和超时设置。"),
        examples=[_example("查看本地模型连接状态")],
    )
    registry.register(
        "获取会话记忆",
        "读取最近的会话记忆记录。",
        lambda limit=20: data.get_session_memories(project_root, limit=limit, user_id=user_id),
        parameters=[_param("limit", "integer", "返回的最大条数。", required=False, default=20)],
        returns=_returns("array<object>", "最近的会话记忆列表。"),
        examples=[_example("读取最近 10 条会话记忆", kwargs={"limit": 10})],
    )
    registry.register(
        "记录会话记忆",
        "写入一条会话记忆。",
        lambda role, content, metadata=None: data.save_session_memory(project_root, role, content, metadata=metadata, user_id=user_id),
        parameters=[
            _param("role", "string", "消息角色，如 user 或 assistant。"),
            _param("content", "string", "会话内容。"),
            _param("metadata", "object", "附加元数据。", required=False, default=None),
        ],
        returns=_returns("integer", "新建会话记忆 ID。"),
        examples=[_example("记录一条用户消息", kwargs={"role": "user", "content": "今天脚趾有点疼"})],
    )
    return registry
