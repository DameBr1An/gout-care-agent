from __future__ import annotations

import os
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from gout_agent.skills.orchestrator import AppOrchestrator
from gout_agent.toolkit import ToolParameterSpec, ToolRegistry, build_default_tool_registry, serialize_tool_result


MCP_TOOL_NAME_MAP: dict[str, str] = {
    "获取用户档案": "get_user_profile",
    "更新用户档案": "update_user_profile",
    "记录日常健康": "log_daily_health_entry",
    "获取近期健康记录": "get_recent_health_entries",
    "记录化验结果": "log_lab_result",
    "获取化验历史": "get_lab_history",
    "记录痛风发作": "log_gout_attack",
    "获取发作历史": "get_attack_history",
    "添加药物方案": "add_medication_plan",
    "获取药物列表": "get_medications",
    "记录服药情况": "log_medication_taken",
    "获取服药依从性": "get_medication_adherence",
    "创建提醒": "create_reminder",
    "获取启用提醒": "list_active_reminders",
    "获取风险快照": "get_risk_snapshots",
    "保存风险快照": "save_risk_snapshot",
    "获取报告历史": "get_report_history",
    "保存报告": "save_report",
    "计算痛风风险": "calculate_gout_risk",
    "识别痛风诱因": "detect_gout_triggers",
    "识别异常指标": "detect_abnormal_metrics",
    "预测发作趋势": "predict_attack_trend",
    "生成周报": "build_weekly_report",
    "生成月报": "build_monthly_report",
    "导出报告": "export_report",
    "调用本地痛风模型": "ask_local_gout_llm",
    "获取本地模型状态": "get_local_llm_status",
    "获取会话记忆": "get_session_memories",
    "记录会话记忆": "save_session_memory",
    "获取长期记忆快照": "get_memory_snapshots",
    "生成长期记忆摘要": "build_long_term_memory",
}


def _resolve_project_root() -> Path:
    configured = os.getenv("GOUT_AGENT_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path.cwd().resolve()


def create_mcp_server(project_root: Path | None = None) -> FastMCP:
    root = project_root.resolve() if project_root else _resolve_project_root()
    registry = build_default_tool_registry(root)
    server = FastMCP(
        name="AI 痛风管理 Agent MCP Server",
        instructions=(
            "这是一个用于痛风与高尿酸血症长期管理的 MCP 服务器。"
            "优先通过 tools 读写本地数据、评估风险、生成报告和执行多步 Agent Loop。"
            "回答和建议仅用于健康管理，不替代医生诊断。"
        ),
        host="127.0.0.1",
        port=8787,
        streamable_http_path="/mcp",
        json_response=True,
    )

    _register_registry_tools(server, registry)
    _register_agent_tools(server, root)
    _register_resources(server, root)
    return server


def create_app(project_root: Path | None = None):
    return create_mcp_server(project_root).streamable_http_app()


def _register_registry_tools(server: FastMCP, registry: ToolRegistry) -> None:
    for internal_name, tool_spec in sorted(registry._tools.items(), key=lambda item: item[0]):  # noqa: SLF001
        public_name = MCP_TOOL_NAME_MAP.get(internal_name)
        if not public_name:
            continue
        server.add_tool(
            _make_registry_tool_callable(registry, internal_name, public_name, tool_spec.parameters),
            name=public_name,
            title=internal_name,
            description=f"{tool_spec.description}（内部工具名：{internal_name}）",
            annotations=_build_tool_annotations(internal_name),
            meta={"internal_tool_name": internal_name},
            structured_output=True,
        )


def _register_agent_tools(server: FastMCP, project_root: Path) -> None:
    @server.tool(
        name="run_agent_loop",
        title="运行 Agent Loop",
        description="执行完整的多步痛风管理 Agent Loop，返回技能路由、步骤轨迹和最终回答。",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def run_agent_loop(question: str, max_steps: int = 8) -> dict[str, Any]:
        orchestrator = AppOrchestrator(project_root)
        context = orchestrator.load_context()
        orchestrator.sync_daily_snapshot(context)
        result = orchestrator.run_agent_loop(question, context, max_steps=max_steps)
        return serialize_tool_result(result)

def _register_resources(server: FastMCP, project_root: Path) -> None:
    @server.resource(
        "gout://profile/current",
        name="current_profile",
        title="当前用户档案",
        description="返回当前用户的基础档案和健康档案摘要。",
        mime_type="application/json",
    )
    def current_profile() -> dict[str, Any]:
        registry = build_default_tool_registry(project_root)
        return {"profile": serialize_tool_result(registry.call("获取用户档案"))}

    @server.resource(
        "gout://memory/latest",
        name="latest_long_term_memory",
        title="最新长期记忆",
        description="返回最近一次保存的长期记忆快照。",
        mime_type="application/json",
    )
    def latest_memory() -> dict[str, Any]:
        registry = build_default_tool_registry(project_root)
        snapshots = registry.call("获取长期记忆快照", "long_term_memory", 1)
        return {"snapshots": serialize_tool_result(snapshots)}


def _make_registry_tool_callable(
    registry: ToolRegistry,
    internal_name: str,
    public_name: str,
    parameters: list[ToolParameterSpec],
):
    def wrapper(**kwargs: Any) -> dict[str, Any]:
        ordered_args: list[Any] = []
        for parameter in parameters:
            if parameter.name in kwargs:
                ordered_args.append(kwargs[parameter.name])
            elif parameter.required and parameter.default is None:
                raise ValueError(f"缺少必填参数：{parameter.name}")
            else:
                ordered_args.append(parameter.default)

        result = registry.call(internal_name, *ordered_args)
        return {
            "tool_name": public_name,
            "internal_tool_name": internal_name,
            "result": serialize_tool_result(result),
        }

    wrapper.__name__ = public_name
    wrapper.__qualname__ = public_name
    wrapper.__doc__ = f"MCP tool wrapper for {internal_name}"
    wrapper.__signature__ = Signature(  # type: ignore[attr-defined]
        parameters=[_build_signature_parameter(item) for item in parameters],
        return_annotation=dict[str, Any],
    )
    wrapper.__annotations__ = {item.name: _python_type_for_parameter(item) for item in parameters} | {"return": dict[str, Any]}
    return wrapper


def _build_signature_parameter(parameter: ToolParameterSpec) -> Parameter:
    default = Parameter.empty if parameter.required and parameter.default is None else parameter.default
    return Parameter(
        name=parameter.name,
        kind=Parameter.KEYWORD_ONLY,
        default=default,
        annotation=_python_type_for_parameter(parameter),
    )


def _python_type_for_parameter(parameter: ToolParameterSpec):
    type_map: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict[str, Any],
        "array<object>": list[dict[str, Any]],
        "array<string>": list[str],
    }
    annotation = type_map.get(parameter.type, Any)
    if not parameter.required:
        return annotation | None
    return annotation


def _build_tool_annotations(internal_name: str) -> ToolAnnotations:
    read_only_prefixes = ("获取", "计算", "识别", "预测", "生成", "调用")
    write_prefixes = ("更新", "记录", "添加", "创建", "保存", "导出")
    is_read_only = internal_name.startswith(read_only_prefixes)
    is_write = internal_name.startswith(write_prefixes)
    return ToolAnnotations(
        title=internal_name,
        readOnlyHint=is_read_only,
        destructiveHint=is_write,
        idempotentHint=is_read_only,
        openWorldHint=False,
    )


server = create_mcp_server()
app = server.streamable_http_app()


if __name__ == "__main__":
    server.run(transport="streamable-http")
