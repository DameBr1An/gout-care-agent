from __future__ import annotations

from typing import Any, Callable


def execute_loop_tool(
    call_skill_tool: Callable[..., Any],
    route_name: str,
    tool_name: str,
    context: Any,
    observations: dict[str, Any],
) -> Any:
    if route_name == "profile":
        return call_skill_tool("profile", tool_name)
    if route_name == "reporting":
        return execute_reporting_loop_tool(call_skill_tool, tool_name, context, observations)
    if tool_name == "计算痛风风险":
        tool_route = "risk_assessment" if route_name == "risk_assessment" else "lifestyle_coach"
        return call_skill_tool(tool_route, tool_name, context.profile, context.logs, context.labs, context.attacks)
    if tool_name == "识别痛风诱因":
        tool_route = "risk_assessment" if route_name == "risk_assessment" else "lifestyle_coach"
        return call_skill_tool(tool_route, tool_name, context.logs, 14)
    if tool_name == "识别异常指标":
        latest_lab = context.labs.iloc[-1].to_dict() if not context.labs.empty else None
        latest_log = context.logs.iloc[-1].to_dict() if not context.logs.empty else None
        return call_skill_tool("risk_assessment", tool_name, context.profile, latest_lab, latest_log)
    if tool_name == "预测发作趋势":
        return call_skill_tool("risk_assessment", tool_name, context.logs, context.labs, 7)
    if tool_name in {"获取药物列表", "获取启用提醒"}:
        return call_skill_tool(route_name, tool_name)
    if tool_name == "获取服药依从性":
        return call_skill_tool(route_name, tool_name, 30)
    return None


def execute_reporting_loop_tool(
    call_skill_tool: Callable[..., Any],
    tool_name: str,
    context: Any,
    observations: dict[str, Any],
) -> Any:
    if tool_name in {"生成周报", "生成月报"}:
        return call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks, context.symptom_logs)
    report_payload = observations.get("生成周报") or observations.get("生成月报")
    if report_payload is None:
        raise RuntimeError("执行报告工具前缺少报告内容。")
    report_type = "monthly" if "生成月报" in observations else "weekly"
    if tool_name == "导出报告":
        return call_skill_tool("reporting", tool_name, report_payload, report_type, "json")
    if tool_name == "保存报告":
        period_start, period_end = report_payload["period"].split(" 至 ")
        return call_skill_tool("reporting", tool_name, report_type, report_payload, period_start, period_end)
    return None


def execute_reporting_plan(
    call_skill_tool: Callable[..., Any],
    plan: list[str],
    context: Any,
    format_name: str,
) -> dict[str, Any]:
    report_payload, path = None, None
    for tool_name in plan:
        if tool_name in {"生成周报", "生成月报"}:
            report_payload = call_skill_tool("reporting", tool_name, context.profile, context.logs, context.labs, context.attacks, context.symptom_logs)
        elif tool_name == "导出报告":
            if report_payload is None:
                raise RuntimeError("导出报告前缺少报告内容。")
            report_type = "monthly" if "月报" in plan[0] else "weekly"
            path = call_skill_tool("reporting", tool_name, report_payload, report_type, format_name)
        elif tool_name == "保存报告":
            if report_payload is None:
                raise RuntimeError("保存报告前缺少报告内容。")
            period_start, period_end = report_payload["period"].split(" 至 ")
            report_type = "monthly" if "月报" in plan[0] else "weekly"
            call_skill_tool("reporting", tool_name, report_type, report_payload, period_start, period_end)
    if report_payload is None:
        raise RuntimeError("未能生成报告。")
    return {"report_payload": report_payload, "path": path}
