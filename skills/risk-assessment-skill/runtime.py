from __future__ import annotations

from typing import Any


def summarize_risk(context: dict) -> str:
    risk_data = context.get("risk_result") or {}
    return (
        f"当前尿酸风险为 {risk_data.get('uric_acid_risk_level_cn', '未知')}，"
        f"发作风险为 {risk_data.get('attack_risk_level_cn', '未知')}，"
        f"整体风险评分为 {risk_data.get('overall_risk_score', '-')}。"
        f"{risk_data.get('explanation') or ''}"
    )


def summarize_triggers(context: dict) -> str:
    trigger_items = context.get("trigger_summary") or []
    if not trigger_items:
        return "最近记录中还没有识别到明确的风险因素。"
    top = [f"{item.get('label', '未知因素')}（{item.get('count', 0)} 次）" for item in trigger_items[:3]]
    return f"近期最需要关注的风险因素包括：{'、'.join(top)}。"


def summarize_abnormal_items(context: dict) -> str:
    abnormal_items = context.get("abnormal_items") or []
    if not abnormal_items:
        return "最近一次数据中没有识别到明显异常。"
    return f"当前需要重点关注：{'、'.join(str(item) for item in abnormal_items[:4])}。"


def prepare(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(context or {})


def run(action: str | None = None, *args, **kwargs) -> Any:
    context = args[0] if args else kwargs.get("context", {})
    if action in {None, "summarize_risk"}:
        return summarize_risk(context)
    if action == "summarize_triggers":
        return summarize_triggers(context)
    if action == "summarize_abnormal_items":
        return summarize_abnormal_items(context)
    raise ValueError(f"risk-assessment-skill 不支持的运行动作：{action}")


def summarize(action: str | None = None, *args, **kwargs) -> str:
    return str(run(action, *args, **kwargs))


def persist(*args, **kwargs) -> None:
    return None
