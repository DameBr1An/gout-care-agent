from __future__ import annotations

from typing import Any


def explain_report(report_payload: dict, context: dict) -> str:
    latest = context.get("risk_result") or {}
    mean_water = report_payload.get("mean_water_ml")
    water_text = f"{mean_water} mL" if mean_water is not None else "暂无数据"
    parts = [
        (
            f"本期报告覆盖 {report_payload.get('period') or '当前周期'}。"
            f"期间共记录 {report_payload.get('entries') or 0} 条健康数据，"
            f"平均饮水约 {water_text}，"
            f"当前发作风险为 {latest.get('attack_risk_level_cn', '未知')}。"
        ),
        "建议继续围绕规律饮水、避免已识别风险因素和按时用药来保持稳定。",
        "如果疼痛、红肿或异常指标持续加重，请及时线下就医。",
    ]
    return " ".join(parts)


def prepare(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(context or {})


def run(action: str | None = None, *args, **kwargs) -> Any:
    if action in {None, "explain_report"}:
        return explain_report(*args, **kwargs)
    raise ValueError(f"report-explanation-skill 不支持的运行动作：{action}")


def summarize(action: str | None = None, *args, **kwargs) -> str:
    if action in {None, "explain_report"}:
        return explain_report(*args, **kwargs)
    raise ValueError(f"report-explanation-skill 不支持的摘要动作：{action}")


def persist(*args, **kwargs) -> None:
    return None
