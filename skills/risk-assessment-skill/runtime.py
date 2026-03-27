from __future__ import annotations


def summarize_risk(context: dict) -> str:
    risk_data = context["risk_result"]
    return (
        f"当前尿酸风险为 {risk_data['uric_acid_risk_level_cn']}，"
        f"发作风险为 {risk_data['attack_risk_level_cn']}，"
        f"整体风险评分为 {risk_data['overall_risk_score']}。"
        f"{risk_data['explanation']}"
    )


def summarize_triggers(context: dict) -> str:
    trigger_items = context.get("trigger_summary") or []
    if not trigger_items:
        return "最近记录中没有识别到明确诱因。"
    top = [f"{item['label']}（{item['count']} 次）" for item in trigger_items[:3]]
    return f"最近需要重点关注的诱因包括：{'、'.join(top)}。"


def summarize_abnormal_items(context: dict) -> str:
    abnormal_items = context.get("abnormal_items") or []
    if not abnormal_items:
        return "最近一次数据中没有识别到明显异常指标。"
    return f"当前需要重点关注的异常包括：{'；'.join(abnormal_items)}。"
