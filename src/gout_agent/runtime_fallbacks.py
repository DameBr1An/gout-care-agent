from __future__ import annotations

from typing import Any, Callable


TRIGGER_LABELS = {
    "alcohol": "饮酒",
    "beer": "啤酒",
    "spirits": "烈酒",
    "seafood": "海鲜",
    "shellfish": "贝类",
    "organ_meat": "动物内脏",
    "red_meat": "红肉",
    "hotpot": "火锅",
    "barbecue": "烧烤",
    "sugary_drinks": "含糖饮料",
    "low_hydration": "饮水不足",
    "missed_medication": "未按时服药",
}


def summarize_profile(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    if profile.get("name"):
        parts.append("当前档案用户：%s。" % profile["name"])
    if profile.get("target_uric_acid"):
        parts.append("目标尿酸为 %s。" % profile["target_uric_acid"])
    conditions: list[str] = []
    if profile.get("has_gout_diagnosis"):
        conditions.append("已确诊痛风")
    if profile.get("has_hyperuricemia"):
        conditions.append("高尿酸血症")
    if profile.get("has_ckd"):
        conditions.append("慢性肾病")
    if profile.get("has_hypertension"):
        conditions.append("高血压")
    if profile.get("has_diabetes"):
        conditions.append("糖尿病")
    if conditions:
        parts.append("当前长期健康背景包括：%s。" % "、".join(conditions))
    if profile.get("doctor_advice"):
        parts.append("AI 管理意见：%s。" % profile["doctor_advice"])
    return " ".join(parts) if parts else "当前还没有完善的长期健康档案，建议先补充目标尿酸、基础病和 AI 管理意见。"


def build_fallback_answer(
    route_name: str,
    question: str,
    context_payload: dict[str, Any],
    observations: dict[str, Any],
    *,
    label_risk: Callable[[str], str],
    get_profile: Callable[[], dict[str, Any]],
    call_reporting_report: Callable[[], Any],
    get_skill_runtime: Callable[[str], Any],
) -> str:
    runtime = get_skill_runtime(route_name)
    if route_name == "profile":
        return summarize_profile(observations.get("获取用户档案") or get_profile())
    if route_name == "risk_assessment":
        risk_payload = observations.get("计算痛风风险")
        trigger_payload = observations.get("识别痛风诱因")
        abnormal_payload = observations.get("识别异常指标")
        if risk_payload is not None:
            local_context = {
                "risk_result": {
                    "uric_acid_risk_level_cn": label_risk(risk_payload.uric_acid_risk_level),
                    "attack_risk_level_cn": label_risk(risk_payload.attack_risk_level),
                    "overall_risk_score": risk_payload.overall_risk_score,
                    "explanation": risk_payload.explanation,
                },
                "trigger_summary": [
                    {"label": TRIGGER_LABELS.get(name, name), "count": count}
                    for name, count in list((trigger_payload or {}).items())[:5]
                ],
                "abnormal_items": [item.message for item in (abnormal_payload or [])],
            }
            return "\n".join(
                [
                    runtime.summarize("summarize_risk", local_context),
                    runtime.summarize("summarize_triggers", local_context),
                    runtime.summarize("summarize_abnormal_items", local_context),
                ]
            )
        return runtime.summarize("summarize_risk", context_payload) + "\n" + runtime.summarize("summarize_triggers", context_payload)
    if route_name == "lifestyle_coach":
        return runtime.summarize("answer_food_question", question, context_payload)
    if route_name == "medication_followup":
        return runtime.summarize("summarize_medication_and_reminders", context_payload)
    if route_name == "reporting":
        report_payload = observations.get("生成周报") or observations.get("生成月报") or call_reporting_report()
        return runtime.summarize("explain_report", report_payload, context_payload)
    risk_runtime = get_skill_runtime("risk_assessment")
    lifestyle_runtime = get_skill_runtime("lifestyle_coach")
    return risk_runtime.summarize("summarize_risk", context_payload) + "\n" + lifestyle_runtime.summarize("build_daily_lifestyle_guidance", context_payload)
