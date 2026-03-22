from __future__ import annotations

import pandas as pd


class RiskResult(object):
    def __init__(
        self,
        uric_acid_risk_level,
        attack_risk_level,
        overall_risk_score,
        abnormal_items,
        top_risk_factors,
        trend_direction,
        explanation,
        hydration_advice,
        diet_advice,
        exercise_advice,
        behavior_goal,
    ):
        self.uric_acid_risk_level = uric_acid_risk_level
        self.attack_risk_level = attack_risk_level
        self.overall_risk_score = overall_risk_score
        self.abnormal_items = abnormal_items
        self.top_risk_factors = top_risk_factors
        self.trend_direction = trend_direction
        self.explanation = explanation
        self.hydration_advice = hydration_advice
        self.diet_advice = diet_advice
        self.exercise_advice = exercise_advice
        self.behavior_goal = behavior_goal


class AbnormalMetric(object):
    def __init__(self, metric, severity, message):
        self.metric = metric
        self.severity = severity
        self.message = message


HIGH_RISK_TRIGGERS = {
    "beer",
    "alcohol",
    "spirits",
    "seafood",
    "shellfish",
    "organ_meat",
    "red_meat",
    "hotpot",
    "barbecue",
    "sugary_drinks",
}

RISK_LEVEL_LABELS = {
    "Low": "低",
    "Moderate": "中",
    "High": "高",
}


def _as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value):
    return str(value or "").strip().lower()


def _risk_bucket(score, low_cutoff, high_cutoff):
    if score >= high_cutoff:
        return "High"
    if score >= low_cutoff:
        return "Moderate"
    return "Low"


def _risk_label(level):
    return RISK_LEVEL_LABELS.get(level, str(level))


def detect_abnormal_metrics(profile, latest_lab, latest_log):
    abnormal = []
    target_uric_acid = _as_float(profile.get("target_uric_acid"), 360.0)

    if latest_lab:
        uric_acid = _as_float(latest_lab.get("uric_acid"))
        if uric_acid >= 540:
            abnormal.append(AbnormalMetric("uric_acid", "high", "尿酸明显升高，近期发作风险偏高。"))
        elif uric_acid >= max(target_uric_acid, 420):
            abnormal.append(AbnormalMetric("uric_acid", "moderate", "尿酸高于目标范围。"))

        egfr = _as_float(latest_lab.get("egfr"))
        if egfr and egfr < 60:
            abnormal.append(AbnormalMetric("egfr", "high", "eGFR 低于 60，建议关注肾功能变化。"))

        creatinine = _as_float(latest_lab.get("creatinine"))
        if creatinine and creatinine > 1.3:
            abnormal.append(AbnormalMetric("creatinine", "moderate", "肌酐偏高，建议结合肾功能复查。"))

        alt = _as_float(latest_lab.get("alt"))
        ast = _as_float(latest_lab.get("ast"))
        if alt and alt > 50:
            abnormal.append(AbnormalMetric("alt", "moderate", "ALT 升高，建议关注肝功能和用药情况。"))
        if ast and ast > 40:
            abnormal.append(AbnormalMetric("ast", "moderate", "AST 升高，建议结合近期化验继续观察。"))

    if latest_log:
        pain_score = int(_as_float(latest_log.get("pain_score")))
        if pain_score >= 7:
            abnormal.append(AbnormalMetric("pain_score", "high", "疼痛评分较高，如伴红肿热痛建议尽快就医。"))
        elif pain_score >= 4:
            abnormal.append(AbnormalMetric("pain_score", "moderate", "疼痛评分偏高，需要继续观察症状变化。"))

        water_ml = _as_float(latest_log.get("water_ml"))
        if water_ml and water_ml < 1500:
            abnormal.append(AbnormalMetric("water_ml", "moderate", "饮水量偏少，建议及时补水。"))

    return abnormal


def detect_gout_triggers(logs, window_days=14):
    if logs.empty:
        return {}

    frame = logs.copy().tail(window_days)
    trigger_counts = {}

    for _, row in frame.iterrows():
        alcohol_text = _normalize_text(row.get("alcohol_intake"))
        if alcohol_text and alcohol_text not in {"none", "0", "no"}:
            trigger_counts["alcohol"] = trigger_counts.get("alcohol", 0) + 1

        combined = " ".join(
            [
                _normalize_text(row.get("diet_notes")),
                _normalize_text(row.get("symptom_notes")),
                _normalize_text(row.get("free_text")),
            ]
        )
        for keyword in HIGH_RISK_TRIGGERS:
            if keyword.replace("_", " ") in combined or keyword in combined:
                trigger_counts[keyword] = trigger_counts.get(keyword, 0) + 1

        if _as_float(row.get("water_ml")) and _as_float(row.get("water_ml")) < 1500:
            trigger_counts["low_hydration"] = trigger_counts.get("low_hydration", 0) + 1

        if int(_as_float(row.get("medication_taken_flag"))) == 0:
            trigger_counts["missed_medication"] = trigger_counts.get("missed_medication", 0) + 1

    return dict(sorted(trigger_counts.items(), key=lambda item: item[1], reverse=True))


def explain_risk_change(latest_score, previous_score):
    if previous_score is None:
        return "stable", "当前还缺少足够的历史数据，先以最新状态为主。"
    delta = latest_score - previous_score
    if delta >= 2:
        return "up", "与上一阶段相比，风险有所上升，建议优先检查诱因和近期症状。"
    if delta <= -2:
        return "down", "与上一阶段相比，风险有所下降，说明最近管理方向整体有效。"
    return "stable", "与上一阶段相比，风险整体较稳定。"


def predict_attack_trend(logs, labs, horizon_days=7):
    latest_log = logs.iloc[-1].to_dict() if not logs.empty else {}
    latest_lab = labs.iloc[-1].to_dict() if not labs.empty else {}
    uric_acid = _as_float(latest_lab.get("uric_acid"))
    pain_score = _as_float(latest_log.get("pain_score"))
    hydration = _as_float(latest_log.get("water_ml"))
    score = 0

    if uric_acid >= 540:
        score += 4
    elif uric_acid >= 420:
        score += 2
    if pain_score >= 7:
        score += 4
    elif pain_score >= 4:
        score += 2
    if hydration and hydration < 1500:
        score += 2

    trend = _risk_bucket(score, 4, 8)
    return {
        "horizon_days": horizon_days,
        "predicted_attack_risk": trend,
        "predicted_score": score,
    }


def calculate_gout_risk(profile, logs, labs, attacks):
    latest_log = logs.iloc[-1].to_dict() if not logs.empty else {}
    latest_lab = labs.iloc[-1].to_dict() if not labs.empty else {}
    target_uric_acid = _as_float(profile.get("target_uric_acid"), 360.0)

    uric_acid = _as_float(latest_lab.get("uric_acid"))
    water_ml = _as_float(latest_log.get("water_ml"))
    pain_score = int(_as_float(latest_log.get("pain_score")))
    steps = int(_as_float(latest_log.get("steps")))
    exercise_minutes = int(_as_float(latest_log.get("exercise_minutes")))
    sleep_hours = _as_float(latest_log.get("sleep_hours"))
    medication_taken = bool(int(_as_float(latest_log.get("medication_taken_flag")))) if latest_log else False
    recent_trigger_counts = detect_gout_triggers(logs, window_days=14)

    uric_score = 0
    attack_score = 0
    factors = []

    if uric_acid >= max(540, target_uric_acid + 120):
        uric_score += 5
        attack_score += 3
        factors.append("尿酸明显高于目标")
    elif uric_acid >= max(420, target_uric_acid):
        uric_score += 3
        attack_score += 2
        factors.append("尿酸仍高于目标")

    if water_ml and water_ml < 1500:
        uric_score += 1
        attack_score += 2
        factors.append("饮水不足")

    if pain_score >= 7:
        attack_score += 5
        factors.append("当前疼痛较重")
    elif pain_score >= 4:
        attack_score += 3
        factors.append("当前有明显疼痛")

    if recent_trigger_counts.get("alcohol"):
        uric_score += 1
        attack_score += 2
        factors.append("近期有饮酒暴露")

    if any(trigger in recent_trigger_counts for trigger in ["seafood", "shellfish", "organ_meat", "red_meat", "hotpot", "barbecue", "sugary_drinks"]):
        uric_score += 1
        attack_score += 1
        factors.append("近期有高风险饮食")

    if not medication_taken and latest_log:
        attack_score += 1
        factors.append("今天未记录服药")

    if sleep_hours and sleep_hours < 6:
        attack_score += 1
        factors.append("睡眠不足")

    if steps >= 8000 or exercise_minutes >= 30:
        attack_score = max(0, attack_score - 1)

    if not attacks.empty:
        attack_count = len(attacks)
        if attack_count >= 3:
            attack_score += 2
            factors.append("近阶段发作频繁")
        elif attack_count >= 1:
            attack_score += 1
            factors.append("近阶段有发作史")

    overall_score = uric_score + attack_score
    uric_level = _risk_bucket(uric_score, 3, 5)
    attack_level = _risk_bucket(attack_score, 4, 8)

    abnormal_items = [item.message for item in detect_abnormal_metrics(profile, latest_lab, latest_log)]

    previous_logs = logs.iloc[:-1] if len(logs) > 1 else pd.DataFrame()
    previous_labs = labs.iloc[:-1] if len(labs) > 1 else pd.DataFrame()
    previous_uric = _as_float(previous_labs.iloc[-1].get("uric_acid")) if not previous_labs.empty else None
    previous_pain = _as_float(previous_logs.iloc[-1].get("pain_score")) if not previous_logs.empty else None
    previous_score = None
    if previous_uric is not None or previous_pain is not None:
        previous_score = 0
        if previous_uric and previous_uric >= 420:
            previous_score += 3
        if previous_uric and previous_uric >= 540:
            previous_score += 2
        if previous_pain and previous_pain >= 4:
            previous_score += 3
        if previous_pain and previous_pain >= 7:
            previous_score += 2
    trend_direction, trend_explanation = explain_risk_change(overall_score, previous_score)

    uric_label = _risk_label(uric_level)
    attack_label = _risk_label(attack_level)
    if factors:
        explanation = "当前尿酸风险为 %s，发作风险为 %s，主要原因包括 %s。%s" % (
            uric_label,
            attack_label,
            "、".join(factors[:3]),
            trend_explanation,
        )
    else:
        explanation = "当前尿酸风险为 %s，发作风险为 %s。%s" % (
            uric_label,
            attack_label,
            trend_explanation,
        )

    hydration_advice = (
        "建议今天把饮水提高到 2000 到 2500 mL，分次补水更稳妥。"
        if not water_ml or water_ml < 1500
        else "饮水表现还可以，继续保持规律补水。"
    )
    diet_advice = (
        "近期有诱因暴露，建议先避免酒精、海鲜、动物内脏和高糖饮料。"
        if recent_trigger_counts
        else "近期诱因不突出，继续保持清淡、规律的饮食结构。"
    )
    exercise_advice = (
        "当前疼痛较明显，今天以轻活动和休息为主，避免高强度运动。"
        if pain_score >= 4
        else "今天可安排 20 到 30 分钟轻中等强度活动，如快走或拉伸。"
    )
    behavior_goal = (
        "今天优先完成补水、避免诱因，并关注疼痛或红肿是否加重。"
        if overall_score >= 8
        else "今天继续保持补水、规律作息和按时记录。"
    )

    return RiskResult(
        uric_level,
        attack_level,
        overall_score,
        abnormal_items,
        factors[:5],
        trend_direction,
        explanation,
        hydration_advice,
        diet_advice,
        exercise_advice,
        behavior_goal,
    )
