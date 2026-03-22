from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

import pandas as pd


TRIGGER_KEYWORDS = {
    "beer": "啤酒",
    "alcohol": "饮酒",
    "seafood": "海鲜",
    "shellfish": "贝类",
    "hotpot": "火锅",
    "barbecue": "烧烤",
    "organ_meat": "动物内脏",
    "red_meat": "红肉",
    "sugary_drinks": "含糖饮料",
    "low_hydration": "饮水不足",
    "missed_medication": "未按时服药",
}


def build_long_term_memory(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "user_preferences": build_user_preferences(profile, logs),
        "ai_advice_summary": summarize_ai_advice(profile),
        "attack_patterns": analyze_attack_patterns(attacks),
        "behavior_portraits": {
            "7d": build_behavior_portrait(logs, labs, attacks, 7),
            "30d": build_behavior_portrait(logs, labs, attacks, 30),
            "90d": build_behavior_portrait(logs, labs, attacks, 90),
        },
        "gout_management_twin_profile": build_gout_management_twin_profile(profile, logs, labs, attacks),
        "updated_at": pd.Timestamp.now().isoformat(),
    }


def build_gout_management_twin_profile(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
) -> dict[str, Any]:
    top_triggers = _build_top_triggers(logs, attacks)
    trigger_patterns = _build_trigger_patterns(logs, attacks)
    risk_windows = _build_risk_windows(logs, attacks)
    behavior_patterns = _build_behavior_patterns(logs, labs)
    management_stability = _build_management_stability(logs, attacks)
    current_shortcomings = _build_current_shortcomings(logs, labs, attacks)

    summary_parts: list[str] = []
    if top_triggers:
        summary_parts.append("当前最需要关注的诱因为：%s。" % "、".join(item["label"] for item in top_triggers[:3]))
    if risk_windows:
        summary_parts.append("发作前高风险窗口主要集中在：%s。" % "；".join(item["label"] for item in risk_windows[:2]))
    if management_stability.get("stability_score") is not None:
        summary_parts.append("当前管理稳定度评分为 %s/100。" % management_stability["stability_score"])
    if current_shortcomings:
        summary_parts.append("近期最需要补强的是：%s。" % "、".join(current_shortcomings[:3]))

    if not summary_parts:
        summary_parts.append("当前记录量还不足，建议继续连续记录饮水、饮食、症状、用药和化验数据。")

    return {
        "summary": " ".join(summary_parts),
        "top_triggers": top_triggers,
        "trigger_patterns": trigger_patterns,
        "risk_windows": risk_windows,
        "behavior_patterns": behavior_patterns,
        "management_stability": management_stability,
        "current_shortcomings": current_shortcomings,
        "updated_at": pd.Timestamp.now().isoformat(),
    }


def build_user_preferences(profile: dict[str, Any], logs: pd.DataFrame) -> dict[str, Any]:
    diet_counter: Counter[str] = Counter()
    alcohol_counter: Counter[str] = Counter()

    if not logs.empty:
        for _, row in logs.iterrows():
            alcohol_value = str(row.get("alcohol_intake") or "").strip().lower()
            if alcohol_value and alcohol_value != "none":
                alcohol_counter[alcohol_value] += 1

            diet_text = " ".join(str(row.get(field) or "") for field in ("diet_notes", "free_text", "symptom_notes"))
            for token in _extract_preference_tokens(diet_text):
                diet_counter[token] += 1

    hydration_avg = _safe_mean(logs, "water_ml")
    step_avg = _safe_mean(logs, "steps")
    exercise_avg = _safe_mean(logs, "exercise_minutes")

    preferred_foods = [item for item, _ in diet_counter.most_common(5)]
    preferred_alcohol = alcohol_counter.most_common(1)[0][0] if alcohol_counter else None

    summary_parts: list[str] = []
    if preferred_foods:
        summary_parts.append("近期常见饮食关键词：%s" % "、".join(preferred_foods[:3]))
    if preferred_alcohol:
        summary_parts.append("近期最常见饮酒类型：%s" % preferred_alcohol)
    if hydration_avg is not None:
        summary_parts.append("平均饮水约 %.0f mL/天" % hydration_avg)
    if step_avg is not None:
        summary_parts.append("平均步数约 %.0f 步/天" % step_avg)
    if exercise_avg is not None and exercise_avg > 0:
        summary_parts.append("平均运动约 %.0f 分钟/天" % exercise_avg)
    if profile.get("target_uric_acid"):
        summary_parts.append("目标尿酸 %s" % profile["target_uric_acid"])

    return {
        "preferred_foods": preferred_foods,
        "preferred_alcohol": preferred_alcohol,
        "average_water_ml": hydration_avg,
        "average_steps": step_avg,
        "average_exercise_minutes": exercise_avg,
        "summary": "；".join(summary_parts) if summary_parts else "暂无足够记录总结长期偏好。",
    }


def summarize_ai_advice(profile: dict[str, Any]) -> dict[str, Any]:
    advice = str(profile.get("doctor_advice") or "").strip()
    target = profile.get("target_uric_acid")
    if not advice and not target:
        return {
            "summary": "暂无明确医生建议摘要。",
            "source_text": "",
            "target_uric_acid": target,
        }

    normalized = advice.replace("\n", "。")
    sentences = [part.strip() for part in normalized.split("。") if part.strip()]
    summary = "；".join(sentences[:2]) if sentences else "请围绕目标尿酸和长期管理持续随访。"
    if target:
        summary = "目标尿酸 %s；%s" % (target, summary)

    return {
        "summary": summary,
        "source_text": advice,
        "target_uric_acid": target,
    }


def analyze_attack_patterns(attacks: pd.DataFrame) -> dict[str, Any]:
    if attacks.empty:
        return {
            "attack_count_180d": 0,
            "common_joint_site": None,
            "common_trigger": None,
            "average_pain_score": None,
            "average_interval_days": None,
            "summary": "最近没有记录到明确的痛风发作模式。",
        }

    frame = attacks.copy()
    frame["attack_date"] = pd.to_datetime(frame["attack_date"], errors="coerce")
    frame = frame.dropna(subset=["attack_date"]).sort_values("attack_date")
    attack_count = int(len(frame))

    common_joint = _most_common_value(frame, "joint_site")
    common_trigger = _most_common_value(frame, "suspected_trigger")
    average_pain = _safe_mean(frame, "pain_score")
    average_interval = None
    if attack_count >= 2:
        diffs = frame["attack_date"].diff().dropna().dt.days
        average_interval = float(diffs.mean()) if not diffs.empty else None

    summary_parts = ["近 180 天记录到 %s 次发作" % attack_count]
    if common_joint:
        summary_parts.append("常见部位为 %s" % common_joint)
    if common_trigger:
        summary_parts.append("常见诱因为 %s" % common_trigger)
    if average_pain is not None:
        summary_parts.append("平均疼痛评分 %.1f" % average_pain)
    if average_interval is not None:
        summary_parts.append("平均发作间隔约 %.0f 天" % average_interval)

    return {
        "attack_count_180d": attack_count,
        "common_joint_site": common_joint,
        "common_trigger": common_trigger,
        "average_pain_score": average_pain,
        "average_interval_days": average_interval,
        "summary": "；".join(summary_parts),
    }


def build_behavior_portrait(
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    window_days: int,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(date.today() - timedelta(days=max(window_days - 1, 0)))
    log_window = logs.copy()
    if not log_window.empty and "log_date" in log_window.columns:
        log_window["log_date"] = pd.to_datetime(log_window["log_date"], errors="coerce")
        log_window = log_window.loc[log_window["log_date"] >= cutoff]

    lab_window = labs.copy()
    if not lab_window.empty and "test_date" in lab_window.columns:
        lab_window["test_date"] = pd.to_datetime(lab_window["test_date"], errors="coerce")
        lab_window = lab_window.loc[lab_window["test_date"] >= cutoff]

    attack_window = attacks.copy()
    if not attack_window.empty and "attack_date" in attack_window.columns:
        attack_window["attack_date"] = pd.to_datetime(attack_window["attack_date"], errors="coerce")
        attack_window = attack_window.loc[attack_window["attack_date"] >= cutoff]

    alcohol_days = 0
    if not log_window.empty and "alcohol_intake" in log_window.columns:
        alcohol_days = int((log_window["alcohol_intake"].fillna("").astype(str).str.lower() != "none").sum())

    portrait = {
        "window_days": window_days,
        "days_with_logs": int(len(log_window)),
        "average_water_ml": _safe_mean(log_window, "water_ml"),
        "average_steps": _safe_mean(log_window, "steps"),
        "average_exercise_minutes": _safe_mean(log_window, "exercise_minutes"),
        "average_sleep_hours": _safe_mean(log_window, "sleep_hours"),
        "alcohol_days": alcohol_days,
        "pain_days": int((pd.to_numeric(log_window.get("pain_score"), errors="coerce").fillna(0) > 0).sum()) if not log_window.empty and "pain_score" in log_window.columns else 0,
        "medication_taken_rate": _binary_rate(log_window, "medication_taken_flag"),
        "latest_uric_acid": _latest_value(lab_window, "test_date", "uric_acid"),
        "attack_count": int(len(attack_window)),
    }

    summary_parts = ["近 %s 天有 %s 天记录" % (window_days, portrait["days_with_logs"])]
    if portrait["average_water_ml"] is not None:
        summary_parts.append("平均饮水 %.0f mL/天" % portrait["average_water_ml"])
    if portrait["average_steps"] is not None:
        summary_parts.append("平均步数 %.0f 步/天" % portrait["average_steps"])
    if portrait["alcohol_days"]:
        summary_parts.append("饮酒 %s 天" % portrait["alcohol_days"])
    if portrait["pain_days"]:
        summary_parts.append("疼痛 %s 天" % portrait["pain_days"])
    if portrait["attack_count"]:
        summary_parts.append("发作 %s 次" % portrait["attack_count"])
    portrait["summary"] = "；".join(summary_parts)
    return portrait


def _build_top_triggers(logs: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()

    if not logs.empty:
        for _, row in logs.iterrows():
            diet_text = " ".join(str(row.get(field) or "").lower() for field in ("diet_notes", "free_text", "symptom_notes"))
            for token, label in TRIGGER_KEYWORDS.items():
                if token in diet_text or label in diet_text:
                    counter[label] += 1

            alcohol = str(row.get("alcohol_intake") or "").strip().lower()
            if alcohol and alcohol != "none":
                counter["饮酒"] += 1

            water_value = pd.to_numeric(pd.Series([row.get("water_ml")]), errors="coerce").iloc[0]
            if pd.notna(water_value) and float(water_value) < 1500:
                counter["饮水不足"] += 1

            taken = pd.to_numeric(pd.Series([row.get("medication_taken_flag")]), errors="coerce").iloc[0]
            if pd.notna(taken) and float(taken) == 0:
                counter["未按时服药"] += 1

    if not attacks.empty and "suspected_trigger" in attacks.columns:
        for value in attacks["suspected_trigger"].dropna().astype(str):
            text = value.strip()
            if text:
                counter[text] += 2

    total = sum(counter.values()) or 1
    return [
        {"label": label, "count": count, "confidence": round(count / total, 2)}
        for label, count in counter.most_common(5)
    ]


def _build_trigger_patterns(logs: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    if attacks.empty:
        return patterns

    attack_frame = attacks.copy()
    attack_frame["attack_date"] = pd.to_datetime(attack_frame["attack_date"], errors="coerce")
    log_frame = logs.copy()
    if not log_frame.empty and "log_date" in log_frame.columns:
        log_frame["log_date"] = pd.to_datetime(log_frame["log_date"], errors="coerce")

    for _, attack in attack_frame.dropna(subset=["attack_date"]).iterrows():
        attack_date = attack["attack_date"]
        if log_frame.empty:
            continue
        recent_logs = log_frame.loc[(log_frame["log_date"] < attack_date) & (log_frame["log_date"] >= attack_date - pd.Timedelta(days=3))]
        labels: list[str] = []
        if not recent_logs.empty:
            if "water_ml" in recent_logs.columns:
                water_mean = _safe_mean(recent_logs, "water_ml")
                if water_mean is not None and water_mean < 1500:
                    labels.append("低饮水")
            if "alcohol_intake" in recent_logs.columns:
                if (recent_logs["alcohol_intake"].fillna("").astype(str).str.lower() != "none").any():
                    labels.append("饮酒")
            text = " ".join(recent_logs.get("diet_notes", pd.Series(dtype=str)).fillna("").astype(str))
            if "海鲜" in text:
                labels.append("海鲜")
            if "火锅" in text:
                labels.append("火锅")
            if "烧烤" in text:
                labels.append("烧烤")
        if labels:
            patterns.append(
                {
                    "attack_date": attack_date.date().isoformat(),
                    "pattern": " + ".join(dict.fromkeys(labels)),
                    "joint_site": attack.get("joint_site"),
                }
            )
    return patterns[:5]


def _build_risk_windows(logs: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    if attacks.empty:
        return []

    windows: list[dict[str, Any]] = []
    attack_frame = attacks.copy()
    attack_frame["attack_date"] = pd.to_datetime(attack_frame["attack_date"], errors="coerce")
    log_frame = logs.copy()
    if not log_frame.empty and "log_date" in log_frame.columns:
        log_frame["log_date"] = pd.to_datetime(log_frame["log_date"], errors="coerce")

    for _, attack in attack_frame.dropna(subset=["attack_date"]).iterrows():
        attack_date = attack["attack_date"]
        if log_frame.empty:
            continue
        window_24h = log_frame.loc[(log_frame["log_date"] < attack_date) & (log_frame["log_date"] >= attack_date - pd.Timedelta(days=1))]
        window_72h = log_frame.loc[(log_frame["log_date"] < attack_date) & (log_frame["log_date"] >= attack_date - pd.Timedelta(days=3))]
        if not window_24h.empty:
            windows.append({"label": "发作前 24 小时", "attack_date": attack_date.date().isoformat(), "log_count": int(len(window_24h))})
        elif not window_72h.empty:
            windows.append({"label": "发作前 72 小时", "attack_date": attack_date.date().isoformat(), "log_count": int(len(window_72h))})
    return windows[:5]


def _build_behavior_patterns(logs: pd.DataFrame, labs: pd.DataFrame) -> dict[str, Any]:
    patterns: dict[str, Any] = {
        "hydration_pattern": "暂无足够数据",
        "alcohol_pattern": "暂无足够数据",
        "diet_pattern": "暂无足够数据",
        "sleep_pattern": "暂无足够数据",
        "exercise_pattern": "暂无足够数据",
        "weekend_variation": "暂无足够数据",
    }

    if logs.empty:
        return patterns

    frame = logs.copy()
    if "log_date" in frame.columns:
        frame["log_date"] = pd.to_datetime(frame["log_date"], errors="coerce")
        frame["weekday"] = frame["log_date"].dt.weekday

    water_mean = _safe_mean(frame, "water_ml")
    if water_mean is not None:
        patterns["hydration_pattern"] = "平均饮水约 %.0f mL/天" % water_mean

    if "alcohol_intake" in frame.columns:
        alcohol_days = int((frame["alcohol_intake"].fillna("").astype(str).str.lower() != "none").sum())
        patterns["alcohol_pattern"] = "近阶段记录到 %s 天饮酒" % alcohol_days

    diet_text = " ".join(frame.get("diet_notes", pd.Series(dtype=str)).fillna("").astype(str))
    if diet_text:
        diet_keywords = [token for token in ["海鲜", "火锅", "烧烤", "红肉", "啤酒"] if token in diet_text]
        patterns["diet_pattern"] = "常见饮食暴露：%s" % ("、".join(diet_keywords) if diet_keywords else "暂无明显高风险关键词")

    sleep_mean = _safe_mean(frame, "sleep_hours")
    if sleep_mean is not None:
        patterns["sleep_pattern"] = "平均睡眠约 %.1f 小时/天" % sleep_mean

    exercise_mean = _safe_mean(frame, "exercise_minutes")
    if exercise_mean is not None:
        patterns["exercise_pattern"] = "平均运动约 %.0f 分钟/天" % exercise_mean

    if "weekday" in frame.columns:
        weekend = frame.loc[frame["weekday"] >= 5]
        weekday = frame.loc[frame["weekday"] < 5]
        weekend_water = _safe_mean(weekend, "water_ml")
        weekday_water = _safe_mean(weekday, "water_ml")
        if weekend_water is not None and weekday_water is not None:
            patterns["weekend_variation"] = "周末平均饮水 %.0f mL/天，工作日平均饮水 %.0f mL/天" % (weekend_water, weekday_water)

    latest_uric = _latest_value(labs, "test_date", "uric_acid")
    if latest_uric is not None:
        patterns["latest_uric_acid"] = latest_uric

    return patterns


def _build_management_stability(logs: pd.DataFrame, attacks: pd.DataFrame) -> dict[str, Any]:
    log_days = int(len(logs))
    medication_rate = _binary_rate(logs, "medication_taken_flag")
    pain_days = int((pd.to_numeric(logs.get("pain_score"), errors="coerce").fillna(0) > 0).sum()) if not logs.empty and "pain_score" in logs.columns else 0
    attack_count = int(len(attacks))

    score = 100
    if log_days < 7:
        score -= 20
    if medication_rate is not None and medication_rate < 70:
        score -= 20
    if pain_days >= 3:
        score -= 15
    if attack_count >= 1:
        score -= min(attack_count * 10, 30)
    water_mean = _safe_mean(logs, "water_ml")
    if water_mean is not None and water_mean < 1500:
        score -= 10
    score = max(score, 0)

    if score >= 80:
        level = "稳定"
    elif score >= 60:
        level = "轻度波动"
    else:
        level = "需要重点关注"

    return {
        "stability_score": score,
        "stability_level": level,
        "log_days": log_days,
        "medication_taken_rate": medication_rate,
        "pain_days": pain_days,
        "attack_count": attack_count,
        "summary": "当前管理稳定度为%s，评分 %s/100。" % (level, score),
    }


def _build_current_shortcomings(logs: pd.DataFrame, labs: pd.DataFrame, attacks: pd.DataFrame) -> list[str]:
    shortcomings: list[str] = []
    water_mean = _safe_mean(logs, "water_ml")
    if water_mean is None or water_mean < 1500:
        shortcomings.append("饮水记录显示整体补水不足")

    medication_rate = _binary_rate(logs, "medication_taken_flag")
    if medication_rate is None or medication_rate < 70:
        shortcomings.append("服药依从性仍需加强")

    latest_uric = _latest_value(labs, "test_date", "uric_acid")
    if latest_uric is not None and float(latest_uric) >= 480:
        shortcomings.append("近期尿酸控制仍不稳定")

    if len(attacks) >= 1:
        shortcomings.append("最近仍有发作记录，需要加强诱因回顾")

    if not shortcomings:
        shortcomings.append("当前整体管理相对稳定，建议继续保持连续记录")
    return shortcomings


def _extract_preference_tokens(text: str) -> list[str]:
    clean_text = str(text or "").replace("，", " ").replace("。", " ").replace("、", " ")
    tokens = [token.strip() for token in clean_text.split() if len(token.strip()) >= 2]
    return tokens[:20]


def _safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _binary_rate(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.clip(0, 1).mean() * 100)


def _most_common_value(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return None
    return str(series.mode().iloc[0])


def _latest_value(frame: pd.DataFrame, sort_column: str, value_column: str) -> Any:
    if frame.empty or sort_column not in frame.columns or value_column not in frame.columns:
        return None
    ordered = frame.copy()
    ordered[sort_column] = pd.to_datetime(ordered[sort_column], errors="coerce")
    ordered = ordered.sort_values(sort_column)
    value = ordered.iloc[-1][value_column]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
