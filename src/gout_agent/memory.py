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

DEFAULT_SITE_LABEL = "未标注部位"

SITE_LABELS = {
    "left_big_toe": "左脚大脚趾",
    "right_big_toe": "右脚大脚趾",
    "left_ankle": "左脚踝",
    "right_ankle": "右脚踝",
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_foot": "左足背",
    "right_foot": "右足背",
    "other": "其他",
}


def build_long_term_memory(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    symptom_logs = symptom_logs if symptom_logs is not None else pd.DataFrame()
    return {
        "behavior_portraits": {
            "7d": build_behavior_portrait(logs, labs, attacks, 7, symptom_logs=symptom_logs),
            "30d": build_behavior_portrait(logs, labs, attacks, 30, symptom_logs=symptom_logs),
            "90d": build_behavior_portrait(logs, labs, attacks, 90, symptom_logs=symptom_logs),
        },
        "gout_management_twin_profile": build_gout_management_twin_profile(
            profile,
            logs,
            labs,
            attacks,
            symptom_logs=symptom_logs,
        ),
        "updated_at": pd.Timestamp.now().isoformat(),
    }


def build_gout_management_twin_profile(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    symptom_logs = symptom_logs if symptom_logs is not None else pd.DataFrame()
    top_triggers = _build_top_triggers(logs, attacks)
    trigger_patterns = _build_trigger_patterns(logs, attacks)
    risk_windows = _build_risk_windows(attacks)
    behavior_patterns = _build_behavior_patterns(logs, labs)
    management_stability = _build_management_stability(logs, attacks)
    current_shortcomings = _build_current_shortcomings(logs, labs, attacks)
    site_trigger_map = _build_site_trigger_map(logs, attacks, symptom_logs)
    site_pain_patterns = _build_site_pain_patterns(attacks, symptom_logs)

    summary_parts: list[str] = []
    if site_trigger_map:
        top_site, triggers = next(iter(site_trigger_map.items()))
        if triggers:
            summary_parts.append(f"{top_site}近期更容易受{'、'.join(triggers[:2])}影响。")
    if site_pain_patterns:
        dominant_site, dominant_payload = max(
            site_pain_patterns.items(),
            key=lambda item: (
                int(item[1].get("attack_count") or 0),
                float(item[1].get("average_pain_score") or 0),
            ),
        )
        average_pain = float(dominant_payload.get("average_pain_score") or 0)
        summary_parts.append(f"{dominant_site}是近期最需要关注的部位，平均疼痛评分约 {average_pain:.1f}。")
    if top_triggers:
        summary_parts.append(f"整体最常见的诱因包括：{'、'.join(item['label'] for item in top_triggers[:3])}。")
    if management_stability.get("stability_score") is not None:
        summary_parts.append(f"当前管理稳定度评分为 {management_stability['stability_score']}/100。")
    if current_shortcomings:
        summary_parts.append(f"近期最需要补强的是：{'、'.join(current_shortcomings[:3])}。")
    if not summary_parts:
        summary_parts.append("当前记录仍然较少，继续记录日常行为、疼痛变化和发作情况后，才能形成更稳定的个人痛风模式。")

    return {
        "summary": " ".join(summary_parts),
        "top_triggers": top_triggers,
        "trigger_patterns": trigger_patterns,
        "risk_windows": risk_windows,
        "behavior_patterns": behavior_patterns,
        "management_stability": management_stability,
        "current_shortcomings": current_shortcomings,
        "site_trigger_map": site_trigger_map,
        "site_pain_patterns": site_pain_patterns,
        "updated_at": pd.Timestamp.now().isoformat(),
    }


def build_behavior_portrait(
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    window_days: int,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    symptom_logs = symptom_logs if symptom_logs is not None else pd.DataFrame()
    reference_date = _resolve_reference_date(logs, labs, attacks, symptom_logs)
    cutoff = pd.Timestamp(reference_date - timedelta(days=max(window_days - 1, 0)))
    log_window = _filter_by_date(logs, "log_date", cutoff)
    lab_window = _filter_by_date(labs, "test_date", cutoff)
    attack_window = _filter_by_date(attacks, "attack_date", cutoff)
    symptom_window = _filter_by_date(symptom_logs, "log_date", cutoff)

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
        "pain_days": int(
            (pd.to_numeric(log_window.get("pain_score"), errors="coerce").fillna(0) > 0).sum()
        )
        if not log_window.empty and "pain_score" in log_window.columns
        else 0,
        "symptom_log_count": int(len(symptom_window)),
        "medication_taken_rate": _binary_rate(log_window, "medication_taken_flag"),
        "latest_uric_acid": _latest_value(lab_window, "test_date", "uric_acid"),
        "attack_count": int(len(attack_window)),
    }

    summary_parts = [f"近 {window_days} 天有 {portrait['days_with_logs']} 天日常记录"]
    if portrait["average_water_ml"] is not None:
        summary_parts.append(f"平均饮水 {portrait['average_water_ml']:.0f} mL/天")
    if portrait["average_steps"] is not None:
        summary_parts.append(f"平均步数 {portrait['average_steps']:.0f} 步/天")
    if portrait["alcohol_days"]:
        summary_parts.append(f"饮酒 {portrait['alcohol_days']} 天")
    if portrait["symptom_log_count"]:
        summary_parts.append(f"记录到 {portrait['symptom_log_count']} 条部位症状")
    if portrait["attack_count"]:
        summary_parts.append(f"记录到 {portrait['attack_count']} 次发作")
    portrait["summary"] = "；".join(summary_parts)
    return portrait


def _resolve_reference_date(
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    symptom_logs: pd.DataFrame,
) -> date:
    candidates: list[pd.Timestamp] = []
    for frame, column in (
        (logs, "log_date"),
        (labs, "test_date"),
        (attacks, "attack_date"),
        (symptom_logs, "log_date"),
    ):
        if frame is None or frame.empty or column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            candidates.append(values.max())
    return max(candidates).date() if candidates else date.today()


def _build_top_triggers(logs: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()

    if not logs.empty:
        for _, row in logs.iterrows():
            text = " ".join(str(row.get(field) or "").lower() for field in ("diet_notes", "free_text", "symptom_notes"))
            for token, label in TRIGGER_KEYWORDS.items():
                if token in text or label in text:
                    counter[label] += 1

            alcohol = str(row.get("alcohol_intake") or "").strip().lower()
            if alcohol and alcohol != "none":
                counter[TRIGGER_KEYWORDS.get(alcohol, "饮酒")] += 1

            water_ml = _coerce_float(row.get("water_ml"))
            if water_ml is not None and water_ml < 1500:
                counter["饮水不足"] += 1

            if not bool(row.get("medication_taken_flag")):
                counter["未按时服药"] += 1

    if not attacks.empty and "suspected_trigger" in attacks.columns:
        for trigger in attacks["suspected_trigger"].dropna().astype(str):
            cleaned = trigger.strip()
            if cleaned:
                counter[_normalize_trigger_label(cleaned)] += 2

    return [{"label": label, "count": count} for label, count in counter.most_common(6)]


def _build_trigger_patterns(logs: pd.DataFrame, attacks: pd.DataFrame) -> list[dict[str, Any]]:
    if logs.empty or attacks.empty:
        return []

    log_frame = logs.copy()
    attack_frame = attacks.copy()
    log_frame["log_date"] = pd.to_datetime(log_frame["log_date"], errors="coerce")
    attack_frame["attack_date"] = pd.to_datetime(attack_frame["attack_date"], errors="coerce")
    log_frame = log_frame.dropna(subset=["log_date"])
    attack_frame = attack_frame.dropna(subset=["attack_date"])
    if log_frame.empty or attack_frame.empty:
        return []

    pattern_counter: Counter[str] = Counter()
    for _, attack in attack_frame.iterrows():
        start = attack["attack_date"] - pd.Timedelta(days=2)
        window = log_frame.loc[(log_frame["log_date"] >= start) & (log_frame["log_date"] <= attack["attack_date"])]
        labels = _extract_trigger_labels(window)
        if labels:
            pattern_counter[" + ".join(sorted(labels)[:3])] += 1

    return [{"pattern": pattern, "count": count} for pattern, count in pattern_counter.most_common(5)]


def _build_risk_windows(attacks: pd.DataFrame) -> list[dict[str, Any]]:
    if attacks.empty:
        return []

    frame = attacks.copy()
    frame["attack_date"] = pd.to_datetime(frame["attack_date"], errors="coerce")
    frame = frame.dropna(subset=["attack_date"])
    if frame.empty:
        return []

    weekday_buckets: Counter[str] = Counter()
    hour_buckets: Counter[str] = Counter()
    for attack_date in frame["attack_date"]:
        weekday_buckets[attack_date.day_name()] += 1
        if attack_date.hour < 12:
            hour_buckets["上午"] += 1
        elif attack_date.hour < 18:
            hour_buckets["下午"] += 1
        else:
            hour_buckets["夜间"] += 1

    windows: list[dict[str, Any]] = []
    windows.extend({"label": label, "count": count} for label, count in weekday_buckets.most_common(2))
    windows.extend({"label": label, "count": count} for label, count in hour_buckets.most_common(1))
    return windows


def _build_behavior_patterns(logs: pd.DataFrame, labs: pd.DataFrame) -> dict[str, Any]:
    if logs.empty:
        return {}

    payload = {
        "average_water_ml": _safe_mean(logs, "water_ml"),
        "average_sleep_hours": _safe_mean(logs, "sleep_hours"),
        "average_steps": _safe_mean(logs, "steps"),
        "average_exercise_minutes": _safe_mean(logs, "exercise_minutes"),
        "drinking_days": int((logs.get("alcohol_intake", pd.Series(dtype=str)).fillna("").astype(str).str.lower() != "none").sum())
        if "alcohol_intake" in logs.columns
        else 0,
        "latest_uric_acid": _latest_value(labs, "test_date", "uric_acid"),
    }

    summary_parts: list[str] = []
    if payload["average_water_ml"] is not None:
        summary_parts.append(f"平均饮水约 {float(payload['average_water_ml']):.0f} mL/天")
    if payload["drinking_days"]:
        summary_parts.append(f"最近记录到 {payload['drinking_days']} 天饮酒")
    if payload["average_sleep_hours"] is not None:
        summary_parts.append(f"平均睡眠约 {float(payload['average_sleep_hours']):.1f} 小时")
    if payload["average_exercise_minutes"] is not None and float(payload["average_exercise_minutes"]) > 0:
        summary_parts.append(f"平均运动约 {float(payload['average_exercise_minutes']):.0f} 分钟/天")
    if payload["average_steps"] is not None:
        summary_parts.append(f"平均步数约 {float(payload['average_steps']):.0f} 步/天")
    if payload["latest_uric_acid"] is not None:
        summary_parts.append(f"最近一次尿酸约 {float(payload['latest_uric_acid']):.0f} umol/L")

    payload["summary"] = "；".join(summary_parts) if summary_parts else "当前记录主要集中在饮水、饮酒、症状和服药，其它生活行为数据仍然较少。"
    return payload


def _build_management_stability(logs: pd.DataFrame, attacks: pd.DataFrame) -> dict[str, Any]:
    scores: list[float] = []

    water_avg = _safe_mean(logs, "water_ml")
    if water_avg is not None:
        scores.append(100.0 if water_avg >= 2000 else max(0.0, water_avg / 20))

    medication_rate = _binary_rate(logs, "medication_taken_flag")
    if medication_rate is not None:
        scores.append(medication_rate)

    sleep_avg = _safe_mean(logs, "sleep_hours")
    if sleep_avg is not None:
        scores.append(100.0 if sleep_avg >= 7 else max(0.0, sleep_avg / 7 * 100))

    if not attacks.empty:
        recent_attacks = len(_filter_by_date(attacks, "attack_date", pd.Timestamp(date.today() - timedelta(days=30))))
        scores.append(max(0.0, 100 - recent_attacks * 20))

    if not scores:
        return {}

    stability_score = round(sum(scores) / len(scores), 1)
    if stability_score >= 80:
        level = "稳定"
    elif stability_score >= 60:
        level = "一般"
    else:
        level = "波动较大"

    return {
        "stability_score": stability_score,
        "stability_level": level,
        "summary": f"当前管理稳定度为{level}，综合评分 {stability_score}/100。",
    }


def _build_current_shortcomings(logs: pd.DataFrame, labs: pd.DataFrame, attacks: pd.DataFrame) -> list[str]:
    items: list[str] = []

    water_avg = _safe_mean(logs, "water_ml")
    if water_avg is not None and water_avg < 1800:
        items.append("饮水量仍然偏低")

    medication_rate = _binary_rate(logs, "medication_taken_flag")
    if medication_rate is not None and medication_rate < 80:
        items.append("服药稳定性不足")

    sleep_avg = _safe_mean(logs, "sleep_hours")
    if sleep_avg is not None and sleep_avg < 7:
        items.append("睡眠不足")

    latest_uric_acid = _latest_value(labs, "test_date", "uric_acid")
    if latest_uric_acid is not None and float(latest_uric_acid) >= 420:
        items.append("尿酸控制仍需加强")

    recent_attacks = len(_filter_by_date(attacks, "attack_date", pd.Timestamp(date.today() - timedelta(days=30))))
    if recent_attacks >= 1:
        items.append("近 30 天仍有发作记录")

    return items


def _build_site_trigger_map(logs: pd.DataFrame, attacks: pd.DataFrame, symptom_logs: pd.DataFrame) -> dict[str, list[str]]:
    site_counter: dict[str, Counter[str]] = {}

    attack_frame = attacks.copy()
    if not attack_frame.empty:
        attack_frame["attack_date"] = pd.to_datetime(attack_frame["attack_date"], errors="coerce")
        attack_frame = attack_frame.dropna(subset=["attack_date"])

    symptom_frame = symptom_logs.copy()
    if not symptom_frame.empty:
        symptom_frame["log_date"] = pd.to_datetime(symptom_frame["log_date"], errors="coerce")
        symptom_frame = symptom_frame.dropna(subset=["log_date"])

    log_frame = logs.copy()
    if not log_frame.empty:
        log_frame["log_date"] = pd.to_datetime(log_frame["log_date"], errors="coerce")
        log_frame = log_frame.dropna(subset=["log_date"])

    def assign(site: str, event_time: pd.Timestamp) -> None:
        if log_frame.empty:
            return
        start = event_time - pd.Timedelta(days=2)
        window = log_frame.loc[(log_frame["log_date"] >= start) & (log_frame["log_date"] <= event_time)]
        labels = _extract_trigger_labels(window)
        if not labels:
            return
        counter = site_counter.setdefault(site or DEFAULT_SITE_LABEL, Counter())
        for label in labels:
            counter[label] += 1

    for _, row in attack_frame.iterrows():
        assign(_normalize_site_label(row.get("joint_site")), row["attack_date"])

    for _, row in symptom_frame.iterrows():
        pain_score = _coerce_float(row.get("pain_score")) or 0
        if pain_score >= 3:
            assign(_normalize_site_label(row.get("body_site")), row["log_date"])

    result: dict[str, list[str]] = {}
    for site, counter in sorted(site_counter.items()):
        result[site] = [label for label, _ in counter.most_common(3)]
    return result


def _build_site_pain_patterns(attacks: pd.DataFrame, symptom_logs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    site_values: dict[str, dict[str, list[float] | int]] = {}

    def record(site_value: Any, pain_value: Any, is_attack: bool) -> None:
        site = _normalize_site_label(site_value)
        bucket = site_values.setdefault(site, {"pain_scores": [], "attack_count": 0, "symptom_count": 0})
        pain = _coerce_float(pain_value)
        if pain is not None:
            bucket["pain_scores"].append(pain)
        if is_attack:
            bucket["attack_count"] += 1
        else:
            bucket["symptom_count"] += 1

    if not attacks.empty:
        for _, row in attacks.iterrows():
            record(row.get("joint_site"), row.get("pain_score"), True)

    if not symptom_logs.empty:
        for _, row in symptom_logs.iterrows():
            record(row.get("body_site"), row.get("pain_score"), False)

    results: dict[str, dict[str, Any]] = {}
    for site, payload in site_values.items():
        pain_scores = payload["pain_scores"]
        results[site] = {
            "average_pain_score": round(sum(pain_scores) / len(pain_scores), 1) if pain_scores else None,
            "max_pain_score": max(pain_scores) if pain_scores else None,
            "attack_count": payload["attack_count"],
            "symptom_count": payload["symptom_count"],
        }
    return results


def _extract_trigger_labels(logs: pd.DataFrame) -> list[str]:
    labels: set[str] = set()
    if logs.empty:
        return []

    for _, row in logs.iterrows():
        text = " ".join(str(row.get(field) or "").lower() for field in ("diet_notes", "free_text", "symptom_notes"))
        for token, label in TRIGGER_KEYWORDS.items():
            if token in text or label in text:
                labels.add(label)

        alcohol = str(row.get("alcohol_intake") or "").strip().lower()
        if alcohol and alcohol != "none":
            labels.add(TRIGGER_KEYWORDS.get(alcohol, "饮酒"))

        water_ml = _coerce_float(row.get("water_ml"))
        if water_ml is not None and water_ml < 1500:
            labels.add("饮水不足")

        if not bool(row.get("medication_taken_flag")):
            labels.add("未按时服药")

    return sorted(labels)


def _filter_by_date(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=frame.columns if not frame.empty else None)

    copied = frame.copy()
    copied[column] = pd.to_datetime(copied[column], errors="coerce")
    copied = copied.dropna(subset=[column])
    return copied.loc[copied[column] >= cutoff]


def _safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.mean()), 1)


def _binary_rate(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.clip(0, 1).mean() * 100), 1)


def _latest_value(frame: pd.DataFrame, sort_column: str, value_column: str) -> Any:
    if frame.empty or sort_column not in frame.columns or value_column not in frame.columns:
        return None

    copied = frame.copy()
    copied[sort_column] = pd.to_datetime(copied[sort_column], errors="coerce")
    copied = copied.dropna(subset=[sort_column]).sort_values(sort_column)
    if copied.empty:
        return None

    value = copied.iloc[-1][value_column]
    return None if pd.isna(value) else value


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_site_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_SITE_LABEL

    normalized = text.lower().replace("-", "_").replace(" ", "_")
    return SITE_LABELS.get(normalized, text)


def _normalize_trigger_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    for token, label in TRIGGER_KEYWORDS.items():
        if normalized == token or normalized == label.lower():
            return label
    return value.strip()


def build_llm_memory_summary(long_term_memory: dict[str, Any] | None) -> dict[str, Any]:
    memory_payload = long_term_memory or {}
    portraits = memory_payload.get("behavior_portraits") or {}
    twin_profile = memory_payload.get("gout_management_twin_profile") or {}
    portrait_summary = {
        key: (value or {}).get("summary", "")
        for key, value in portraits.items()
        if isinstance(value, dict)
    }
    twin_summary = {
        "summary": twin_profile.get("summary", ""),
        "top_triggers": [item.get("label") for item in (twin_profile.get("top_triggers") or [])[:3] if item.get("label")],
        "current_shortcomings": list((twin_profile.get("current_shortcomings") or [])[:3]),
        "stability_score": ((twin_profile.get("management_stability") or {}).get("stability_score")),
        "site_trigger_map": {
            site: triggers[:2]
            for site, triggers in list((twin_profile.get("site_trigger_map") or {}).items())[:3]
        },
    }
    return {
        "behavior_portraits": portrait_summary,
        "digital_twin_profile": twin_summary,
        "updated_at": memory_payload.get("updated_at"),
    }


def build_report_memory_summary(long_term_memory: dict[str, Any] | None) -> dict[str, Any]:
    memory_payload = long_term_memory or {}
    portraits = memory_payload.get("behavior_portraits") or {}
    twin_profile = memory_payload.get("gout_management_twin_profile") or {}
    management_stability = twin_profile.get("management_stability") or {}
    return {
        "recent_behavior": {
            "7d": (portraits.get("7d") or {}).get("summary", ""),
            "30d": (portraits.get("30d") or {}).get("summary", ""),
            "90d": (portraits.get("90d") or {}).get("summary", ""),
        },
        "twin_summary": twin_profile.get("summary", ""),
        "top_triggers": [item.get("label") for item in (twin_profile.get("top_triggers") or [])[:5] if item.get("label")],
        "focus_sites": list((twin_profile.get("site_trigger_map") or {}).keys())[:3],
        "management_stability": {
            "score": management_stability.get("stability_score"),
            "level": management_stability.get("stability_level"),
            "summary": management_stability.get("summary"),
        },
    }
