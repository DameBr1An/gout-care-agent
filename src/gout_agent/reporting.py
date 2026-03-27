from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gout_agent import memory
from gout_agent.risk import calculate_gout_risk, detect_gout_triggers


def _period_frame(frame: pd.DataFrame, date_column: str, start_date: date, end_date: date) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return frame.copy()
    scoped = frame.copy()
    scoped[date_column] = pd.to_datetime(scoped[date_column], errors="coerce")
    return scoped.loc[scoped[date_column].dt.date.between(start_date, end_date, inclusive="both")].copy()


def _mean_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return None
    return round(float(numeric.mean()), 1)


def _rate_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty or not numeric.notna().any():
        return None
    return round(float(numeric.fillna(0).mean() * 100), 1)


def _risk_label_cn(value: str | None) -> str:
    return {
        "Low": "低",
        "Moderate": "中",
        "High": "高",
        "low": "低",
        "moderate": "中",
        "high": "高",
    }.get(value, str(value or "未知"))


def _trigger_label(name: str) -> str:
    mapping = {
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
    return mapping.get(name, str(name))


def _build_summary(period_logs: pd.DataFrame, period_attacks: pd.DataFrame, latest_risk, period_days: int) -> str:
    if period_logs.empty and period_attacks.empty:
        return f"最近 {period_days} 天还没有形成足够记录，建议继续补充日常行为、部位症状和发作信息。"

    parts: list[str] = []
    mean_water = _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None
    if mean_water is not None:
        parts.append(f"平均饮水约 {mean_water:.0f} mL/天")
    if len(period_attacks):
        parts.append(f"记录到 {len(period_attacks)} 次明确发作")
    parts.append(f"当前发作风险为{_risk_label_cn(latest_risk.attack_risk_level)}")
    return "；".join(parts) + "。"


def _build_executive_summary(period_logs: pd.DataFrame, period_attacks: pd.DataFrame, latest_risk, period_days: int) -> str:
    if period_logs.empty and period_attacks.empty:
        return "本期记录量偏少，建议先连续记录饮水、饮酒、饮食、部位症状和发作情况。"
    return (
        f"过去 {period_days} 天内，系统判断当前发作风险为{_risk_label_cn(latest_risk.attack_risk_level)}，"
        "建议优先关注补水、诱因控制和部位症状变化。"
    )


def _build_key_findings(
    period_logs: pd.DataFrame,
    period_attacks: pd.DataFrame,
    latest_risk,
    triggers: dict[str, int],
    twin_profile: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    mean_water = _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None
    if mean_water is not None:
        findings.append(f"本期平均饮水量约为 {mean_water:.0f} mL/天。")
    if len(period_attacks):
        findings.append(f"本期共记录到 {len(period_attacks)} 次明确发作。")
    trigger_labels = [_trigger_label(name) for name, _count in list(triggers.items())[:3]]
    if trigger_labels:
        findings.append(f"近期重点诱因包括：{'、'.join(trigger_labels)}。")

    site_pain_patterns = twin_profile.get("site_pain_patterns") or {}
    if site_pain_patterns:
        dominant_site, dominant_payload = max(
            site_pain_patterns.items(),
            key=lambda item: (item[1].get("max_pain_score") or 0, item[1].get("average_pain_score") or 0),
        )
        findings.append(
            f"{dominant_site}是近期最需要关注的部位，平均疼痛约 {float(dominant_payload.get('average_pain_score') or 0):.1f}/10。"
        )

    findings.append(f"当前整体风险评分为 {latest_risk.overall_risk_score}。")
    return findings


def _build_action_plan(period_logs: pd.DataFrame, latest_risk, triggers: dict[str, int], twin_profile: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    mean_water = _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None
    if mean_water is None or mean_water < 1800:
        actions.append("优先把每日饮水量稳定提升到建议范围，并尽量分次完成。")
    else:
        actions.append("继续保持当前补水节奏，避免长时间未饮水。")

    if triggers:
        actions.append("优先减少近期已识别的高风险诱因，如饮酒、海鲜、火锅或烧烤。")

    twin_summary = twin_profile.get("summary")
    if twin_summary:
        actions.append(f"结合个人模式，近期重点关注：{twin_summary}")

    if latest_risk.attack_risk_level in {"High", "Moderate"}:
        actions.append("近期请持续记录部位症状变化，若疼痛加重或范围扩大，尽快线下就医。")
    else:
        actions.append("继续维持规律记录，帮助系统更稳定地识别你的个人模式。")
    return actions


def _build_medical_notice(period_attacks: pd.DataFrame, latest_risk) -> str:
    if len(period_attacks) >= 2 or latest_risk.attack_risk_level == "High":
        return "如果近期疼痛明显加重、关节红肿发热、活动受限或反复发作，请及时线下就医。"
    return "本报告用于健康管理，不替代医生诊断；如症状持续或加重，请及时就医。"


def build_period_report(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    period_days: int,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days - 1)
    period_logs = _period_frame(logs, "log_date", start_date, end_date)
    period_labs = _period_frame(labs, "test_date", start_date, end_date)
    period_attacks = _period_frame(attacks, "attack_date", start_date, end_date)
    period_symptom_logs = _period_frame(symptom_logs if symptom_logs is not None else pd.DataFrame(), "log_date", start_date, end_date)

    latest_risk = calculate_gout_risk(profile, logs, labs, attacks)
    triggers = detect_gout_triggers(period_logs, window_days=period_days)
    medication_rate = None
    if not period_logs.empty and "medication_taken_flag" in period_logs.columns:
        medication_rate = _rate_or_none(period_logs["medication_taken_flag"])

    twin_profile = memory.build_gout_management_twin_profile(
        profile,
        period_logs,
        period_labs,
        period_attacks,
        symptom_logs=period_symptom_logs,
    )

    return {
        "report_title": "痛风管理周报" if period_days == 7 else "痛风管理月报",
        "report_subtitle": "采用结论摘要、关键发现和下一步建议的统一表达方式。",
        "period": f"{start_date} 至 {end_date}",
        "period_days": period_days,
        "entries": int(len(period_logs)),
        "attack_entries": int(len(period_attacks)),
        "symptom_entries": int(len(period_symptom_logs)),
        "mean_water_ml": _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None,
        "mean_pain_score": _mean_or_none(period_logs["pain_score"]) if "pain_score" in period_logs else None,
        "medication_adherence_rate": medication_rate,
        "top_triggers": [(_trigger_label(name), count) for name, count in list(triggers.items())[:5]],
        "latest_risk": {
            "uric_acid_risk_level": latest_risk.uric_acid_risk_level,
            "uric_acid_risk_level_cn": _risk_label_cn(latest_risk.uric_acid_risk_level),
            "attack_risk_level": latest_risk.attack_risk_level,
            "attack_risk_level_cn": _risk_label_cn(latest_risk.attack_risk_level),
            "overall_risk_score": latest_risk.overall_risk_score,
            "trend_direction": latest_risk.trend_direction,
            "top_risk_factors": latest_risk.top_risk_factors,
        },
        "summary": _build_summary(period_logs, period_attacks, latest_risk, period_days),
        "executive_summary": _build_executive_summary(period_logs, period_attacks, latest_risk, period_days),
        "key_findings": _build_key_findings(period_logs, period_attacks, latest_risk, triggers, twin_profile),
        "action_plan": _build_action_plan(period_logs, latest_risk, triggers, twin_profile),
        "medical_notice": _build_medical_notice(period_attacks, latest_risk),
        "personal_pattern_summary": twin_profile.get("summary"),
        "digital_twin_profile": twin_profile,
    }


def build_weekly_report(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return build_period_report(profile, logs, labs, attacks, period_days=7, symptom_logs=symptom_logs)


def build_monthly_report(
    profile: dict[str, Any],
    logs: pd.DataFrame,
    labs: pd.DataFrame,
    attacks: pd.DataFrame,
    symptom_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return build_period_report(profile, logs, labs, attacks, period_days=30, symptom_logs=symptom_logs)


def export_report(root: Path, report: dict[str, Any], report_type: str, format_name: str = "json") -> Path:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    suffix = format_name.lower()
    file_path = reports_dir / f"{report_type}_gout_report.{suffix}"

    if suffix == "json":
        file_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix == "html":
        html = "<html><body><pre>" + json.dumps(report, indent=2, ensure_ascii=False) + "</pre></body></html>"
        file_path.write_text(html, encoding="utf-8")
    else:
        raise ValueError(f"不支持的导出格式：{format_name}")

    return file_path
