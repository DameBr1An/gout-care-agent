import json
from datetime import date, timedelta

import pandas as pd

from gout_agent.risk import calculate_gout_risk, detect_gout_triggers


def _period_frame(frame, date_column, start_date, end_date):
    if frame.empty:
        return frame.copy()
    scoped = frame.copy()
    scoped[date_column] = pd.to_datetime(scoped[date_column], errors="coerce")
    return scoped.loc[scoped[date_column].dt.date.between(start_date, end_date, inclusive="both")].copy()


def _mean_or_none(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return None
    return round(float(numeric.mean()), 1)


def _rate_or_none(values):
    if values.empty:
        return None
    return round(float(values.mean() * 100), 1)


def build_period_report(profile, logs, labs, attacks, period_days):
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days - 1)
    period_logs = _period_frame(logs, "log_date", start_date, end_date)
    period_labs = _period_frame(labs, "test_date", start_date, end_date)
    period_attacks = _period_frame(attacks, "attack_date", start_date, end_date)
    latest_risk = calculate_gout_risk(profile, logs, labs, attacks)
    triggers = detect_gout_triggers(period_logs, window_days=period_days)
    medication_rate = None
    if not period_logs.empty and "medication_taken_flag" in period_logs.columns:
        medication_series = pd.to_numeric(period_logs["medication_taken_flag"], errors="coerce").fillna(0).clip(0, 1)
        medication_rate = _rate_or_none(medication_series)

    return {
        "report_title": "痛风管理周报" if period_days == 7 else "痛风管理月报",
        "report_subtitle": "统一采用结论摘要、关键发现和下一步建议的表达方式",
        "period": "%s 至 %s" % (start_date, end_date),
        "period_days": period_days,
        "entries": int(len(period_logs)),
        "lab_entries": int(len(period_labs)),
        "attack_entries": int(len(period_attacks)),
        "mean_uric_acid": _mean_or_none(period_labs["uric_acid"]) if "uric_acid" in period_labs else None,
        "mean_water_ml": _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None,
        "mean_pain_score": _mean_or_none(period_logs["pain_score"]) if "pain_score" in period_logs else None,
        "medication_adherence_rate": medication_rate,
        "top_triggers": list(triggers.items())[:5],
        "latest_risk": {
            "uric_acid_risk_level": latest_risk.uric_acid_risk_level,
            "uric_acid_risk_level_cn": _risk_label_cn(latest_risk.uric_acid_risk_level),
            "attack_risk_level": latest_risk.attack_risk_level,
            "attack_risk_level_cn": _risk_label_cn(latest_risk.attack_risk_level),
            "overall_risk_score": latest_risk.overall_risk_score,
            "trend_direction": latest_risk.trend_direction,
            "top_risk_factors": latest_risk.top_risk_factors,
        },
        "summary": _build_summary(period_logs, period_labs, period_attacks, latest_risk, period_days),
        "executive_summary": _build_executive_summary(period_logs, period_labs, period_attacks, latest_risk, period_days),
        "key_findings": _build_key_findings(period_logs, period_labs, period_attacks, latest_risk, triggers),
        "action_plan": _build_action_plan(period_logs, latest_risk, triggers),
        "medical_notice": _build_medical_notice(period_labs, period_attacks, latest_risk),
    }


def _build_summary(period_logs, period_labs, period_attacks, latest_risk, period_days):
    if period_logs.empty and period_labs.empty and period_attacks.empty:
        return "最近 %s 天还没有记录到痛风管理数据。" % period_days

    parts = []
    if not period_labs.empty and "uric_acid" in period_labs:
        mean_uric = _mean_or_none(period_labs["uric_acid"])
        if mean_uric is not None:
            parts.append("平均尿酸为 %s umol/L" % mean_uric)
    if not period_logs.empty and "water_ml" in period_logs:
        mean_water = _mean_or_none(period_logs["water_ml"])
        if mean_water is not None:
            parts.append("平均饮水量为 %s mL/天" % mean_water)
    if len(period_attacks):
        parts.append("记录到 %s 次发作" % len(period_attacks))
    parts.append("当前发作风险为 %s" % _risk_label_cn(latest_risk.attack_risk_level))
    return "；".join(parts) + "。"


def _build_executive_summary(period_logs, period_labs, period_attacks, latest_risk, period_days):
    if period_logs.empty and period_labs.empty and period_attacks.empty:
        return "本期暂无足够记录，建议先连续补充饮水、症状、饮食和化验数据，以便形成更稳定的管理判断。"
    return (
        "过去 %s 天内，当前整体发作风险为%s。"
        "建议优先关注补水、诱因控制和规律记录，如症状明显加重请及时线下就医。"
        % (period_days, _risk_label_cn(latest_risk.attack_risk_level))
    )


def _build_key_findings(period_logs, period_labs, period_attacks, latest_risk, triggers):
    findings = []
    if not period_labs.empty and "uric_acid" in period_labs:
        mean_uric = _mean_or_none(period_labs["uric_acid"])
        if mean_uric is not None:
            findings.append("本期平均尿酸为 %s umol/L。" % mean_uric)
    if not period_logs.empty and "water_ml" in period_logs:
        mean_water = _mean_or_none(period_logs["water_ml"])
        if mean_water is not None:
            findings.append("本期平均饮水量为 %s mL/天。" % mean_water)
    if len(period_attacks):
        findings.append("本期共记录到 %s 次痛风发作。" % len(period_attacks))
    trigger_labels = [_trigger_label(name) for name, _count in list(triggers.items())[:3]]
    if trigger_labels:
        findings.append("近期重点诱因包括：%s。" % "、".join(trigger_labels))
    findings.append("当前整体风险评分为 %s。" % latest_risk.overall_risk_score)
    return findings


def _build_action_plan(period_logs, latest_risk, triggers):
    actions = []
    mean_water = _mean_or_none(period_logs["water_ml"]) if "water_ml" in period_logs else None
    if mean_water is None or mean_water < 1800:
        actions.append("把每日饮水量稳定提升到建议范围，并尽量分次完成。")
    else:
        actions.append("继续保持当前补水节奏，避免长时间未饮水。")
    if triggers:
        actions.append("优先减少近期已识别的高风险诱因，例如饮酒、海鲜、火锅或烧烤。")
    else:
        actions.append("继续记录饮食和症状变化，帮助系统更准确识别个人诱因。")
    if latest_risk.attack_risk_level in {"High", "Moderate"}:
        actions.append("近期请规律记录疼痛、关节情况和服药状态，必要时尽快复查。")
    else:
        actions.append("继续维持规律作息、适量运动和按时随访。")
    return actions


def _build_medical_notice(period_labs, period_attacks, latest_risk):
    if len(period_attacks) >= 2 or latest_risk.attack_risk_level == "High":
        return "如果近期疼痛明显加重、关节红肿发热、活动受限或反复发作，请及时线下就医。"
    if not period_labs.empty and "uric_acid" in period_labs:
        latest_uric = pd.to_numeric(period_labs["uric_acid"], errors="coerce").dropna()
        if not latest_uric.empty and float(latest_uric.iloc[-1]) >= 540:
            return "如果尿酸持续明显升高，建议尽快结合医生意见复查并评估降尿酸方案。"
    return "本报告用于健康管理，不替代医生诊断；如症状持续或异常指标加重，请及时就医。"


def _risk_label_cn(value):
    return {"Low": "低", "Moderate": "中", "High": "高", "low": "低", "moderate": "中", "high": "高"}.get(value, str(value))


def _trigger_label(name):
    return {
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
    }.get(name, str(name))


def build_weekly_report(profile, logs, labs, attacks):
    return build_period_report(profile, logs, labs, attacks, period_days=7)


def build_monthly_report(profile, logs, labs, attacks):
    return build_period_report(profile, logs, labs, attacks, period_days=30)


def export_report(root, report, report_type, format_name="json"):
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    suffix = format_name.lower()
    file_path = reports_dir / ("%s_gout_report.%s" % (report_type, suffix))

    if suffix == "json":
        file_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix == "html":
        html = "<html><body><pre>" + json.dumps(report, indent=2, ensure_ascii=False) + "</pre></body></html>"
        file_path.write_text(html, encoding="utf-8")
    else:
        raise ValueError("不支持的导出格式：%s" % format_name)

    return file_path
