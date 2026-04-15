from __future__ import annotations

import json
from typing import Any

import pandas as pd

from gout_agent.toolkit import serialize_tool_result


def build_twin_state(long_term_memory: dict[str, Any], risk_overview: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior_portraits": long_term_memory.get("behavior_portraits") or {},
        "digital_twin_profile": long_term_memory.get("gout_management_twin_profile") or {},
        "memory_summary": long_term_memory.get("memory_summary") or {},
        "report_memory_summary": long_term_memory.get("report_memory_summary") or {},
        "risk_anchor": {
            "attack_risk_label": risk_overview.get("attack_risk_label"),
            "overall_risk_score": risk_overview.get("overall_risk_score"),
            "trigger_summary": risk_overview.get("trigger_summary") or [],
        },
        "risk_view": {
            "attack_risk_label": risk_overview.get("attack_risk_label"),
            "overall_risk_score": risk_overview.get("overall_risk_score"),
            "explanation": risk_overview.get("explanation"),
            "hydration_advice": risk_overview.get("hydration_advice"),
            "diet_advice": risk_overview.get("diet_advice"),
            "exercise_advice": risk_overview.get("exercise_advice"),
            "behavior_goal": risk_overview.get("behavior_goal"),
            "trigger_summary": risk_overview.get("trigger_summary") or [],
            "abnormal_items": risk_overview.get("abnormal_items") or [],
        },
    }


def build_user_journal(profile: dict[str, Any], logs: pd.DataFrame) -> dict[str, Any]:
    recent_records = logs.tail(20).to_dict(orient="records") if not logs.empty else []
    latest_record = logs.iloc[-1].to_dict() if not logs.empty else {}
    return {
        "profile": serialize_tool_result(profile),
        "recent_health_records": serialize_tool_result(recent_records),
        "latest_record": serialize_tool_result(latest_record),
    }


def build_site_history(symptom_logs: pd.DataFrame, attacks: pd.DataFrame) -> pd.DataFrame:
    symptom_frame = symptom_logs.copy()
    attack_frame = attacks.copy()
    normalized_frames: list[pd.DataFrame] = []
    if not symptom_frame.empty:
        symptom_frame = symptom_frame.assign(
            event_type="symptom",
            event_date=symptom_frame.get("log_date"),
            site=symptom_frame.get("body_site"),
            trigger_notes=None,
            duration_hours=None,
            resolved_flag=None,
        )
        normalized_frames.append(
            symptom_frame[
                [
                    "event_type",
                    "event_date",
                    "site",
                    "pain_score",
                    "swelling_flag",
                    "redness_flag",
                    "stiffness_flag",
                    "symptom_notes",
                    "trigger_notes",
                    "duration_hours",
                    "resolved_flag",
                ]
            ].copy()
        )
    if not attack_frame.empty:
        attack_frame = attack_frame.assign(
            event_type="attack",
            event_date=attack_frame.get("attack_date"),
            site=attack_frame.get("joint_site"),
            stiffness_flag=None,
            symptom_notes=attack_frame.get("notes"),
            trigger_notes=attack_frame.get("suspected_trigger"),
        )
        normalized_frames.append(
            attack_frame[
                [
                    "event_type",
                    "event_date",
                    "site",
                    "pain_score",
                    "swelling_flag",
                    "redness_flag",
                    "stiffness_flag",
                    "symptom_notes",
                    "trigger_notes",
                    "duration_hours",
                    "resolved_flag",
                ]
            ].copy()
        )
    if not normalized_frames:
        return pd.DataFrame(
            columns=[
                "event_type",
                "event_date",
                "site",
                "pain_score",
                "swelling_flag",
                "redness_flag",
                "stiffness_flag",
                "symptom_notes",
                "trigger_notes",
                "duration_hours",
                "resolved_flag",
            ]
        )
    history = pd.concat(normalized_frames, ignore_index=True)
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    history = history.sort_values(["event_date", "event_type"], ascending=[False, True]).reset_index(drop=True)
    return history


def build_risk_overview(risk_result: Any, trigger_summary: list[dict[str, Any]], abnormal_items: list[str], label_risk) -> dict[str, Any]:
    return {
        "uric_acid_risk_label": label_risk(risk_result.uric_acid_risk_level),
        "attack_risk_label": label_risk(risk_result.attack_risk_level),
        "overall_risk_score": risk_result.overall_risk_score,
        "explanation": risk_result.explanation,
        "hydration_advice": risk_result.hydration_advice,
        "diet_advice": risk_result.diet_advice,
        "exercise_advice": risk_result.exercise_advice,
        "behavior_goal": risk_result.behavior_goal,
        "trigger_summary": trigger_summary,
        "abnormal_items": abnormal_items,
    }


def build_harness_state_summary(context: Any) -> dict[str, Any]:
    latest_sites: list[str] = []
    if not context.site_history.empty and "site" in context.site_history.columns:
        latest_sites = list(dict.fromkeys(context.site_history.head(5)["site"].dropna().astype(str).tolist()))
    return {
        "user_journal_summary": {
            "has_profile": bool(context.user_journal.get("profile")),
            "recent_record_count": len(context.user_journal.get("recent_health_records", [])),
            "latest_record": context.user_journal.get("latest_record", {}),
        },
        "site_history_summary": {
            "recent_event_count": int(min(len(context.site_history), 14)),
            "latest_sites": latest_sites,
        },
        "risk_summary": {
            "attack_risk_label": context.risk_overview.get("attack_risk_label"),
            "overall_risk_score": context.risk_overview.get("overall_risk_score"),
            "top_triggers": [item.get("label") for item in context.risk_overview.get("trigger_summary", [])[:3] if item.get("label")],
            "abnormal_items": context.risk_overview.get("abnormal_items", [])[:3],
        },
    }


def serialize_context_payload(context: Any, label_risk) -> dict[str, Any]:
    return {
        "user_journal": context.user_journal,
        "site_history": context.site_history.tail(20).to_dict(orient="records") if not context.site_history.empty else [],
        "risk_overview": context.risk_overview,
        "risk_result": {
            "uric_acid_risk_level": context.risk_result.uric_acid_risk_level,
            "attack_risk_level": context.risk_result.attack_risk_level,
            "uric_acid_risk_level_cn": label_risk(context.risk_result.uric_acid_risk_level),
            "attack_risk_level_cn": label_risk(context.risk_result.attack_risk_level),
            "overall_risk_score": context.risk_result.overall_risk_score,
            "explanation": context.risk_result.explanation,
            "hydration_advice": context.risk_result.hydration_advice,
            "diet_advice": context.risk_result.diet_advice,
            "exercise_advice": context.risk_result.exercise_advice,
            "behavior_goal": context.risk_result.behavior_goal,
        },
        "trigger_summary": context.trigger_summary,
        "abnormal_items": context.abnormal_items,
        "medication_completion_rate": context.medication_completion_rate,
        "active_reminder_count": len(context.reminders),
        "twin_state": context.twin_state,
        "long_term_memory": context.long_term_memory,
        "session_memories": context.session_memories,
        "recent_symptom_logs": context.symptom_logs.tail(20).to_dict(orient="records") if not context.symptom_logs.empty else [],
    }


def build_llm_context_payload(context: Any, label_risk) -> dict[str, Any]:
    payload = serialize_context_payload(context, label_risk)
    payload.update(
        {
            "user_profile": context.user_journal.get("profile", {}),
            "recent_health_records": context.user_journal.get("recent_health_records", []),
            "site_history_preview": context.site_history.head(10).to_dict(orient="records") if not context.site_history.empty else [],
            "latest_daily_log": context.logs.iloc[-1].to_dict() if not context.logs.empty else {},
            "recent_symptom_logs": context.symptom_logs.head(10).to_dict(orient="records") if not context.symptom_logs.empty else [],
            "recent_attack_records": context.attacks.head(5).to_dict(orient="records") if not context.attacks.empty else [],
            "behavior_portraits": context.twin_state.get("behavior_portraits"),
            "digital_twin_profile": context.twin_state.get("digital_twin_profile"),
            "recent_session_memories": context.session_memories[-6:],
            "memory_summary": context.twin_state.get("memory_summary"),
            "state_summary": build_harness_state_summary(context),
            "twin_state": context.twin_state,
        }
    )
    return payload


def build_report_history_summaries(report_history: pd.DataFrame | None) -> list[dict[str, Any]]:
    history_summaries: list[dict[str, Any]] = []
    if report_history is None or report_history.empty:
        return history_summaries
    for _, row in report_history.head(4).iterrows():
        report_payload = {}
        raw = row.get("report_json")
        if isinstance(raw, str) and raw.strip():
            try:
                report_payload = json.loads(raw)
            except json.JSONDecodeError:
                report_payload = {}
        history_summaries.append(
            {
                "report_type": row.get("report_type"),
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "summary": report_payload.get("executive_summary") or report_payload.get("summary"),
                "action_plan": report_payload.get("action_plan") or [],
            }
        )
    return history_summaries


def build_interpretation_context_payload(
    context: Any,
    label_risk,
    report_history: pd.DataFrame | None,
    *,
    selected_report: dict[str, Any] | None = None,
    period_type: str | None = None,
    uploaded_lab_reports: list[dict[str, Any]] | None = None,
    parsed_lab_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_llm_context_payload(context, label_risk)
    payload.update(
        {
            "selected_report": selected_report or {},
            "report_period_type": period_type,
            "report_history_summaries": build_report_history_summaries(report_history),
            "current_risk_overview": context.risk_overview,
            "uploaded_lab_reports": uploaded_lab_reports or [],
            "parsed_lab_reports": parsed_lab_reports or {},
            "report_memory_summary": context.twin_state.get("report_memory_summary"),
        }
    )
    return payload
