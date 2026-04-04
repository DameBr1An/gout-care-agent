from __future__ import annotations

from typing import Any, Callable

from gout_agent import data


def execute_background_job(
    project_root,
    user_id: int,
    job_type: str,
    payload: dict[str, Any],
    *,
    load_context: Callable[[], Any],
    explain_report: Callable[[str, Any], dict[str, Any]],
    refresh_context_state: Callable[[], Any],
    get_skill_runtime: Callable[[str], Any],
    extract_lab_metrics_with_local_model: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    context = load_context()
    if job_type == "report_generation":
        report_type = str(payload.get("report_type") or "weekly")
        report_result = explain_report(report_type, context)
        report_payload = report_result.get("report") or {}
        period_text = str(report_payload.get("period") or "")
        period_start, period_end = (period_text.split(" 至 ", 1) + [None])[:2] if " 至 " in period_text else (None, None)
        summary_payload = {
            "summary": report_payload.get("executive_summary") or report_payload.get("summary"),
            "action_plan": report_payload.get("action_plan") or report_payload.get("suggestions") or [],
            "source": report_result.get("source"),
            "explanation": report_result.get("explanation"),
            "report": report_payload,
        }
        data.save_report_summary(project_root, report_type, summary_payload, period_start, period_end, user_id=user_id)
        return {"report_type": report_type, "summary": summary_payload.get("summary"), "source": report_result.get("source")}
    if job_type == "lab_report_parse":
        uploaded_files = list(payload.get("uploaded_files") or [])
        runtime = get_skill_runtime("lab_report")
        parsed = runtime.run("parse_uploaded_lab_files", uploaded_files, extract_lab_metrics_with_local_model)
        source_name = uploaded_files[0].get("name") if uploaded_files else None
        parse_status = "parsed" if parsed.get("metrics") else "fallback"
        data.save_lab_report_parse_result(project_root, source_name, parse_status, parsed, parsed.get("raw_text"), user_id=user_id)
        return {"parse_status": parse_status, "metric_keys": list((parsed.get("metrics") or {}).keys())}
    if job_type == "twin_refresh":
        refreshed_context = refresh_context_state()
        twin_profile = refreshed_context.long_term_memory.get("gout_management_twin_profile") or {}
        return {"summary": twin_profile.get("summary"), "updated_at": twin_profile.get("updated_at")}
    raise ValueError(f"不支持的后台任务类型：{job_type}")
