from __future__ import annotations

from typing import Any, Callable

from gout_agent import data


def _build_progress(total_steps: int, completed_steps: int) -> dict[str, Any]:
    total = max(int(total_steps), 0)
    completed = max(min(int(completed_steps), total), 0)
    return {
        "total_steps": total,
        "completed_steps": completed,
        "completion_rate": int(round((completed / total) * 100)) if total else 0,
    }


def _build_generic_task_flow(
    title: str,
    steps: list[dict[str, Any]],
    *,
    phases: list[dict[str, Any]] | None = None,
    status: str = "completed",
    next_action: str = "",
) -> dict[str, Any]:
    completed = sum(1 for step in steps if str(step.get("status") or "") == "done")
    return {
        "title": title,
        "status": status,
        "phases": phases or [],
        "steps": steps,
        "progress": _build_progress(len(steps), completed),
        "next_action": next_action,
    }


def _build_report_generation_flow(report_type: str) -> dict[str, Any]:
    label = "周报" if report_type == "weekly" else "月报"
    steps = [
        {"id": "load_state", "title": "读取当前健康分身与近期记录", "status": "done"},
        {"id": "generate_report", "title": f"生成{label}结构化摘要", "status": "done"},
        {"id": "interpret_report", "title": "结合全局状态完成 AI 解读", "status": "done"},
        {"id": "persist_report", "title": "保存报告摘要供后续复用", "status": "done"},
    ]
    return {
        "title": f"{label}生成任务",
        "status": "completed",
        "phases": [
            {"id": "prepare", "title": "读取状态", "window": "阶段 1"},
            {"id": "generate", "title": "生成并解读报告", "window": "阶段 2"},
            {"id": "persist", "title": "持久化结果", "window": "阶段 3"},
        ],
        "steps": steps,
        "progress": _build_progress(len(steps), len(steps)),
        "next_action": f"{label}已生成完成，可以直接查看摘要与 AI 解读。",
    }


def _build_lab_parse_flow(metric_keys: list[str], parse_status: str) -> dict[str, Any]:
    steps = [
        {"id": "load_file", "title": "读取上传文件", "status": "done"},
        {"id": "extract_text", "title": "提取文本与关键片段", "status": "done"},
        {"id": "parse_metrics", "title": "识别化验指标", "status": "done" if metric_keys else "failed"},
        {"id": "persist_result", "title": "保存识别结果供后续解读", "status": "done"},
    ]
    next_action = "识别结果已写入系统，接下来会结合健康分身和近期行为做解读。"
    if not metric_keys:
        next_action = "这次没有稳定识别出指标值，系统会按文件信息和已有记录做保守解读。"
    return {
        "title": "化验报告识别任务",
        "status": "completed" if parse_status == "parsed" else "needs_adjustment",
        "phases": [
            {"id": "prepare", "title": "读取材料", "window": "阶段 1"},
            {"id": "parse", "title": "识别指标", "window": "阶段 2"},
            {"id": "persist", "title": "保存结果", "window": "阶段 3"},
        ],
        "steps": steps,
        "progress": _build_progress(len(steps), sum(1 for step in steps if step["status"] == "done")),
        "next_action": next_action,
    }


def _build_care_plan_task_flow(plan_payload: dict[str, Any], *, replanned: bool = False) -> dict[str, Any]:
    steps = [
        {
            "id": step.get("id"),
            "title": step.get("title") or "未命名步骤",
            "status": step.get("status") or "pending",
            "phase_id": step.get("phase_id"),
        }
        for step in list(plan_payload.get("steps") or [])
    ]
    return _build_generic_task_flow(
        "管理计划重规划执行流" if replanned else "管理计划执行流",
        steps,
        phases=list(plan_payload.get("phases") or []),
        status=str(plan_payload.get("status") or "active"),
        next_action=str(plan_payload.get("update_plan") or "继续按当前计划执行，并根据新记录更新下一轮计划。"),
    )


def _restore_uploaded_file_payloads(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload or {})
    files: list[dict[str, Any]] = []
    for item in list(restored.get("uploaded_files") or []):
        current = dict(item)
        raw_value = current.get("bytes")
        if isinstance(raw_value, str):
            try:
                current["bytes"] = bytes.fromhex(raw_value)
            except ValueError:
                current["bytes"] = raw_value.encode("utf-8", errors="ignore")
        files.append(current)
    restored["uploaded_files"] = files
    return restored


def _inherit_completed_steps(previous_plan: dict[str, Any], new_plan: dict[str, Any]) -> dict[str, Any]:
    previous_steps = {
        str(step.get("id")): dict(step)
        for step in list((previous_plan or {}).get("steps") or [])
        if str(step.get("id") or "").strip()
    }
    inherited_titles: list[str] = []
    merged_steps: list[dict[str, Any]] = []
    for step in list((new_plan or {}).get("steps") or []):
        current_step = dict(step)
        previous_step = previous_steps.get(str(current_step.get("id") or ""))
        if previous_step and str(previous_step.get("status") or "") == "done":
            current_step["status"] = "done"
            current_step["completion_source"] = previous_step.get("completion_source") or "inherited"
            inherited_titles.append(str(current_step.get("title") or "未命名步骤"))
        merged_steps.append(current_step)
    new_plan["steps"] = merged_steps
    if inherited_titles:
        new_plan["inherited_completed_steps"] = inherited_titles
    return new_plan


def execute_background_job(
    project_root,
    user_id: int,
    job_type: str,
    payload: dict[str, Any],
    *,
    load_context: Callable[[], Any],
    explain_report: Callable[[str, Any], dict[str, Any]],
    generate_care_plan: Callable[[int, Any], dict[str, Any]],
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
        return {
            "report_type": report_type,
            "summary": summary_payload.get("summary"),
            "source": report_result.get("source"),
            "task_flow": _build_report_generation_flow(report_type),
        }
    if job_type == "care_plan_generation":
        horizon_days = int(payload.get("horizon_days") or 7)
        plan_type = "30d" if horizon_days >= 30 else "7d"
        plan_result = generate_care_plan(horizon_days, context)
        data.save_care_plan_summary(project_root, plan_type, horizon_days, plan_result, user_id=user_id)
        run_id = data.create_care_plan_run(project_root, plan_type, horizon_days, plan_result, user_id=user_id)
        return {
            "plan_type": plan_type,
            "summary": plan_result.get("summary"),
            "focus_site": plan_result.get("focus_site"),
            "care_plan_run_id": run_id,
            "task_flow": _build_care_plan_task_flow(plan_result),
        }
    if job_type == "care_plan_replan":
        horizon_days = int(payload.get("horizon_days") or 7)
        plan_type = "30d" if horizon_days >= 30 else "7d"
        plan_result = generate_care_plan(horizon_days, context)
        previous_run_id = payload.get("previous_run_id")
        previous_run = None
        if previous_run_id:
            previous_run = data.get_care_plan_run_by_id(project_root, int(previous_run_id), user_id=user_id)
            data.update_care_plan_run(project_root, int(previous_run_id), status="archived")
        if previous_run:
            plan_result = _inherit_completed_steps(previous_run.get("plan_payload") or {}, plan_result)
            runtime = get_skill_runtime("care_plan")
            if runtime is not None:
                plan_result = runtime.run(
                    "evaluate_care_plan",
                    plan_result,
                    {
                        "twin_state": context.twin_state,
                        "current_risk_overview": context.risk_overview,
                        "site_history": context.site_history.head(20).to_dict(orient="records") if not context.site_history.empty else [],
                    },
                )
        data.save_care_plan_summary(project_root, plan_type, horizon_days, plan_result, user_id=user_id)
        run_id = data.create_care_plan_run(project_root, plan_type, horizon_days, plan_result, user_id=user_id)
        return {
            "plan_type": plan_type,
            "summary": plan_result.get("summary"),
            "focus_site": plan_result.get("focus_site"),
            "care_plan_run_id": run_id,
            "replanned": True,
            "reason": payload.get("reason"),
            "task_flow": _build_care_plan_task_flow(plan_result, replanned=True),
        }
    if job_type == "lab_report_parse":
        payload = _restore_uploaded_file_payloads(payload)
        uploaded_files = list(payload.get("uploaded_files") or [])
        runtime = get_skill_runtime("lab_report")
        parsed = runtime.run("parse_uploaded_lab_files", uploaded_files, extract_lab_metrics_with_local_model)
        source_name = uploaded_files[0].get("name") if uploaded_files else None
        parse_status = "parsed" if parsed.get("metrics") else "fallback"
        data.save_lab_report_parse_result(project_root, source_name, parse_status, parsed, parsed.get("raw_text"), user_id=user_id)
        metric_keys = list((parsed.get("metrics") or {}).keys())
        return {
            "parse_status": parse_status,
            "metric_keys": metric_keys,
            "task_flow": _build_lab_parse_flow(metric_keys, parse_status),
        }
    if job_type == "twin_refresh":
        refreshed_context = refresh_context_state()
        twin_profile = refreshed_context.long_term_memory.get("gout_management_twin_profile") or {}
        return {
            "summary": twin_profile.get("summary"),
            "updated_at": twin_profile.get("updated_at"),
            "task_flow": _build_generic_task_flow(
                "健康分身刷新任务",
                [
                    {"id": "load_records", "title": "加载最新记录", "status": "done"},
                    {"id": "refresh_twin", "title": "刷新健康分身状态", "status": "done"},
                    {"id": "persist_state", "title": "持久化更新结果", "status": "done"},
                ],
                phases=[
                    {"id": "load", "title": "读取当前状态", "window": "阶段 1"},
                    {"id": "refresh", "title": "刷新分身", "window": "阶段 2"},
                    {"id": "persist", "title": "保存结果", "window": "阶段 3"},
                ],
                status="completed",
                next_action="刷新完成后，可继续查看健康分身、风险概览或生成新的管理计划。",
            ),
        }
    raise ValueError(f"不支持的后台任务类型：{job_type}")
