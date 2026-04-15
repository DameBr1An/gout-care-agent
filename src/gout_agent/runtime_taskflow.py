from __future__ import annotations

from typing import Any


def build_progress(total_steps: int, completed_steps: int) -> dict[str, Any]:
    total = max(int(total_steps), 0)
    completed = max(min(int(completed_steps), total), 0)
    return {
        "total_steps": total,
        "completed_steps": completed,
        "completion_rate": int(round((completed / total) * 100)) if total else 0,
    }


def build_generic_task_flow(
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
        "progress": build_progress(len(steps), completed),
        "next_action": next_action,
    }


_WRITE_TASK_TEMPLATES: dict[str, dict[str, str]] = {
    "记录日常健康": {
        "title": "日常行为记录任务",
        "prepare": "整理今天的饮水、饮酒、饮食与服药信息",
        "write": "写入日常行为记录",
        "refresh": "刷新当前风险与健康分身状态",
    },
    "记录部位症状": {
        "title": "疼痛记录任务",
        "prepare": "整理部位、疼痛程度和局部症状",
        "write": "写入部位症状记录",
        "refresh": "刷新当前风险与健康分身状态",
    },
    "记录痛风发作": {
        "title": "发作记录任务",
        "prepare": "整理发作部位、持续时间和诱因线索",
        "write": "写入发作记录",
        "refresh": "刷新当前风险与健康分身状态",
    },
    "记录服药情况": {
        "title": "服药记录任务",
        "prepare": "整理本次服药状态与时间",
        "write": "写入服药状态",
        "refresh": "刷新当前风险与健康分身状态",
    },
    "添加药物方案": {
        "title": "药物方案维护任务",
        "prepare": "整理药物名称、剂量和频率",
        "write": "写入药物方案",
        "refresh": "刷新当前风险与健康分身状态",
    },
    "更新用户档案": {
        "title": "基础资料更新任务",
        "prepare": "整理基础资料与健康背景",
        "write": "写入用户档案",
        "refresh": "刷新当前风险与健康分身状态",
    },
}


def build_context_next_action(context: Any | None) -> str:
    if context is None:
        return ""
    risk_overview = getattr(context, "risk_overview", {}) or {}
    advice_parts = [
        risk_overview.get("hydration_advice"),
        risk_overview.get("diet_advice"),
        risk_overview.get("behavior_goal"),
    ]
    cleaned = [str(item).strip() for item in advice_parts if str(item or "").strip()]
    if cleaned:
        return "；".join(cleaned[:3])
    twin_state = getattr(context, "twin_state", {}) or {}
    twin_profile = twin_state.get("digital_twin_profile") or {}
    return str(twin_profile.get("summary") or "").strip()


def build_write_task_flow(tool_name: str, *, next_action: str = "", status: str = "completed") -> dict[str, Any]:
    template = _WRITE_TASK_TEMPLATES.get(
        tool_name,
        {
            "title": "记录写入任务",
            "prepare": "整理本次输入信息",
            "write": f"执行 {tool_name}",
            "refresh": "刷新相关状态",
        },
    )
    steps = [
        {"id": "prepare_input", "title": template["prepare"], "status": "done"},
        {"id": "write_record", "title": template["write"], "status": "done"},
    ]
    return build_generic_task_flow(template["title"], steps, status=status, next_action=next_action)


def build_risk_refresh_task_flow(context: Any | None, *, title: str = "风险刷新任务", next_action: str = "") -> dict[str, Any]:
    risk_overview = getattr(context, "risk_overview", {}) or {}
    risk_label = str(risk_overview.get("attack_risk_label") or "未知")
    trigger_summary = list(risk_overview.get("trigger_summary") or [])
    abnormal_items = list(risk_overview.get("abnormal_items") or [])
    reason_parts: list[str] = []
    if trigger_summary:
        reason_parts.append(f"近期风险因素 {trigger_summary[0].get('label')}")
    if abnormal_items:
        reason_parts.append(f"需要关注 {abnormal_items[0]}")
    refresh_next_action = next_action or build_context_next_action(context)
    if reason_parts and refresh_next_action:
        refresh_next_action = f"{'；'.join(reason_parts[:2])}；{refresh_next_action}"
    elif reason_parts:
        refresh_next_action = "；".join(reason_parts[:2])
    steps = [
        {"id": "load_risk_state", "title": "读取最新行为、症状与发作状态", "status": "done"},
        {"id": "recompute_risk", "title": f"刷新当前风险视图（当前风险：{risk_label}）", "status": "done"},
        {"id": "sync_twin", "title": "同步健康分身与后续建议", "status": "done"},
    ]
    return build_generic_task_flow(title, steps, status="completed", next_action=refresh_next_action)


def build_twin_refresh_task_flow(context: Any | None, *, title: str = "健康分身刷新任务", next_action: str = "") -> dict[str, Any]:
    twin_state = getattr(context, "twin_state", {}) or {}
    twin_profile = twin_state.get("digital_twin_profile") or {}
    focus_site = str(twin_profile.get("focus_site") or "").strip()
    summary = str(twin_profile.get("summary") or "").strip()
    refresh_next_action = next_action or build_context_next_action(context) or summary
    if focus_site and refresh_next_action:
        refresh_next_action = f"重点观察 {focus_site}；{refresh_next_action}"
    elif focus_site:
        refresh_next_action = f"重点观察 {focus_site}"
    steps = [
        {"id": "load_history", "title": "读取近期行为、疼痛和发作历史", "status": "done"},
        {"id": "rebuild_portrait", "title": "生成近期行为画像与个人痛风模式", "status": "done"},
        {"id": "sync_twin_state", "title": "同步健康分身中心状态", "status": "done"},
    ]
    return build_generic_task_flow(title, steps, status="completed", next_action=refresh_next_action)


def build_analysis_task_flow(
    skill_name: str,
    *,
    source: str = "",
    next_action: str = "",
    title: str = "分析任务",
) -> dict[str, Any]:
    source_label = "本地模型解释" if source == "local_llm" else "规则与状态回退"
    steps = [
        {"id": "load_state", "title": "读取当前健康分身、风险和近期记录", "status": "done"},
        {"id": "route_skill", "title": f"路由到 {skill_name} 技能", "status": "done"},
        {"id": "generate_answer", "title": f"基于 {source_label} 生成分析结果", "status": "done"},
    ]
    return build_generic_task_flow(title, steps, status="completed", next_action=next_action)


def build_background_job_task_flow(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        return {}
    result_payload = dict(job.get("result_payload") or {})
    task_flow = dict(result_payload.get("task_flow") or {})
    job_type = str(job.get("job_type") or "")
    payload = dict(job.get("payload") or {})
    raw_status = str(job.get("status") or "").strip().lower()
    status = {
        "pending": "queued",
        "queued": "queued",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
    }.get(raw_status, raw_status or "queued")
    if not task_flow:
        task_flow = _build_job_skeleton(job_type, payload)
    task_flow["status"] = status
    progress = dict(task_flow.get("progress") or {})
    steps = list(task_flow.get("steps") or [])
    total_steps = int(progress.get("total_steps") or len(steps))
    completed_steps = int(progress.get("completed_steps") or 0)
    if status == "queued":
        completed_steps = 0
        steps = _mark_job_steps(steps, "pending")
        if steps:
            steps[0]["status"] = "pending"
    elif status == "running":
        completed_steps = min(max(completed_steps, 1), max(total_steps - 1, 1)) if total_steps else 0
        steps = _mark_job_steps(steps, "pending")
        for index, step in enumerate(steps):
            if index < completed_steps:
                step["status"] = "done"
            elif index == completed_steps:
                step["status"] = "in_progress"
                break
    elif status == "failed":
        failed_index = min(completed_steps, max(len(steps) - 1, 0)) if steps else 0
        steps = _mark_job_steps(steps, "pending")
        for index, step in enumerate(steps):
            if index < failed_index:
                step["status"] = "done"
            elif index == failed_index:
                step["status"] = "failed"
                break
    else:
        completed_steps = total_steps
    task_flow["steps"] = steps
    task_flow["progress"] = build_progress(total_steps or len(steps), completed_steps)
    return task_flow


def _mark_job_steps(steps: list[dict[str, Any]], default_status: str) -> list[dict[str, Any]]:
    return [{**dict(step), "status": default_status} for step in steps]


def _build_job_skeleton(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if job_type == "report_generation":
        label = "周报" if str(payload.get("report_type") or "weekly") == "weekly" else "月报"
        return build_generic_task_flow(
            f"{label}生成任务",
            [
                {"id": "prepare", "title": "读取近期记录和健康分身状态", "status": "pending"},
                {"id": "generate", "title": f"生成{label}摘要与建议", "status": "pending"},
                {"id": "persist", "title": "保存报告摘要与解读结果", "status": "pending"},
            ],
            status="queued",
            next_action=f"{label}任务完成后，这里会显示最新摘要和 AI 解读。",
        )
    if job_type == "lab_report_parse":
        return build_generic_task_flow(
            "化验识别任务",
            [
                {"id": "prepare", "title": "读取上传文件", "status": "pending"},
                {"id": "parse", "title": "提取文本并识别关键指标", "status": "pending"},
                {"id": "persist", "title": "保存识别结果并准备 AI 解读", "status": "pending"},
            ],
            status="queued",
            next_action="识别完成后，这里会显示结合健康分身的化验解读。",
        )
    if job_type in {"care_plan_generation", "care_plan_replan"}:
        horizon_days = int(payload.get("horizon_days") or 7)
        action = "重新规划" if job_type == "care_plan_replan" else "生成"
        return build_generic_task_flow(
            f"{horizon_days}天管理计划{action}任务",
            [
                {"id": "prepare", "title": "读取健康分身、风险与近期行为", "status": "pending"},
                {"id": "plan", "title": f"{action}未来 {horizon_days} 天管理计划", "status": "pending"},
                {"id": "persist", "title": "保存计划运行状态与摘要", "status": "pending"},
            ],
            status="queued",
            next_action="计划准备好后，这里会显示阶段、步骤和执行进度。",
        )
    if job_type == "twin_refresh":
        return build_twin_refresh_task_flow(None)
    return build_generic_task_flow(
        "任务执行过程",
        [{"id": "prepare", "title": "等待任务开始", "status": "pending"}],
        status="queued",
    )


def merge_task_flows(title: str, flows: list[dict[str, Any]], *, next_action: str = "", status: str = "completed") -> dict[str, Any]:
    merged_steps: list[dict[str, Any]] = []
    for flow in flows:
        for step in list(flow.get("steps") or []):
            merged_steps.append(dict(step))
    if not merged_steps:
        merged_steps.append({"id": "empty", "title": "等待任务开始", "status": "pending"})
        status = "pending"
    return build_generic_task_flow(title, merged_steps, status=status, next_action=next_action)
