from __future__ import annotations

from datetime import datetime
from typing import Any


def prepare(
    context: dict[str, Any] | None = None,
    *,
    horizon_days: int | None = None,
    twin_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(context or {})
    if horizon_days is not None:
        payload["plan_horizon_days"] = int(horizon_days)
    if twin_state is not None:
        payload["twin_state"] = twin_state
    return payload


def run(action: str | None = None, *args, **kwargs) -> Any:
    if action in {None, "build_care_plan"}:
        return build_care_plan(*args, **kwargs)
    if action == "evaluate_care_plan":
        return evaluate_care_plan(*args, **kwargs)
    raise ValueError(f"care-plan-skill does not support run(action={action!r})")


def summarize(action: str | None = None, *args, **kwargs) -> Any:
    if action in {None, "summarize_care_plan"}:
        return summarize_care_plan(*args, **kwargs)
    raise ValueError(f"care-plan-skill does not support summarize(action={action!r})")


def persist(*args, **kwargs) -> None:
    return None


def build_care_plan(context: dict[str, Any], *, horizon_days: int = 7) -> dict[str, Any]:
    twin_state = context.get("twin_state") or {}
    twin_profile = twin_state.get("digital_twin_profile") or {}
    portraits = twin_state.get("behavior_portraits") or {}
    portrait_key = "30d" if horizon_days >= 30 else "7d"
    portrait = portraits.get(portrait_key) or portraits.get("7d") or {}
    risk_overview = context.get("current_risk_overview") or context.get("risk_overview") or {}
    site_history = context.get("site_history_preview") or context.get("site_history") or []

    attack_risk = str(risk_overview.get("attack_risk_label") or "中等")
    focus_site = _pick_focus_site(twin_profile, site_history)
    focus_site_triggers = _focus_site_triggers(twin_profile, focus_site)
    phases = _build_phases(horizon_days)
    steps = _build_plan_steps(horizon_days, portrait, attack_risk, focus_site, focus_site_triggers)
    progress = _calculate_progress(steps)
    summary = _build_summary(horizon_days, attack_risk, focus_site, focus_site_triggers, steps)

    return {
        "plan_type": "30d" if horizon_days >= 30 else "7d",
        "horizon_days": horizon_days,
        "summary": summary,
        "phases": phases,
        "steps": steps,
        "progress": progress,
        "key_goals": [step["title"] for step in steps[:3]],
        "today_actions": [step["description"] for step in steps[:3]],
        "focus_site": focus_site,
        "focus_site_reason": _build_focus_site_reason(focus_site, focus_site_triggers),
        "review_timing": _build_review_timing(horizon_days, attack_risk),
        "update_plan": _build_update_plan(horizon_days),
        "failure_adjustments": [],
        "replan_reason": "",
        "status": "active",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def evaluate_care_plan(plan_payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    updated = dict(plan_payload or {})
    steps = [dict(step) for step in (updated.get("steps") or [])]
    if not steps:
        updated["progress"] = {"total_steps": 0, "completed_steps": 0, "completion_rate": 0}
        updated["status"] = "active"
        return updated

    portraits = ((context.get("twin_state") or {}).get("behavior_portraits") or {})
    portrait = portraits.get("7d") or portraits.get("30d") or {}
    site_history = context.get("site_history_preview") or context.get("site_history") or []
    risk_overview = context.get("current_risk_overview") or context.get("risk_overview") or {}
    auto_reasons: list[str] = []
    failure_adjustments: list[str] = []

    for step in steps:
        current_status = str(step.get("status") or "pending")
        if current_status in {"done", "failed"}:
            if current_status == "failed":
                failure_adjustments.append(str(step.get("failure_hint") or f"先把{step.get('title') or '当前步骤'}拆成更小动作。"))
            continue

        auto_rule = step.get("auto_rule") or {}
        auto_done, reason = _evaluate_auto_rule(auto_rule, portrait, site_history, risk_overview, updated)
        if auto_done:
            step["status"] = "done"
            step["completion_source"] = "auto"
            if reason:
                auto_reasons.append(reason)
            continue

    progress = _calculate_progress(steps)
    status = "completed" if progress["completed_steps"] >= progress["total_steps"] and progress["total_steps"] > 0 else "active"
    replan_reason = _detect_replan_reason(updated, context)
    if status != "completed" and replan_reason:
        status = "needs_replan"
    if status != "completed" and any(step.get("status") == "failed" for step in steps):
        status = "needs_adjustment"

    updated["steps"] = steps
    updated["progress"] = progress
    updated["status"] = status
    updated["auto_completion_reasons"] = auto_reasons[:4]
    updated["failure_adjustments"] = _dedupe(failure_adjustments)[:4]
    updated["replan_reason"] = replan_reason
    updated["summary"] = _refresh_summary(updated)
    updated["last_evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    return updated


def summarize_care_plan(plan_payload: dict[str, Any]) -> str:
    if not isinstance(plan_payload, dict):
        return "管理计划暂时不可用。"
    progress = plan_payload.get("progress") or {}
    completion_rate = int(progress.get("completion_rate") or 0)
    parts = [str(plan_payload.get("summary") or "").strip()]
    if completion_rate:
        parts.append(f"当前完成进度约 {completion_rate}%")
    focus_site = str(plan_payload.get("focus_site") or "").strip()
    if focus_site:
        parts.append(f"重点观察 {focus_site}")
    review_timing = str(plan_payload.get("review_timing") or "").strip()
    if review_timing:
        parts.append(review_timing)
    return "；".join(part for part in parts if part)


def _build_phases(horizon_days: int) -> list[dict[str, Any]]:
    if horizon_days >= 30:
        return [
            {"id": "stabilize", "title": "第 1 周先稳住当前状态", "window": "Day 1-7"},
            {"id": "observe", "title": "第 2-3 周持续观察重点部位", "window": "Day 8-21"},
            {"id": "review", "title": "第 4 周复盘并准备下一轮调整", "window": "Day 22-30"},
        ]
    return [
        {"id": "stabilize", "title": "前 2 天先稳住当前状态", "window": "Day 1-2"},
        {"id": "observe", "title": "接下来几天持续观察变化", "window": "Day 3-5"},
        {"id": "review", "title": "最后做一次阶段复盘", "window": "Day 6-7"},
    ]


def _build_plan_steps(
    horizon_days: int,
    portrait: dict[str, Any],
    attack_risk: str,
    focus_site: str,
    focus_site_triggers: list[str],
) -> list[dict[str, Any]]:
    hydration_target = 1800 if attack_risk == "高" else 1600
    med_target = 85 if horizon_days >= 30 else 80
    trigger_text = "、".join(focus_site_triggers[:2]) or "高风险诱因"
    steps = [
        {
            "id": "hydration",
            "phase_id": "stabilize",
            "title": "先把饮水恢复到稳定水平",
            "description": f"未来几天尽量把日均饮水提升到 {hydration_target} mL 以上，先把明显偏低的状态拉回来。",
            "status": "in_progress",
            "auto_rule": {"type": "water_ml_min", "value": hydration_target},
            "failure_hint": "如果连续两天还是偏低，可以把喝水拆成早中晚三段完成。",
        },
        {
            "id": "medication",
            "phase_id": "stabilize",
            "title": "把服药执行恢复到稳定节奏",
            "description": f"优先保证近期服药完成率达到 {med_target}% 左右，先避免继续漏服。",
            "status": "pending",
            "auto_rule": {"type": "medication_rate_min", "value": med_target},
            "failure_hint": "如果总是忘记服药，先把服药时间固定在每天同一时段。",
        },
        {
            "id": "focus_site_observation",
            "phase_id": "observe",
            "title": f"重点观察 {focus_site}",
            "description": f"这段时间重点留意 {focus_site} 的疼痛、红肿和僵硬变化，并尽量减少 {trigger_text}。",
            "status": "pending",
            "auto_rule": {"type": "site_observed", "site": focus_site},
            "failure_hint": f"如果 {focus_site} 仍反复不适，先补充更连续的疼痛记录，方便下一轮计划更准确。",
        },
        {
            "id": "trigger_control",
            "phase_id": "observe",
            "title": "把近期最常见诱因压下来",
            "description": f"这轮计划里先重点压低 {trigger_text} 的暴露频率，优先看能不能减少明显诱因。",
            "status": "pending",
            "auto_rule": {"type": "alcohol_days_max", "value": 0 if attack_risk == "高" else 1},
            "failure_hint": "如果完全避免有困难，先从减少频率和避免叠加诱因开始。",
        },
        {
            "id": "review",
            "phase_id": "review",
            "title": "阶段复盘并更新下一轮计划",
            "description": "完成这一轮后，再根据新的行为和部位变化刷新计划，确认哪些动作最有效。",
            "status": "pending",
            "auto_rule": {"type": "completion_rate_min", "value": 60 if horizon_days >= 30 else 50},
            "failure_hint": "如果执行进度一直很低，下一轮计划先缩短周期、减少同时推进的目标。",
        },
    ]
    if horizon_days >= 30:
        steps.insert(
            4,
            {
                "id": "mid_review",
                "phase_id": "review",
                "title": "第 2-3 周做一次中途校准",
                "description": "如果风险或重点部位发生变化，就在中途重规划，避免整轮计划失真。",
                "status": "pending",
                "auto_rule": {"type": "risk_not_high"},
                "failure_hint": "如果中途风险反而上升，优先缩短目标周期并重新生成一轮 7 天计划。",
            },
        )
    return steps


def _calculate_progress(steps: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(steps)
    completed = sum(1 for step in steps if str(step.get("status")) == "done")
    return {
        "total_steps": total,
        "completed_steps": completed,
        "completion_rate": int(round((completed / total) * 100)) if total else 0,
    }


def _pick_focus_site(twin_profile: dict[str, Any], site_history: list[dict[str, Any]]) -> str:
    site_patterns = twin_profile.get("site_pain_patterns") or {}
    if site_patterns:
        focus_site, _ = max(
            site_patterns.items(),
            key=lambda item: (
                float((item[1] or {}).get("average_pain_score") or 0),
                int((item[1] or {}).get("attack_count") or 0),
            ),
        )
        return str(focus_site)
    for item in site_history:
        site = str(item.get("site") or "").strip()
        if site:
            return site
    return "左脚大脚趾"


def _focus_site_triggers(twin_profile: dict[str, Any], focus_site: str) -> list[str]:
    site_trigger_map = twin_profile.get("site_trigger_map") or {}
    top_triggers = twin_profile.get("top_triggers") or []
    site_triggers = list(site_trigger_map.get(focus_site) or [])
    if site_triggers:
        return [str(item) for item in site_triggers if str(item).strip()]
    labels = [str(item.get("label") or "").strip() for item in top_triggers if isinstance(item, dict)]
    return [label for label in labels if label]


def _build_focus_site_reason(focus_site: str, focus_site_triggers: list[str]) -> str:
    trigger_text = "、".join(focus_site_triggers[:2]) or "近期高风险因素"
    return f"{focus_site} 最近更容易受到 {trigger_text} 的影响，适合作为这一轮计划的重点观察部位。"


def _build_review_timing(horizon_days: int, attack_risk: str) -> str:
    if attack_risk == "高":
        return "建议在 3-5 天内先复盘一次，如果疼痛或红肿加重，尽快线下就医。"
    if horizon_days >= 30:
        return "建议每 7-10 天做一次小复盘，到第 30 天再做完整复盘。"
    return "建议在 5-7 天后复盘一次，确认这一轮动作是否让风险和部位症状稳定下来。"


def _build_update_plan(horizon_days: int) -> str:
    if horizon_days >= 30:
        return "连续执行后，系统会根据新的行为记录、风险变化和重点部位状态，自动判断是否需要中途重规划或进入下一轮 30 天计划。"
    return "连续执行几天后，系统会根据新的记录判断哪些步骤已完成、哪些需要重规划，并生成下一轮更贴近当前状态的短周期计划。"


def _build_summary(
    horizon_days: int,
    attack_risk: str,
    focus_site: str,
    focus_site_triggers: list[str],
    steps: list[dict[str, Any]],
) -> str:
    trigger_text = "、".join(focus_site_triggers[:2]) or "近期高风险因素"
    first_goal = steps[0]["title"] if steps else "先把当前状态稳住"
    return f"这是一轮面向未来 {horizon_days} 天的渐进式管理计划。当前发作风险为{attack_risk}，先从“{first_goal}”开始，再持续观察 {focus_site}，并尽量压低 {trigger_text} 的影响。"


def _evaluate_auto_rule(
    auto_rule: dict[str, Any],
    portrait: dict[str, Any],
    site_history: list[dict[str, Any]],
    risk_overview: dict[str, Any],
    plan_payload: dict[str, Any],
) -> tuple[bool, str]:
    rule_type = str(auto_rule.get("type") or "").strip()
    if not rule_type:
        return False, ""

    if rule_type == "water_ml_min":
        avg_water = float(portrait.get("average_water_ml") or 0)
        threshold = float(auto_rule.get("value") or 0)
        return avg_water >= threshold, f"近期平均饮水已达到约 {avg_water:.0f} mL/天。"

    if rule_type == "medication_rate_min":
        med_rate = float(portrait.get("medication_taken_rate") or 0)
        threshold = float(auto_rule.get("value") or 0)
        return med_rate >= threshold, f"近期服药完成率约 {med_rate:.0f}%。"

    if rule_type == "site_observed":
        site = str(auto_rule.get("site") or "").strip()
        observed = any(str(item.get("site") or "").strip() == site for item in site_history)
        return observed, f"近期已经补充过 {site} 的部位记录。"

    if rule_type == "alcohol_days_max":
        alcohol_days = int(portrait.get("alcohol_days") or 0)
        threshold = int(auto_rule.get("value") or 0)
        return alcohol_days <= threshold, f"近期饮酒天数已压到 {alcohol_days} 天。"

    if rule_type == "completion_rate_min":
        progress = plan_payload.get("progress") or {}
        current_rate = int(progress.get("completion_rate") or 0)
        threshold = int(auto_rule.get("value") or 0)
        return current_rate >= threshold, f"计划执行进度已达到 {current_rate}%。"

    if rule_type == "risk_not_high":
        attack_risk = str(risk_overview.get("attack_risk_label") or "")
        return attack_risk != "高", f"当前发作风险已不是高风险。"

    return False, ""


def _detect_replan_reason(plan_payload: dict[str, Any], context: dict[str, Any]) -> str:
    risk_overview = context.get("current_risk_overview") or context.get("risk_overview") or {}
    site_history = context.get("site_history_preview") or context.get("site_history") or []
    current_focus_site = str(plan_payload.get("focus_site") or "").strip()
    current_risk = str(risk_overview.get("attack_risk_label") or "")
    latest_site = ""
    for item in site_history:
        latest_site = str(item.get("site") or "").strip()
        if latest_site:
            break
    if current_risk == "高" and latest_site and latest_site != current_focus_site:
        return f"最近重点不适部位已经从 {current_focus_site} 转向 {latest_site}，建议重新规划。"
    if current_risk == "高" and any(str(item.get("event_type") or "") == "attack" for item in site_history[:3]):
        return "计划执行过程中仍出现了新的发作记录，建议切换到更短周期计划。"
    return ""


def _refresh_summary(plan_payload: dict[str, Any]) -> str:
    progress = plan_payload.get("progress") or {}
    rate = int(progress.get("completion_rate") or 0)
    focus_site = str(plan_payload.get("focus_site") or "重点部位").strip()
    status = str(plan_payload.get("status") or "active")
    if status == "completed":
        return f"这轮计划已完成，当前重点部位 {focus_site} 已完成本轮观察，接下来可以进入下一轮复盘或长期维护。"
    if status == "needs_replan":
        return f"这轮计划需要中途重规划。当前重点部位仍是 {focus_site}，但近期状态已经发生变化。"
    if status == "needs_adjustment":
        return f"这轮计划执行受阻，目前完成进度约 {rate}%。建议先缩小目标，再继续推进。"
    return f"这轮计划正在执行中，当前完成进度约 {rate}%，接下来仍需持续观察 {focus_site}。"


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result
