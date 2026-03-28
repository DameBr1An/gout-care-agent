from __future__ import annotations

from typing import Any


def summarize_medication_and_reminders(context: dict) -> str:
    adherence_rate = context.get("medication_completion_rate")
    reminder_count = context.get("active_reminder_count", 0)
    if adherence_rate is None:
        adherence_text = "近 7 天暂时没有足够的服药记录，建议尽快补齐打卡。"
    else:
        adherence_text = f"近 7 天服药完成率约为 {adherence_rate:.0f}%。"
    return f"{adherence_text} 当前启用中的提醒共有 {reminder_count} 个，建议继续保持规律服药与按时记录。"


def prepare(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(context or {})


def run(action: str | None = None, *args, **kwargs) -> Any:
    context = args[0] if args else kwargs.get("context", {})
    if action in {None, "summarize_medication_and_reminders"}:
        return summarize_medication_and_reminders(context)
    raise ValueError(f"medication-followup-skill 不支持的运行动作：{action}")


def summarize(action: str | None = None, *args, **kwargs) -> str:
    return str(run(action, *args, **kwargs))


def persist(*args, **kwargs) -> None:
    return None
