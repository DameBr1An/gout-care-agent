from __future__ import annotations


def summarize_medication_and_reminders(context: dict) -> str:
    adherence_rate = context.get("medication_completion_rate")
    reminder_count = context.get("active_reminder_count", 0)
    if adherence_rate is None:
        adherence_text = "近 7 天暂时没有足够的服药记录，建议尽快补齐打卡信息。"
    else:
        adherence_text = "近 7 天服药完成率约为 %.0f%%。" % adherence_rate
    return (
        "%s 当前启用中的提醒共有 %s 个，建议继续保持规律服药、按时复查和及时记录。"
        % (adherence_text, reminder_count)
    )
