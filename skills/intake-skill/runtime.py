from __future__ import annotations

import re
from datetime import date
from typing import Any


def parse_free_text_entry(text: str) -> dict:
    payload = {
        "log_date": str(date.today()),
        "weight_kg": None,
        "water_ml": None,
        "alcohol_intake": None,
        "diet_notes": text.strip() or None,
        "symptom_notes": None,
        "pain_score": None,
        "joint_pain_flag": False,
        "medication_taken_flag": None,
        "free_text": text.strip() or None,
    }

    normalized = text.lower()

    uric_match = re.search(r"尿酸\s*(\d+)", text)
    if uric_match:
        payload["uric_acid"] = float(uric_match.group(1))

    water_match = re.search(r"(\d{3,4})\s*(ml|毫升)", normalized)
    if water_match:
        payload["water_ml"] = float(water_match.group(1))

    weight_match = re.search(r"体重\s*(\d+(?:\.\d+)?)", text)
    if weight_match:
        payload["weight_kg"] = float(weight_match.group(1))

    pain_match = re.search(r"(疼痛|痛感|疼)\s*(\d+)", text)
    if pain_match:
        payload["pain_score"] = int(pain_match.group(2))
        payload["joint_pain_flag"] = int(pain_match.group(2)) > 0

    if "啤酒" in text or "beer" in normalized:
        payload["alcohol_intake"] = "beer"
    elif "葡萄酒" in text or "wine" in normalized:
        payload["alcohol_intake"] = "wine"
    elif "烈酒" in text or "spirits" in normalized:
        payload["alcohol_intake"] = "spirits"
    elif "喝酒" in text or "饮酒" in text:
        payload["alcohol_intake"] = "other"

    if any(keyword in text for keyword in ("疼", "肿", "红", "关节")):
        payload["symptom_notes"] = text.strip()
        payload["joint_pain_flag"] = True

    if "已服药" in text or "吃药了" in text:
        payload["medication_taken_flag"] = True
    elif "没服药" in text or "漏服" in text:
        payload["medication_taken_flag"] = False

    return payload


def prepare(context: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(context or {})


def run(action: str | None = None, *args, **kwargs) -> Any:
    if action in {None, "parse_free_text_entry"}:
        return parse_free_text_entry(*args, **kwargs)
    raise ValueError(f"intake-skill 不支持的运行动作：{action}")


def summarize(action: str | None = None, *args, **kwargs) -> str:
    if action in {None, "parse_free_text_entry"}:
        payload = parse_free_text_entry(*args, **kwargs)
        parts: list[str] = []
        if payload.get("water_ml") is not None:
            parts.append(f"饮水约 {int(payload['water_ml'])} mL")
        if payload.get("alcohol_intake"):
            parts.append(f"饮酒类型：{payload['alcohol_intake']}")
        if payload.get("pain_score") is not None:
            parts.append(f"疼痛 {int(payload['pain_score'])} 分")
        if payload.get("joint_pain_flag"):
            parts.append("提到了关节疼痛")
        return "；".join(parts) if parts else "这是一条待写入的日常或症状描述。"
    raise ValueError(f"intake-skill 不支持的摘要动作：{action}")


def persist(*args, **kwargs) -> None:
    return None
