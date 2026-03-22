from __future__ import annotations

import re
from datetime import date


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

    pain_match = re.search(r"疼(?:痛)?\s*(\d+)", text)
    if pain_match:
        payload["pain_score"] = int(pain_match.group(1))
        payload["joint_pain_flag"] = int(pain_match.group(1)) > 0

    if "啤酒" in text or "beer" in normalized:
        payload["alcohol_intake"] = "beer"
    elif "葡萄酒" in text or "wine" in normalized:
        payload["alcohol_intake"] = "wine"
    elif "烈酒" in text or "spirits" in normalized:
        payload["alcohol_intake"] = "spirits"
    elif "喝酒" in text or "饮酒" in text:
        payload["alcohol_intake"] = "other"

    if "痛" in text or "肿" in text or "红" in text:
        payload["symptom_notes"] = text.strip()
        payload["joint_pain_flag"] = True

    if "已服药" in text or "吃药了" in text:
        payload["medication_taken_flag"] = True
    elif "没服药" in text or "漏服" in text:
        payload["medication_taken_flag"] = False

    return payload