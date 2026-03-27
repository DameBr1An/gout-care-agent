from __future__ import annotations

import re
from typing import Any, Callable


LAB_PATTERNS: dict[str, tuple[str, ...]] = {
    "uric_acid": (r"尿酸", r"uric\s*acid", r"\bua\b"),
    "creatinine": (r"肌酐", r"creatinine", r"\bcr\b"),
    "egfr": (r"egfr", r"估算肾小球滤过率", r"肾小球滤过率"),
    "bun": (r"尿素氮", r"\bbun\b"),
    "alt": (r"\balt\b", r"谷丙转氨酶"),
    "ast": (r"\bast\b", r"谷草转氨酶"),
    "crp": (r"\bcrp\b", r"c-?reactive\s*protein", r"c反应蛋白"),
    "esr": (r"\besr\b", r"血沉"),
}

DISPLAY_LABELS = {
    "uric_acid": "尿酸",
    "creatinine": "肌酐",
    "egfr": "eGFR",
    "bun": "尿素氮",
    "alt": "ALT",
    "ast": "AST",
    "crp": "CRP",
    "esr": "ESR",
}


def parse_uploaded_lab_files(
    uploaded_files: list[dict[str, Any]],
    image_ocr_callback: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    file_summaries: list[dict[str, Any]] = []
    extracted_metrics: dict[str, dict[str, Any]] = {}
    extracted_texts: list[str] = []
    image_files: list[dict[str, Any]] = []

    for item in uploaded_files or []:
        name = str(item.get("name") or "未命名文件")
        mime_type = str(item.get("type") or "")
        content = item.get("bytes") or b""
        if isinstance(content, str):
            content = content.encode("utf-8", errors="ignore")
        if mime_type.startswith("image/"):
            image_files.append(item)

        text = _extract_text_from_bytes(content)
        metrics = _extract_metrics_from_text(text)
        if text:
            extracted_texts.append(text[:4000])
        for key, payload in metrics.items():
            extracted_metrics[key] = payload

        file_summaries.append(
            {
                "name": name,
                "type": mime_type or "未知",
                "has_text": bool(text.strip()),
                "metric_count": len(metrics),
            }
        )

    vision_result = None
    if not extracted_metrics and image_files and image_ocr_callback:
        vision_result = image_ocr_callback(image_files) or {}
        for key, payload in (vision_result.get("metrics") or {}).items():
            extracted_metrics[key] = payload

    summary_bits: list[str] = []
    if extracted_metrics:
        labels = [DISPLAY_LABELS.get(key, key) for key in extracted_metrics.keys()]
        summary_bits.append(f"识别到的指标包括：{'、'.join(labels[:5])}")
    if any(file_item["has_text"] for file_item in file_summaries):
        summary_bits.append("已从上传文件中提取到可解析文本")
    if vision_result and vision_result.get("used_vision"):
        summary_bits.append("已尝试用本地模型识别图片内容")
    if not summary_bits:
        summary_bits.append("暂未从上传文件中提取到明确指标")

    return {
        "files": file_summaries,
        "metrics": extracted_metrics,
        "summary": "；".join(summary_bits),
        "raw_text_preview": "\n".join(extracted_texts)[:6000],
        "used_vision": bool(vision_result and vision_result.get("used_vision")),
    }


def _extract_text_from_bytes(content: bytes) -> str:
    if not content:
        return ""
    candidates: list[tuple[int, str]] = []
    for encoding in ("utf-8", "utf-16", "gb18030", "latin-1"):
        try:
            decoded = content.decode(encoding, errors="ignore")
        except Exception:
            continue
        decoded = re.sub(r"\s+", " ", decoded)
        if len(decoded.strip()) >= 12:
            score = _score_decoded_text(decoded)
            candidates.append((score, decoded))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return candidates[0][1]


def _score_decoded_text(text: str) -> int:
    lowered = text.lower()
    score = 0
    for token in ("尿酸", "肌酐", "egfr", "uric acid", "creatinine", "crp", "esr", "alt", "ast"):
        if token in lowered or token in text:
            score += 5
    score += len(re.findall(r"[\u4e00-\u9fff]", text)) // 4
    score += len(re.findall(r"\d{2,4}(?:\.\d+)?", text))
    return score


def _extract_metrics_from_text(text: str) -> dict[str, dict[str, Any]]:
    if not text:
        return {}
    normalized = re.sub(r"\s+", " ", text)
    metrics: dict[str, dict[str, Any]] = {}
    for key, aliases in LAB_PATTERNS.items():
        for alias in aliases:
            pattern = re.compile(rf"(?:{alias})[^0-9]{{0,20}}(\d{{1,4}}(?:\.\d+)?)", re.IGNORECASE)
            match = pattern.search(normalized)
            if match:
                metrics[key] = {
                    "label": DISPLAY_LABELS.get(key, key),
                    "value": float(match.group(1)),
                    "source": "text_parse",
                }
                break
    return metrics
