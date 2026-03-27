from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "FreedomIntelligence/HuatuoGPT-o1-7B"
DEFAULT_TIMEOUT = 60


@dataclass
class LocalLLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int


@dataclass
class LocalLLMResult:
    ok: bool
    content: str
    used_model: str
    error_message: str | None = None


def get_local_llm_config() -> LocalLLMConfig:
    timeout_text = os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT))
    try:
        timeout_seconds = int(timeout_text)
    except ValueError:
        timeout_seconds = DEFAULT_TIMEOUT

    return LocalLLMConfig(
        base_url=os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"),
        model=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
        timeout_seconds=max(timeout_seconds, 5),
    )


def get_local_llm_status() -> dict[str, Any]:
    config = get_local_llm_config()
    return {
        "base_url": config.base_url,
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
    }


def build_gout_messages(question: str, context: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "你是一个面向痛风与高尿酸血症长期管理场景的中文健康助手。"
        "你的职责是帮助用户记录情况、解释变化、提示风险并给出日常管理建议，但不能替代医生诊断。"
        "回答时要结合用户当前的尿酸、症状、饮水、诱因、用药、历史画像和规则引擎结果。"
        "表达风格要像正式健康管理产品：语气稳定、清楚、克制，先给结论，再给原因和建议。"
        "如果出现剧烈疼痛、明显红肿、发热、持续恶化、肾功能异常或潜在用药风险，请明确建议及时线下就医。"
        "不要编造检查结果、药物剂量、医生意见或诊断结论。"
        "输出尽量使用简洁自然的中文，优先给出可执行的下一步。"
    )
    user_prompt = (
        "以下是用户当前的结构化管理信息，请基于这些信息回答最后的问题。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n用户问题："
        + question
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_local_openai_compatible(messages: list[dict[str, str]]) -> LocalLLMResult:
    config = get_local_llm_config()
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 700,
    }

    req = request.Request(
        url=config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.api_key,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return LocalLLMResult(
                ok=False,
                content="",
                used_model=config.model,
                error_message="本地模型接口返回成功，但没有生成内容。",
            )
        return LocalLLMResult(ok=True, content=content, used_model=config.model)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        return LocalLLMResult(
            ok=False,
            content="",
            used_model=config.model,
            error_message="本地模型接口 HTTP 错误：%s %s" % (exc.code, detail[:300]),
        )
    except error.URLError as exc:
        return LocalLLMResult(
            ok=False,
            content="",
            used_model=config.model,
            error_message="无法连接到本地模型接口：%s" % exc,
        )
    except Exception as exc:
        return LocalLLMResult(
            ok=False,
            content="",
            used_model=config.model,
            error_message="调用本地模型时发生异常：%s" % exc,
        )


def ask_local_gout_llm(question: str, context: dict[str, Any]) -> LocalLLMResult:
    messages = build_gout_messages(question, context)
    return call_local_openai_compatible(messages)


def ask_local_lab_vision_llm(file_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    image_items = []
    for item in file_payloads:
        mime_type = str(item.get("type") or "")
        content = item.get("bytes") or b""
        if not mime_type.startswith("image/") or not content:
            continue
        if isinstance(content, str):
            content = content.encode("utf-8", errors="ignore")
        data_url = f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
        image_items.append({"type": "image_url", "image_url": {"url": data_url}})

    if not image_items:
        return {"used_vision": False, "metrics": {}, "error": "没有可供识别的图片文件。"}

    prompt = (
        "请识别这些化验报告图片中的关键指标，并只输出 JSON。"
        "可识别字段包括：uric_acid、creatinine、egfr、bun、alt、ast、crp、esr。"
        "JSON 格式示例："
        '{"metrics":{"uric_acid":{"label":"尿酸","value":430},"creatinine":{"label":"肌酐","value":85}}}'
        "如果没有识别到指标，请输出 {\"metrics\":{}}。"
    )
    messages = [
        {"role": "system", "content": "你是一个化验报告结构化提取助手，只能输出 JSON。"},
        {"role": "user", "content": [{"type": "text", "text": prompt}, *image_items]},
    ]
    result = call_local_openai_compatible(messages)  # type: ignore[arg-type]
    if not result.ok:
        return {"used_vision": True, "metrics": {}, "error": result.error_message}
    payload = _extract_json_payload(result.content)
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    return {"used_vision": True, "metrics": metrics or {}, "error": None}


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
