from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "何时使用": ("何时使用", "When To Use"),
    "核心职责": ("核心职责", "Core Responsibilities"),
    "执行步骤": ("执行步骤", "Steps"),
    "输出规则": ("输出规则", "Output Rules"),
    "决策提示词": ("决策提示词", "Decision Prompt"),
}


@dataclass
class SkillSpec:
    name: str
    route_name: str
    description: str
    module: str | None
    directory: str
    skill_file: str
    recommended_tools: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    route_hints: list[str] = field(default_factory=list)
    execution_steps: list[str] = field(default_factory=list)
    execution_tools: list[str] = field(default_factory=list)
    when_to_use: str = ""
    decision_prompt: str = ""
    write_permissions: list[str] = field(default_factory=list)


class SkillRegistry:
    def __init__(self, skills: list[SkillSpec]) -> None:
        self._skills = {skill.name: skill for skill in skills}
        self._route_index = {skill.route_name: skill for skill in skills}

    def list(self) -> list[SkillSpec]:
        return [self._skills[name] for name in sorted(self._skills.keys())]

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def get_by_route(self, route_name: str) -> SkillSpec | None:
        return self._route_index.get(route_name)

    def get_allowed_tools(self, route_name: str) -> list[str]:
        skill = self.get_by_route(route_name)
        return list(skill.recommended_tools) if skill else []

    def get_execution_steps(self, route_name: str) -> list[str]:
        skill = self.get_by_route(route_name)
        return list(skill.execution_steps) if skill else []

    def get_execution_tools(self, route_name: str) -> list[str]:
        skill = self.get_by_route(route_name)
        return list(skill.execution_tools) if skill else []

    def get_decision_prompt(self, route_name: str) -> str:
        skill = self.get_by_route(route_name)
        return skill.decision_prompt if skill else ""

    def describe(self) -> list[dict[str, Any]]:
        return [serialize_skill(skill) for skill in self.list()]

    def match_question(self, question: str) -> dict[str, Any] | None:
        normalized_question = _normalize_text(question)
        best_match: dict[str, Any] | None = None

        for skill in self.list():
            if skill.route_name == "orchestrator":
                continue
            score, matched_hints = _score_skill(skill, normalized_question)
            if score <= 0:
                continue
            candidate = {
                "name": skill.name,
                "route_name": skill.route_name,
                "score": score,
                "matched_hints": matched_hints[:6],
                "allowed_tools": list(skill.recommended_tools),
                "execution_tools": list(skill.execution_tools),
                "write_permissions": list(skill.write_permissions),
            }
            if best_match is None or candidate["score"] > best_match["score"]:
                best_match = candidate

        return best_match


def serialize_skill(skill: SkillSpec) -> dict[str, Any]:
    payload = asdict(skill)
    payload["section_names"] = list(skill.sections.keys())
    payload["allowed_tools"] = list(skill.recommended_tools)
    return payload


def load_skill_registry(skills_root: Path) -> SkillRegistry:
    skills: list[SkillSpec] = []
    if not skills_root.exists():
        return SkillRegistry(skills)

    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        skills.append(parse_skill_file(skill_file, skills_root))
    return SkillRegistry(skills)


def parse_skill_file(skill_file: Path, skills_root: Path) -> SkillSpec:
    text = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    meta = _parse_frontmatter(frontmatter)
    sections = _canonicalize_sections(_parse_sections(body))
    name = str(meta.get("name") or skill_file.parent.name)
    description = str(meta.get("description") or "")
    recommended_tools = _as_string_list(meta.get("recommended_tools"))
    route_name = _infer_route_name(name)
    execution_steps = _parse_execution_steps(sections)
    execution_tools = _infer_execution_tools(recommended_tools, execution_steps)
    route_hints = _build_route_hints(name, description, recommended_tools, sections)
    when_to_use = sections.get("何时使用", "").strip()
    return SkillSpec(
        name=name,
        route_name=route_name,
        description=description,
        module=str(meta.get("module")) if meta.get("module") else None,
        directory=str(skill_file.parent.relative_to(skills_root)),
        skill_file=str(skill_file.relative_to(skills_root.parent)),
        recommended_tools=recommended_tools,
        sections=sections,
        route_hints=route_hints,
        execution_steps=execution_steps,
        execution_tools=execution_tools,
        when_to_use=when_to_use,
        decision_prompt=sections.get("决策提示词", "").strip(),
        write_permissions=_infer_write_permissions(recommended_tools),
    )


def _split_frontmatter(text: str) -> tuple[str, str]:
    text = text.lstrip("\ufeff")
    if not text.startswith("---\n"):
        return "", text
    marker = "\n---\n"
    end = text.find(marker, 4)
    if end == -1:
        return "", text
    return text[4:end], text[end + len(marker) :]


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        clean_value = value.strip()
        result[current_key] = [] if clean_value == "" else clean_value
    return result


def _parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("# "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line[2:].strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()
    return sections


def _canonicalize_sections(sections: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for heading, content in sections.items():
        canonical = heading
        for target, aliases in SECTION_ALIASES.items():
            if heading in aliases:
                canonical = target
                break
        normalized[canonical] = content
    return normalized


def _parse_execution_steps(sections: dict[str, str]) -> list[str]:
    raw_text = sections.get("执行步骤") or ""
    if not raw_text:
        return []

    steps: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line:
            steps.append(line)
    return steps


def _infer_execution_tools(recommended_tools: list[str], execution_steps: list[str]) -> list[str]:
    ordered_tools: list[str] = []
    for step in execution_steps:
        for tool_name in recommended_tools:
            if tool_name in step and tool_name not in ordered_tools:
                ordered_tools.append(tool_name)
    return ordered_tools


def _infer_write_permissions(recommended_tools: list[str]) -> list[str]:
    return [
        tool_name
        for tool_name in recommended_tools
        if any(token in tool_name for token in ("记录", "更新", "保存", "创建", "添加", "导出"))
    ]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _infer_route_name(name: str) -> str:
    mapping = {
        "orchestrator-skill": "orchestrator",
        "profile-skill": "profile",
        "intake-skill": "intake",
        "risk-assessment-skill": "risk_assessment",
        "lifestyle-coach-skill": "lifestyle_coach",
        "medication-followup-skill": "medication_followup",
        "report-explanation-skill": "reporting",
        "lab-report-skill": "lab_report",
        "care-plan-skill": "care_plan",
    }
    return mapping.get(name, name.replace("-skill", "").replace("-", "_"))


def _build_route_hints(
    name: str,
    description: str,
    recommended_tools: list[str],
    sections: dict[str, str],
) -> list[str]:
    raw_chunks: list[str] = [name, description]
    raw_chunks.extend(recommended_tools)
    for key in ("何时使用", "核心职责"):
        if key in sections:
            raw_chunks.append(sections[key])

    hints: list[str] = []
    for chunk in raw_chunks:
        for hint in _extract_hints(chunk):
            if hint not in hints:
                hints.append(hint)
    return hints


def _extract_hints(text: str) -> list[str]:
    hints: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\d\.\s]+", "", line)
        line = line.strip("，。！：；（）()[]【】")
        if len(line) >= 2:
            hints.append(line)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_-]*|[\u4e00-\u9fff]{2,}", line):
            if len(token) >= 2:
                hints.append(token)
                hints.extend(_expand_cjk_token(token))
    return hints


def _normalize_text(text: str) -> str:
    return str(text or "").strip().lower()


def _expand_cjk_token(token: str) -> list[str]:
    if not re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
        return []
    fragments: list[str] = []
    max_width = min(4, len(token))
    for width in range(2, max_width + 1):
        for start in range(0, len(token) - width + 1):
            fragment = token[start : start + width]
            if fragment not in fragments:
                fragments.append(fragment)
    return fragments


def _score_skill(skill: SkillSpec, normalized_question: str) -> tuple[int, list[str]]:
    score = 0
    matched_hints: list[str] = []

    for hint in skill.route_hints:
        normalized_hint = _normalize_text(hint)
        if len(normalized_hint) < 2:
            continue
        if normalized_hint in normalized_question:
            score += min(len(normalized_hint), 8)
            matched_hints.append(hint)

    for tool_name in skill.recommended_tools:
        simplified = tool_name
        for prefix in ("获取", "记录", "生成", "保存", "更新", "创建", "调用", "导出", "识别", "解析"):
            simplified = simplified.replace(prefix, "")
        normalized_tool = _normalize_text(simplified)
        if len(normalized_tool) >= 2 and normalized_tool in normalized_question:
            score += 2
            matched_hints.append(tool_name)

    return score, matched_hints
