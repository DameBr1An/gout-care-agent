from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass
class SkillRuntimeAdapter:
    skill_directory: str
    module: ModuleType

    def prepare(self, context: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        payload = dict(context or {})
        payload.update(kwargs)
        custom_prepare = getattr(self.module, "prepare", None)
        if callable(custom_prepare):
            return custom_prepare(payload)
        return payload

    def run(self, action: str | None = None, *args, **kwargs) -> Any:
        custom_run = getattr(self.module, "run", None)
        if callable(custom_run):
            return custom_run(action, *args, **kwargs)
        if action and hasattr(self.module, action):
            target = getattr(self.module, action)
            if callable(target):
                return target(*args, **kwargs)
        raise AttributeError(f"{self.skill_directory} runtime 不支持 run(action={action!r})")

    def summarize(self, action: str | None = None, *args, **kwargs) -> Any:
        custom_summarize = getattr(self.module, "summarize", None)
        if callable(custom_summarize):
            return custom_summarize(action, *args, **kwargs)
        if action and hasattr(self.module, action):
            target = getattr(self.module, action)
            if callable(target):
                return target(*args, **kwargs)
        raise AttributeError(f"{self.skill_directory} runtime 不支持 summarize(action={action!r})")

    def persist(self, *args, **kwargs) -> Any:
        custom_persist = getattr(self.module, "persist", None)
        if callable(custom_persist):
            return custom_persist(*args, **kwargs)
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.module, item)
