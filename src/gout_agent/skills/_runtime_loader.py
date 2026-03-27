from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def load_runtime_module(skill_directory: str):
    runtime_path = _project_root() / "skills" / skill_directory / "runtime.py"
    if not runtime_path.exists():
        raise FileNotFoundError(f"未找到 Skill 运行时文件：{runtime_path}")

    spec = spec_from_file_location(
        f"gout_agent.skill_runtime.{skill_directory.replace('-', '_')}",
        runtime_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Skill 运行时文件：{runtime_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
