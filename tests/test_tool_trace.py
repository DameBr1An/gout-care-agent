from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent.toolkit import build_default_tool_registry


class ToolTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / "tests_tmp" / ("tool_trace_" + uuid.uuid4().hex)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_registry_records_successful_trace(self) -> None:
        registry = build_default_tool_registry(self.temp_root)
        profile = registry.call(
            "获取用户档案",
            _trace_context={"route_name": "profile", "skill_name": "profile-skill", "source": "test"},
        )
        self.assertIn("name", profile)

        traces = registry.get_traces(5)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["tool_name"], "获取用户档案")
        self.assertTrue(traces[0]["success"])
        self.assertEqual(traces[0]["route_name"], "profile")


if __name__ == "__main__":
    unittest.main()
