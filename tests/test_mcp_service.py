from __future__ import annotations

import asyncio
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent.mcp_service import create_app, create_mcp_server


class MCPServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / "tests_tmp" / ("mcp_" + uuid.uuid4().hex)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PROJECT_ROOT / "skills", self.temp_root / "skills")
        self.server = create_mcp_server(self.temp_root)
        self.client = TestClient(create_app(self.temp_root))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_streamable_http_route_exists(self) -> None:
        route_paths = {route.path for route in self.client.app.routes}
        self.assertIn("/mcp", route_paths)

    def test_tools_list_contains_real_mcp_tools(self) -> None:
        tools = asyncio.run(self.server.list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("get_user_profile", tool_names)
        self.assertIn("run_agent_loop", tool_names)
        profile_tool = next(tool for tool in tools if tool.name == "update_user_profile")
        self.assertIn("payload", profile_tool.inputSchema["properties"])

    def test_agent_loop_tool_returns_steps(self) -> None:
        _content, payload = asyncio.run(self.server.call_tool("run_agent_loop", {"question": "最近为什么风险升高了？", "max_steps": 5}))
        self.assertIn("agent_loop", payload)
        self.assertTrue(payload["agent_loop"]["steps"])

if __name__ == "__main__":
    unittest.main()
