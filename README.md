# AI 痛风管理助手

这是一个面向高尿酸血症与痛风长期管理场景的本地优先 Agent MVP，核心目标是帮助用户完成记录、风险预警、个性化建议和长期陪伴管理。

## 当前能力

- 本地 SQLite 数据存储
- 用户档案与健康档案管理
- 每日健康记录
- 化验结果记录
- 痛风发作追踪
- 药物与提醒管理
- 基于规则引擎的尿酸风险与发作风险评估
- 周报与月报生成、导出与解读
- 中文 Streamlit 多页面界面
- 本地 `HuatuoGPT-o1-7B` OpenAI-compatible 适配层
- 本地模型不可用时自动回退到规则引擎
- 轻量 Skill 架构与 `SKILL.md` 驱动
- 真实 MCP 服务，基于官方 `FastMCP`

## 主要目录

- `streamlit_app.py`：Streamlit 入口
- `src/gout_agent/data.py`：SQLite 数据层
- `src/gout_agent/risk.py`：风险评估与异常识别
- `src/gout_agent/reporting.py`：周报、月报与导出
- `src/gout_agent/llm.py`：本地模型适配层
- `src/gout_agent/ui.py`：中文前端界面
- `src/gout_agent/toolkit.py`：内部工具注册表
- `src/gout_agent/skill_registry.py`：`SKILL.md` 解析与技能注册
- `src/gout_agent/skills/`：运行时技能实现
- `src/gout_agent/mcp_service.py`：真实 MCP 服务
- `skills/`：给模型读取的 `SKILL.md` 技能目录
- `tests/`：单元测试与集成测试

## 启动应用

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Windows 本地也可以直接使用：

- `run_local.bat`

## 启动真实 MCP 服务

```bash
uvicorn gout_agent.mcp_service:app --host 127.0.0.1 --port 8787 --app-dir src
```

启动后，真实 MCP 的 `streamable-http` 入口为：

- `http://127.0.0.1:8787/mcp`

当前服务基于官方 `FastMCP`，已经把项目里的核心能力暴露成真实 MCP tools，并额外提供：

- `run_agent_loop`
- `preview_agent_loop`
- `gout://profile/current`
- `gout://memory/latest`

## 本地存储

- SQLite 数据库：`data/gout_management.db`
- 报告导出目录：`reports/`

## 本地模型配置

项目默认面向本地 OpenAI-compatible 接口，例如 `LM Studio`：

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:1234/v1"
$env:LOCAL_LLM_API_KEY = "lm-studio"
$env:LOCAL_LLM_MODEL = "FreedomIntelligence/HuatuoGPT-o1-7B"
$env:LOCAL_LLM_TIMEOUT_SECONDS = "60"
```

如果本地模型可用，`AI 教练` 页面会优先结合模型回答；如果模型不可用或超时，系统会自动回退到规则引擎。

## 说明

- 本项目用于健康管理与教育，不用于诊断。
- 风险逻辑以规则引擎为安全基线。
- 本地大模型主要负责解释、问答和陪伴式建议，不替代医生判断。
