# AI 痛风管理助手

这是一个面向痛风与高尿酸血症长期管理场景的本地优先应用，核心目标是帮助用户完成日常记录、风险预警、个性化建议和长期陪伴管理。

## 当前能力

- 本地 SQLite 数据存储
- 用户资料与长期画像管理
- 日常记录、发作记录、用药记录
- 可选高级化验录入
- 基于规则引擎的风险评估、诱因识别与异常提醒
- 周报、月报生成与解读
- 本地大模型问答与报告解释
- Skill 驱动的 Agent 编排
- LangGraph 多步执行与回退

## 主要目录

- `streamlit_app.py`：Streamlit 入口
- `src/gout_agent/ui.py`：中文界面
- `src/gout_agent/data.py`：SQLite 数据层
- `src/gout_agent/risk.py`：风险评估与诱因识别
- `src/gout_agent/reporting.py`：报告生成与导出
- `src/gout_agent/memory.py`：长期记忆与行为画像
- `src/gout_agent/llm.py`：本地模型适配
- `src/gout_agent/skill_registry.py`：`SKILL.md` 解析与技能注册
- `src/gout_agent/skills/`：运行时技能实现
- `skills/`：给模型读取的 `SKILL.md` 技能目录
- `tests/`：自动化测试

## 启动应用

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Windows 本地也可以直接使用：

- `run_local.bat`

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

如果本地模型可用，`AI 管理助手` 页面会优先结合模型回答；如果模型不可用或超时，系统会自动回退到规则引擎。

## 说明

- 本项目用于健康管理与教育，不用于诊断。
- 风险逻辑以规则引擎为安全基线。
- 本地大模型主要负责解释、问答和陪伴式建议，不替代医生判断。
