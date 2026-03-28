# AI Gout Management Agent

一款以**痛风健康分身**为核心的本地智能应用，采用 **harness engineering** 设计哲学构建：不把能力全部压在模型本身，而是通过统一 Agent Runtime、Skill 协议、Tool 边界、外置状态、权限与后台任务，为模型提供一个稳定、可解释、可恢复的运行环境。

## 当前能力

- 健康分身：部位热力图、近期行为、个人痛风模式
- 风险概览：当前风险、变化原因、今天怎么做
- 数据记录：日常行为、疼痛记录、服药记录
- 报告中心：周报 / 月报生成，化验报告上传与 AI 解读
- 智能问答助手：基于全局记录、健康分身和历史状态进行问答
- 本地多用户：注册、登录、密码哈希、账号注销
- 后台任务：报告生成、化验报告识别、健康分身重算
- 敏感写审计：对高风险写操作进行确认与审计落盘

## 技术架构

当前项目采用：

- `Streamlit`：前台交互
- `SQLite`：本地持久化与迁移
- `LangGraph`：执行图与主控流程
- `Skill + Tool`：任务协议与环境能力边界
- `本地大模型`：问答、报告解读、化验辅助解析

更完整的分层架构图和文件对应关系见：

- [ARCHITECTURE.md](/d:/ai-gout-management-agent/ARCHITECTURE.md)

## 主要目录

- `streamlit_app.py`
- `src/gout_agent/ui.py`
- `src/gout_agent/data.py`
- `src/gout_agent/memory.py`
- `src/gout_agent/risk.py`
- `src/gout_agent/reporting.py`
- `src/gout_agent/llm.py`
- `src/gout_agent/skill_registry.py`
- `src/gout_agent/skills/orchestrator.py`
- `skills/`
- `tests/`

## 本地启动

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Windows 下也可以直接运行：

```powershell
.\run_local.bat
```

## 说明

- 本项目用于健康管理和教育，不替代医生诊断
- 风险判断以规则引擎为基础，模型主要负责解释与交互
- 当前支持本地模型与规则回退协同运行
