# Skills 目录说明

这个目录保存项目中每个 Skill 的两部分内容：

- `SKILL.md`：定义 Skill 的职责、适用场景、推荐工具和执行步骤
- `runtime.py`：实现该 Skill 的运行时代码

当前目录结构以“一个 Skill 一个目录”为准，例如：

- `intake-skill/`
- `risk-assessment-skill/`
- `lifestyle-coach-skill/`
- `medication-followup-skill/`
- `report-explanation-skill/`
- `lab-report-skill/`

其中：

- `orchestrator-skill/` 和 `profile-skill/` 目前只保留 `SKILL.md`，用于路由和能力定义
- 业务运行时代码统一从本目录加载，不再通过 `src/gout_agent/skills/` 中的旧兼容入口转发

运行时加载关系：

`SKILL.md -> skill_registry -> orchestrator -> skills/*/runtime.py`
