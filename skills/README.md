# Skill 定义目录

这个目录保存了项目中 6 个轻量 Skill 的正式 `SKILL.md` 定义。

每个 Skill 都采用同样的结构：

- 一个目录
- 一个 `SKILL.md`
- frontmatter 元数据
- 使用时机
- 执行步骤
- 输出要求

当前 Skill 包括：

- `orchestrator-skill`
- `intake-skill`
- `risk-assessment-skill`
- `lifestyle-coach-skill`
- `medication-followup-skill`
- `report-explanation-skill`

这些 `SKILL.md` 是给大模型 / Agent 运行时读取的行为说明书。
当前真正可执行的 Python Skill 模块位于：

- `src/gout_agent/skills/`