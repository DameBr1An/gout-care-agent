"""技能运行时包。

这里仅保留主控编排和运行时加载器：
- orchestrator.py：主控 Agent 编排入口
- _runtime_loader.py：按 skill 目录动态加载顶层 runtime.py

各业务 Skill 的定义和运行时代码统一放在项目根目录的 skills/* 下。
"""
