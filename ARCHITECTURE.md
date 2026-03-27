# 项目架构

## 总览

当前项目是一套面向痛风与高尿酸血症长期管理场景的本地优先应用，核心由五部分组成：

- Streamlit 中文界面
- SQLite 本地数据存储
- Skill 驱动的主控编排层
- LangGraph 运行时与路由级子图
- 规则引擎、本地模型与长期记忆

## 高层架构图

```mermaid
flowchart TB
    U[用户 / 浏览器]
    S[Streamlit 界面\nui.py + streamlit_app.py]

    O[主控 Orchestrator\nskills/orchestrator.py]
    LG[LangGraph Runtime\n主图 + 子图]

    I[Intake Skill\nskills/intake.py]
    R[风险 Skill\nskills/risk_skill.py]
    L[生活方式 Skill\nskills/lifestyle.py]
    M[用药随访 Skill\nskills/medication.py]
    P[报告解读 Skill\nskills/reporting_skill.py]
    PF[档案 Skill\nprofile-skill]

    T[内部工具注册表\ntoolkit.py]
    SR[Skill Registry\nskill_registry.py + skills/*/SKILL.md]

    D[数据层\ndata.py]
    RK[风险引擎\nrisk.py]
    RP[报告引擎\nreporting.py]
    MEM[记忆层\nmemory.py]
    LM[本地模型适配\nllm.py]

    DB[(SQLite\ndata/gout_management.db)]
    FS[(报告目录\nreports/)]
    LLM[(LM Studio / HuatuoGPT-o1-7B\nOpenAI-compatible API)]

    U --> S
    S --> O
    O --> LG
    O --> SR
    O --> T

    LG --> I
    LG --> R
    LG --> L
    LG --> M
    LG --> P
    LG --> PF

    T --> D
    T --> RK
    T --> RP
    T --> MEM
    T --> LM

    D --> DB
    RP --> FS
    LM --> LLM
```

## 分层说明

### 1. 界面层

文件：
- `streamlit_app.py`
- `src/gout_agent/ui.py`

职责：
- 提供中文 Web 界面
- 管理总览、我的资料、记录、风险监测和 AI 管理助手页面
- 将用户操作交给 orchestrator

### 2. 编排层

文件：
- `src/gout_agent/skills/orchestrator.py`

职责：
- 加载上下文
- 根据 `SKILL.md` 和规则路由到对应 skill
- 基于 `allowed_tools` 约束工具调用
- 调用 LangGraph 主图或对应子图
- 在本地模型与规则回退之间组织最终回答

### 3. LangGraph 运行时

文件：
- `src/gout_agent/skills/orchestrator.py`

职责：
- 承载多步 Agent Loop
- 统一运行态与预演态
- 为不同 skill 提供专用子图

当前已经拆出的子图包括：
- `intake_graph / preview_intake_graph`
- `profile_graph / preview_profile_graph`
- `reporting_graph / preview_reporting_graph`
- `medication_graph / preview_medication_graph`
- `risk_graph / preview_risk_graph`
- `lifestyle_graph / preview_lifestyle_graph`

### 4. Skill 层

文件：
- `src/gout_agent/skills/intake.py`
- `src/gout_agent/skills/risk_skill.py`
- `src/gout_agent/skills/lifestyle.py`
- `src/gout_agent/skills/medication.py`
- `src/gout_agent/skills/reporting_skill.py`
- `skills/*/SKILL.md`

职责：
- `Intake Skill`：结构化记录
- `风险 Skill`：风险、诱因与异常解释
- `生活方式 Skill`：饮食、饮水与运动建议
- `用药随访 Skill`：药物、服药与提醒
- `报告解读 Skill`：周报、月报解释
- `档案 Skill`：长期资料与 AI 管理助手长期建议

### 5. 工具与业务层

文件：
- `src/gout_agent/toolkit.py`
- `src/gout_agent/data.py`
- `src/gout_agent/risk.py`
- `src/gout_agent/reporting.py`
- `src/gout_agent/memory.py`
- `src/gout_agent/llm.py`

职责：
- `toolkit.py`：统一内部工具注册
- `data.py`：SQLite 建表与 CRUD
- `risk.py`：风险计算、诱因识别、异常识别、趋势预测
- `reporting.py`：周报、月报与导出
- `memory.py`：长期记忆、行为画像、AI 管理助手长期建议摘要
- `llm.py`：本地模型接入与回答组织

## 当前使用的 Skill

当前项目实际使用的 Skill 包括：

- 主控 Skill
- Intake Skill
- 风险评估 Skill
- 生活方式 Skill
- 用药随访 Skill
- 报告解读 Skill
- 档案管理 Skill

## 请求流程

### 1. 日常记录提交流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as Streamlit UI
    participant O as Orchestrator
    participant T as Tool Registry
    participant DB as SQLite

    User->>UI: 提交日常记录
    UI->>O: save_daily_log(payload)
    O->>T: 记录日常健康
    T->>DB: 写入 daily_logs
    DB-->>T: 返回记录 ID
    T-->>O: 保存成功
    O-->>UI: 返回成功消息
```

### 2. 风险刷新流程

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant O as Orchestrator
    participant T as Tool Registry
    participant R as Risk Engine
    participant DB as SQLite

    UI->>O: load_context()
    O->>T: 读取资料、记录、发作、药物与提醒
    T->>DB: 读取本地数据
    DB-->>T: 返回数据
    O->>T: 计算痛风风险
    T->>R: 执行规则引擎
    R-->>T: 返回风险结果
    O-->>UI: 返回上下文与快照
```

### 3. AI 管理助手问答流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as Streamlit UI
    participant O as Orchestrator
    participant SR as Skill Registry
    participant LG as LangGraph
    participant T as Tool Registry
    participant LLM as Local LLM

    User->>UI: 输入问题
    UI->>O: answer_coach_question(question, context)
    O->>SR: 匹配 Skill
    SR-->>O: route_name / allowed_tools / execution_steps
    O->>LG: 选择主图或对应子图
    LG->>T: 调用允许范围内的工具
    T-->>LG: 返回 observation
    LG-->>O: 返回步骤轨迹和执行结果
    O->>LLM: 结合上下文组织回答
    LLM-->>O: 返回模型答案
    O-->>UI: 返回 answer
```
