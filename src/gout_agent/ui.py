from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from gout_agent.skills.orchestrator import AppOrchestrator


ALCOHOL_OPTIONS = {
    "none": "无",
    "beer": "啤酒",
    "wine": "葡萄酒",
    "spirits": "烈酒",
    "other": "其他",
}

GENDER_OPTIONS = {
    "unknown": "未知",
    "male": "男",
    "female": "女",
    "other": "其他",
}

REMINDER_OPTIONS = {
    "medication": "用药提醒",
    "hydration": "饮水提醒",
    "checkup": "复查提醒",
}

STATUS_OPTIONS = {
    "taken": "已服药",
    "missed": "漏服",
    "skipped": "跳过",
}


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

        :root {
            --bg-soft: #f6f3ec;
            --panel: rgba(255,255,255,0.82);
            --panel-strong: #fffdf8;
            --line: rgba(120, 93, 62, 0.16);
            --text-main: #2e261f;
            --text-soft: #6b5a49;
            --accent: #9d5c2f;
            --accent-soft: #efe1cf;
            --nav-bg: linear-gradient(180deg, #efe6d7 0%, #e4d5bf 100%);
            --success: #3d7b54;
            --warning: #b46a1f;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(224, 201, 173, 0.55), transparent 28%),
                radial-gradient(circle at top right, rgba(198, 217, 192, 0.45), transparent 24%),
                linear-gradient(180deg, #f8f4ed 0%, #f3eee6 100%);
            color: var(--text-main);
            font-family: "Noto Sans SC", sans-serif;
        }

        header[data-testid="stHeader"] {
            background: rgba(248, 244, 237, 0.92);
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(10px);
        }

        [data-testid="stToolbar"] {
            right: 1rem;
            top: 0.5rem;
        }

        section[data-testid="stSidebar"] {
            background: var(--nav-bg);
            border-right: 1px solid rgba(120, 93, 62, 0.14);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }

        section[data-testid="stSidebar"] .stRadio > div {
            background: rgba(255,255,255,0.42);
            border-radius: 18px;
            padding: 0.35rem;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 32px rgba(95, 73, 46, 0.06);
        }

        [data-testid="stVerticalBlock"] div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.65);
            color: var(--text-soft);
            padding: 0.5rem 1rem;
        }

        [data-testid="stVerticalBlock"] div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(180deg, #b86f3d 0%, #9d5c2f 100%);
            color: white;
            border-color: rgba(157, 92, 47, 0.45);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .app-shell-header {
            background: linear-gradient(135deg, rgba(255,253,248,0.94) 0%, rgba(248,241,229,0.92) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 42px rgba(93, 70, 45, 0.08);
        }

        .app-shell-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.15rem;
        }

        .app-shell-subtitle {
            font-size: 0.94rem;
            color: var(--text-soft);
        }

        .app-shell-chip {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem 0.4rem 1.1rem;
            box-shadow: 0 16px 34px rgba(98, 76, 52, 0.06);
            margin-bottom: 1rem;
        }

        div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"], div[data-testid="stExpander"] {
            background: var(--panel-strong);
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_shell_header(page: str) -> None:
    subtitles = {
        "总览": "聚焦今天的风险、诱因和管理重点。",
        "我的资料": "统一查看基础资料和长期健康画像。",
        "记录": "用更轻的方式记录日常、发作和用药情况。",
        "风险监测": "追踪风险变化，理解最近为什么波动。",
        "AI 管理助手": "把问答、建议和报告解读放到同一个入口。",
        "工具与服务": "查看当前系统能力、工具和服务接口。",
    }
    st.markdown(
        f"""
        <div class="app-shell-header">
            <div class="app-shell-chip">{page}</div>
            <div class="app-shell-title">痛风日常管理</div>
            <div class="app-shell-subtitle">{subtitles.get(page, "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app(project_root: Path) -> None:
    orchestrator = AppOrchestrator(project_root)
    context = orchestrator.load_context()
    orchestrator.sync_daily_snapshot(context)
    context = orchestrator.load_context()
    snapshot = orchestrator.get_ui_snapshot(context)
    _apply_theme()

    with st.sidebar:
        st.header("导航")
        page = st.radio(
            "前往页面",
            ["总览", "我的资料", "记录", "风险监测", "AI 管理助手"],
        )
        st.divider()
        st.write(f"**用户**：{context.profile.get('name') or 'Demo User'}")
        st.write(f"**目标尿酸**：{context.profile.get('target_uric_acid') or 360} umol/L")
        st.write(f"**当前发作风险**：{snapshot['attack_risk_label']}")
        st.caption("本地模型")
        st.write(f"模型：`{snapshot['llm_status']['model']}`")
        st.write(f"接口：`{snapshot['llm_status']['base_url']}`")

    _render_shell_header(page)

    if page == "总览":
        _render_dashboard(orchestrator, context, snapshot)
    elif page == "我的资料":
        _render_profile_hub(orchestrator, context)
    elif page == "记录":
        _render_record_hub(orchestrator, context)
    elif page == "风险监测":
        _render_risk_monitor(orchestrator, context, snapshot)
    elif page == "AI 管理助手":
        _render_ai_coach(orchestrator, context)


def _render_profile_hub(orchestrator: AppOrchestrator, context) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("我的资料")
    st.caption("这里统一管理基础资料和长期画像。")
    tab1, tab2 = st.tabs(["基础资料", "长期画像"])
    with tab1:
        _render_profile_management(orchestrator, context)
    with tab2:
        _render_memory_portrait(context, nested=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_record_hub(orchestrator: AppOrchestrator, context) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("记录")
    st.caption("这里统一记录日常情况、发作事件和用药相关信息。")
    tab1, tab2, tab3 = st.tabs(["日常记录", "发作记录", "用药记录"])
    with tab1:
        _render_daily_log(orchestrator, context, nested=True)
    with tab2:
        _render_attack_tracker(orchestrator, context, nested=True)
    with tab3:
        _render_medication_management(orchestrator, context, nested=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_dashboard(orchestrator: AppOrchestrator, context, snapshot: dict) -> None:
    latest_lab = context.labs.iloc[-1].to_dict() if not context.labs.empty else {}
    latest_uric = latest_lab.get("uric_acid") if latest_lab else None
    risk_snapshots = orchestrator.registry.call("获取风险快照", 90)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("当前尿酸", f"{latest_uric} umol/L" if latest_uric else "暂无化验")
    k2.metric("发作风险", snapshot["attack_risk_label"])
    k3.metric("风险评分", snapshot["overall_risk_score"])
    k4.metric(
        "近 7 天服药完成率",
        f"{snapshot['medication_completion_rate']:.0f}%" if snapshot["medication_completion_rate"] is not None else "暂无记录",
    )

    col1, col2 = st.columns([1.3, 0.7], gap="large")
    with col1:
        st.subheader("风险趋势")
        if risk_snapshots.empty:
            st.info("添加记录后，这里会显示风险快照趋势。")
        else:
            frame = risk_snapshots.copy()
            frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
            fig = px.line(frame, x="snapshot_date", y="overall_risk_score", markers=True, title="整体风险评分趋势")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("今日建议")
        st.markdown(f"**饮水**：{snapshot['hydration_advice']}")
        st.markdown(f"**饮食**：{snapshot['diet_advice']}")
        st.markdown(f"**运动**：{snapshot['exercise_advice']}")
        st.markdown(f"**今日目标**：{snapshot['behavior_goal']}")

    with col2:
        st.subheader("提醒与预警")
        if not latest_uric:
            st.info("当前没有化验数据，系统会先根据症状、饮水、饮食、饮酒和服药情况进行日常风险评估。")
        if snapshot["abnormal_items"]:
            for item in snapshot["abnormal_items"]:
                st.warning(item)
        else:
            st.success("最近一次数据中未发现明显异常提醒。")

        st.subheader("本周完成情况")
        weekly_water = _mean_numeric(context.logs.tail(7), "water_ml")
        weekly_steps = _mean_numeric(context.logs.tail(7), "steps")
        st.write(f"平均饮水：{weekly_water if weekly_water is not None else '-'} mL/天")
        st.write(f"平均步数：{weekly_steps if weekly_steps is not None else '-'} 步/天")
        st.write(f"启用中的提醒：{snapshot['active_reminder_count']}")
        st.write(f"当前药物数：{snapshot['active_medication_count']}")

    st.subheader("近期诱因")
    if snapshot["trigger_summary"]:
        trigger_frame = pd.DataFrame(snapshot["trigger_summary"])
        st.dataframe(
            trigger_frame[["label", "count"]].rename(columns={"label": "诱因", "count": "次数"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("近期还没有识别到明显诱因。")


def _render_daily_log(orchestrator: AppOrchestrator, context, nested: bool = False) -> None:
    if not nested:
        st.subheader("每日记录")
        st.caption("默认只展示日常生活记录。化验结果录入已降级为可选高级功能。")
    tab1, tab2 = st.tabs(["日常健康", "个人档案"])

    with tab1:
        with st.form("daily_health_form"):
            log_date = st.date_input("日期", value=date.today())
            weight_kg = st.number_input("体重 (kg)", min_value=0.0, max_value=300.0, value=float(context.profile.get("baseline_weight_kg") or 70.0), step=0.5)
            water_ml = st.number_input("饮水量 (mL)", min_value=0, max_value=6000, value=1800, step=100)
            steps = st.number_input("步数", min_value=0, max_value=50000, value=6000, step=500)
            exercise_minutes = st.number_input("运动时长 (分钟)", min_value=0, max_value=300, value=20, step=5)
            sleep_hours = st.number_input("睡眠时长 (小时)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            alcohol_key = st.selectbox("饮酒情况", list(ALCOHOL_OPTIONS.keys()), format_func=lambda x: ALCOHOL_OPTIONS.get(x, x))
            diet_notes = st.text_input("饮食备注", placeholder="例如：海鲜、火锅、烧烤")
            symptom_notes = st.text_input("症状备注", placeholder="例如：左脚大脚趾轻微疼痛")
            pain_score = st.slider("疼痛评分", min_value=0, max_value=10, value=0)
            joint_pain_flag = st.checkbox("今天有关节痛", value=False)
            medication_taken_flag = st.checkbox("今天已服药", value=True)
            free_text = st.text_area("还有什么想补充的？")
            submitted = st.form_submit_button("保存日常记录")
        if submitted:
            orchestrator.save_daily_log(
                {
                    "log_date": str(log_date),
                    "weight_kg": weight_kg,
                    "water_ml": water_ml,
                    "steps": steps,
                    "exercise_minutes": exercise_minutes,
                    "sleep_hours": sleep_hours,
                    "alcohol_intake": alcohol_key,
                    "diet_notes": diet_notes,
                    "symptom_notes": symptom_notes,
                    "pain_score": pain_score,
                    "joint_pain_flag": joint_pain_flag,
                    "medication_taken_flag": medication_taken_flag,
                    "free_text": free_text,
                }
            )
            st.success("日常健康记录已保存，请刷新或切换页面查看最新结果。")

        with st.expander("高级功能：补充化验结果", expanded=False):
            st.caption("化验结果不是日常必填项；如近期有抽血或复查结果，可在这里补充。")
            with st.form("lab_form"):
                test_date = st.date_input("化验日期", value=date.today(), key="lab_date")
                uric_acid = st.number_input("尿酸 (umol/L)", min_value=0.0, max_value=1200.0, value=420.0, step=1.0)
                creatinine = st.number_input("肌酐 (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
                egfr = st.number_input("eGFR", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
                crp = st.number_input("CRP", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
                esr = st.number_input("ESR", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
                ast = st.number_input("AST", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
                alt = st.number_input("ALT", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
                notes = st.text_area("化验备注")
                submitted = st.form_submit_button("保存化验结果")
            if submitted:
                orchestrator.save_lab_result(
                    {
                        "test_date": str(test_date),
                        "uric_acid": uric_acid,
                        "creatinine": creatinine,
                        "egfr": egfr,
                        "crp": crp,
                        "esr": esr,
                        "ast": ast,
                        "alt": alt,
                        "notes": notes,
                    }
                )
                st.success("化验结果已保存，请刷新或切换页面查看最新结果。")

    with tab2:
        _render_profile_form(orchestrator, context, form_key="profile_form")


def _render_profile_management(orchestrator: AppOrchestrator, context) -> None:
    st.subheader("基础资料")
    profile = orchestrator.get_profile()

    col1, col2 = st.columns([0.9, 1.1], gap="large")
    with col1:
        st.write("**当前档案摘要**")
        st.write(orchestrator._summarize_profile(profile))

        tags: list[str] = []
        if profile.get("has_gout_diagnosis"):
            tags.append("已确诊痛风")
        if profile.get("has_hyperuricemia"):
            tags.append("高尿酸血症")
        if profile.get("has_ckd"):
            tags.append("慢性肾病")
        if profile.get("has_hypertension"):
            tags.append("高血压")
        if profile.get("has_diabetes"):
            tags.append("糖尿病")

        st.write("**长期健康背景**")
        if tags:
            st.write("、".join(tags))
        else:
            st.caption("当前还没有明确记录基础病信息。")
        if profile.get("allergy_notes"):
            st.write(f"过敏备注：{profile.get('allergy_notes')}")

    with col2:
        _render_profile_form(orchestrator, context, form_key="profile_management_form")


def _render_risk_monitor(orchestrator: AppOrchestrator, context, snapshot: dict) -> None:
    st.subheader("风险监测")
    risk_snapshots = orchestrator.registry.call("获取风险快照", 90)
    has_lab_data = not context.labs.empty

    col1, col2 = st.columns(2)
    col1.metric("当前尿酸风险", snapshot["uric_acid_risk_label"])
    col2.metric("当前发作风险", snapshot["attack_risk_label"])

    if not has_lab_data:
        st.info("当前风险结果主要基于症状、饮水、饮酒、饮食诱因、服药和发作记录生成，后续如补充化验结果会进一步提高评估完整度。")

    st.write(snapshot["explanation"])

    if not risk_snapshots.empty:
        frame = risk_snapshots.copy()
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        fig = px.line(frame, x="snapshot_date", y="overall_risk_score", markers=True, title="风险评分趋势")
        st.plotly_chart(fig, use_container_width=True)

    st.write("**异常提醒**")
    if snapshot["abnormal_items"]:
        for item in snapshot["abnormal_items"]:
            st.warning(item)
    else:
        st.success("暂无明显异常提醒。")

    st.write("**诱因回顾**")
    if snapshot["trigger_summary"]:
        trigger_frame = pd.DataFrame(snapshot["trigger_summary"])
        st.dataframe(
            trigger_frame[["label", "count"]].rename(columns={"label": "诱因", "count": "次数"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("最近暂无明显诱因。")


def _render_attack_tracker(orchestrator: AppOrchestrator, context, nested: bool = False) -> None:
    if not nested:
        st.subheader("发作追踪")
    with st.form("attack_form"):
        attack_date = st.date_input("发作日期", value=date.today())
        joint_site = st.text_input("发作关节", value="左脚大脚趾")
        pain_score = st.slider("疼痛评分", min_value=0, max_value=10, value=6)
        swelling_flag = st.checkbox("是否肿胀", value=True)
        redness_flag = st.checkbox("是否发红", value=False)
        duration_hours = st.number_input("持续时间 (小时)", min_value=0.0, max_value=240.0, value=24.0, step=1.0)
        suspected_trigger = st.text_input("疑似诱因", placeholder="例如：啤酒、海鲜、饮水不足")
        resolved_flag = st.checkbox("是否已缓解", value=False)
        notes = st.text_area("备注")
        submitted = st.form_submit_button("保存发作记录")
    if submitted:
        orchestrator.save_attack(
            {
                "attack_date": str(attack_date),
                "joint_site": joint_site,
                "pain_score": pain_score,
                "swelling_flag": swelling_flag,
                "redness_flag": redness_flag,
                "duration_hours": duration_hours,
                "suspected_trigger": suspected_trigger,
                "resolved_flag": resolved_flag,
                "notes": notes,
            }
        )
        st.success("发作记录已保存，请刷新或切换页面查看最新结果。")

    st.subheader("最近发作记录")
    if context.attacks.empty:
        st.info("还没有发作记录。")
    else:
        display = context.attacks[["attack_date", "joint_site", "pain_score", "suspected_trigger", "resolved_flag"]].copy()
        display["resolved_flag"] = display["resolved_flag"].map(lambda x: "是" if bool(x) else "否")
        display.columns = ["日期", "部位", "疼痛评分", "疑似诱因", "是否缓解"]
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_medication_management(orchestrator: AppOrchestrator, context, nested: bool = False) -> None:
    if not nested:
        st.subheader("用药管理")
        st.caption("提醒作为附属能力挂在用药流程下，不再单独提供独立管理页面。")
    left, right = st.columns(2, gap="large")

    with left:
        with st.form("medication_form"):
            medication_name = st.text_input("药物名称", value="Allopurinol")
            dose = st.text_input("剂量", value="100 mg")
            frequency = st.text_input("频率", value="每日一次")
            start_date = st.date_input("开始日期", value=date.today())
            end_date = st.date_input("结束日期", value=date.today() + timedelta(days=30))
            purpose = st.text_input("用途", value="降尿酸")
            active_flag = st.checkbox("启用中", value=True)
            med_submitted = st.form_submit_button("添加药物")
        if med_submitted:
            orchestrator.add_medication(
                {
                    "medication_name": medication_name,
                    "dose": dose,
                    "frequency": frequency,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "purpose": purpose,
                    "active_flag": active_flag,
                }
            )
            st.success("药物已添加，请刷新或切换页面查看最新结果。")

        with st.expander("可选：附加提醒", expanded=False):
            st.caption("如需把用药、饮水或复查提醒和当前管理计划一起保存，可在这里补充。")
            with st.form("reminder_form"):
                reminder_keys = list(REMINDER_OPTIONS.keys())
                reminder_type = st.selectbox("提醒类型", reminder_keys, format_func=lambda x: REMINDER_OPTIONS.get(x, x))
                title = st.text_input("标题", value="晚间服药提醒")
                schedule_rule = st.text_input("提醒规则", value="每天 20:00")
                next_trigger_at = st.text_input("下次提醒时间", value=datetime.now().replace(microsecond=0).isoformat(sep=" "))
                reminder_submitted = st.form_submit_button("保存提醒")
            if reminder_submitted:
                orchestrator.create_reminder(reminder_type, title, schedule_rule, next_trigger_at)
                st.success("提醒已保存，请刷新或切换页面查看最新结果。")

    with right:
        st.write("**当前药物**")
        if context.medications.empty:
            st.info("还没有添加药物。")
        else:
            med_view = context.medications[["id", "medication_name", "dose", "frequency", "purpose", "active_flag"]].copy()
            med_view["active_flag"] = med_view["active_flag"].map(lambda x: "是" if bool(x) else "否")
            show_view = med_view.drop(columns=["id"]).copy()
            show_view.columns = ["药物名称", "剂量", "频率", "用途", "启用中"]
            st.dataframe(show_view, use_container_width=True, hide_index=True)

            medication_choices = {f"{row['medication_name']} ({row['dose']})": int(row['id']) for _, row in med_view.iterrows()}
            status_keys = list(STATUS_OPTIONS.keys())
            selected_label = st.selectbox("记录服药情况", list(medication_choices.keys())) if medication_choices else None
            status = st.selectbox("状态", status_keys, format_func=lambda x: STATUS_OPTIONS.get(x, x)) if medication_choices else None
            if medication_choices and st.button("保存服药记录"):
                taken_time = datetime.now().replace(microsecond=0).isoformat(sep=" ") if status == "taken" else None
                orchestrator.log_medication_taken(medication_choices[selected_label], status, taken_time)
                st.success("服药记录已保存，请刷新或切换页面查看最新结果。")

        st.write("**启用中的提醒**")
        if context.reminders.empty:
            st.info("还没有启用中的提醒。")
        else:
            reminder_view = context.reminders[["reminder_type", "title", "schedule_rule", "next_trigger_at"]].copy()
            reminder_view["reminder_type"] = reminder_view["reminder_type"].map(lambda x: REMINDER_OPTIONS.get(x, x))
            reminder_view.columns = ["提醒类型", "标题", "提醒规则", "下次提醒时间"]
            st.dataframe(reminder_view, use_container_width=True, hide_index=True)

        adherence = orchestrator.registry.call("获取服药依从性", 30)
        if not adherence.empty:
            st.write("**近 30 天服药记录**")
            adherence_view = adherence[["medication_name", "status", "taken_time", "created_at"]].copy()
            adherence_view["status"] = adherence_view["status"].map(lambda x: STATUS_OPTIONS.get(x, x))
            adherence_view.columns = ["药物名称", "状态", "服药时间", "记录时间"]
            st.dataframe(adherence_view, use_container_width=True, hide_index=True)


def _render_ai_coach(orchestrator: AppOrchestrator, context) -> None:
    st.subheader("AI 管理助手")
    st.caption("这里统一承接 AI 问答、AI 管理建议和报告解读，结合你的记录、风险结果和长期记忆给出建议。")

    question = st.text_input("请输入你想咨询的问题", placeholder="例如：最近为什么风险升高了？今晚饮食要注意什么？")
    if question:
        result = orchestrator.answer_coach_question(question, context)
        if result["source"] == "local_llm":
            st.success(f"本次回答已结合本地模型生成：{result['model']}")
        else:
            st.warning(result["error"] or "本地模型暂时不可用，系统已自动切换为规则引擎回答。")
        st.caption(f"当前处理技能：{result['skill']}")
        st.caption(f"当前可调用工具：{'、'.join(result['route_meta'].get('allowed_tools', [])) or '无'}")
        st.markdown("**AI 管理助手建议**")
        st.write(result["answer"])

    quick_cols = st.columns(3)
    quick_questions = ["我今天该怎么做？", "为什么风险变了？", "今晚该避开什么？"]
    for col, quick_question in zip(quick_cols, quick_questions):
        if col.button(quick_question):
            result = orchestrator.answer_coach_question(quick_question, context)
            st.markdown("**AI 管理助手建议**")
            st.write(result["answer"])

    st.divider()
    st.markdown("### 健康报告")
    st.caption("报告统一采用“结论摘要 - 关键发现 - 下一步建议”的表达方式，方便自查和后续复盘。")
    report_type = st.radio("报告类型", ["weekly", "monthly"], horizontal=True, format_func=lambda x: "周报" if x == "weekly" else "月报")
    report_result = orchestrator.explain_report(report_type, context)
    _render_report_preview(report_result["report"])

    if st.button("生成 AI 解读"):
        if report_result["source"] == "local_llm":
            st.success("报告解读已结合本地模型生成。")
        else:
            st.warning(report_result["error"] or "本地模型暂时不可用，系统已自动切换为规则引擎解读。")
        st.markdown("**AI 管理助手解读**")
        st.write(report_result["explanation"])

    export_format = st.selectbox("导出格式", ["json", "html"], format_func=lambda x: x.upper())
    if st.button("导出报告"):
        path = orchestrator.export_report(report_type, export_format, context)
        st.success(f"报告已导出：{path}")

    reports = orchestrator.registry.call("获取报告历史")
    if not reports.empty:
        st.write("**历史报告**")
        report_view = reports[["report_type", "period_start", "period_end", "created_at"]].copy()
        report_view["report_type"] = report_view["report_type"].map(lambda x: "周报" if x == "weekly" else "月报")
        report_view.columns = ["报告类型", "开始日期", "结束日期", "生成时间"]
        st.dataframe(report_view, use_container_width=True, hide_index=True)


def _render_agent_loop(agent_loop: dict) -> None:
    if not agent_loop:
        return
    with st.expander("查看 Agent Loop 过程", expanded=False):
        st.write(f"意图：{agent_loop.get('intent') or 'unknown'}")
        planned = agent_loop.get("planned_tools") or []
        completed = agent_loop.get("completed_tools") or []
        st.write(f"计划工具：{'、'.join(planned) or '无'}")
        st.write(f"已执行工具：{'、'.join(completed) or '无'}")
        steps = agent_loop.get("steps") or []
        if steps:
            step_frame = pd.DataFrame(
                [
                    {
                        "步骤": step.get("index"),
                        "动作": step.get("action"),
                        "工具": step.get("tool_name") or "-",
                        "状态": step.get("status"),
                        "置信度": (step.get("decision") or {}).get("confidence"),
                        "拒绝理由": (step.get("decision") or {}).get("refusal_reason"),
                        "思路": step.get("thought"),
                    }
                    for step in steps
                ]
            )
            st.dataframe(step_frame, use_container_width=True, hide_index=True)
        observations = agent_loop.get("observations") or {}
        if observations:
            st.write("**观测结果**")
            st.json(observations)


def _render_report_preview(report: dict) -> None:
    if not report:
        st.info("当前还没有可展示的报告内容。")
        return

    st.markdown("**报告摘要**")
    st.write(report.get("executive_summary") or report.get("summary") or "暂无摘要。")

    c1, c2, c3 = st.columns(3)
    latest_risk = report.get("latest_risk") or {}
    c1.metric("发作风险", latest_risk.get("attack_risk_level_cn") or latest_risk.get("attack_risk_level") or "未知")
    c2.metric("整体风险评分", latest_risk.get("overall_risk_score") or "-")
    c3.metric("记录条数", report.get("entries") or 0)

    st.markdown("**关键发现**")
    key_findings = report.get("key_findings") or []
    if key_findings:
        for item in key_findings:
            st.write("• " + str(item))
    else:
        st.caption("本期暂时没有足够记录形成关键发现。")

    st.markdown("**下一步建议**")
    action_plan = report.get("action_plan") or []
    if action_plan:
        for item in action_plan:
            st.write("• " + str(item))
    else:
        st.caption("本期暂时没有形成明确行动建议。")

    st.markdown("**就医提醒**")
    st.info(report.get("medical_notice") or "如症状持续或加重，请及时线下就医。")

    with st.expander("查看原始报告数据", expanded=False):
        st.json(report)


def _render_memory_portrait(context, nested: bool = False) -> None:
    if not nested:
        st.subheader("记忆与画像")
        st.caption("这里展示 Agent 当前沉淀下来的长期记忆、行为画像和最近会话记忆。")

    memory_payload = context.long_term_memory or {}
    preferences = memory_payload.get("user_preferences") or {}
    ai_advice_summary = memory_payload.get("ai_advice_summary") or memory_payload.get("doctor_advice_summary") or {}
    attack_patterns = memory_payload.get("attack_patterns") or {}
    portraits = memory_payload.get("behavior_portraits") or {}
    twin_profile = memory_payload.get("gout_management_twin_profile") or {}

    st.markdown("**个人痛风管理分身**")
    st.write(twin_profile.get("summary") or "当前还没有形成稳定的个人痛风管理分身，请继续补充记录。")

    twin_left, twin_right = st.columns(2, gap="large")
    with twin_left:
        st.markdown("**风险诱因层**")
        top_triggers = twin_profile.get("top_triggers") or []
        if top_triggers:
            st.dataframe(pd.DataFrame(top_triggers), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无足够数据形成个人诱因排序。")

        st.markdown("**发作前模式**")
        trigger_patterns = twin_profile.get("trigger_patterns") or []
        if trigger_patterns:
            st.dataframe(pd.DataFrame(trigger_patterns), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无明确的发作前组合模式。")

        st.markdown("**高风险时间窗口**")
        risk_windows = twin_profile.get("risk_windows") or []
        if risk_windows:
            st.dataframe(pd.DataFrame(risk_windows), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无足够数据识别高风险窗口。")

    with twin_right:
        st.markdown("**行为模式层**")
        behavior_patterns = twin_profile.get("behavior_patterns") or {}
        if behavior_patterns:
            st.json(behavior_patterns)
        else:
            st.caption("暂无行为模式摘要。")

        st.markdown("**管理稳定度层**")
        management_stability = twin_profile.get("management_stability") or {}
        if management_stability:
            col1, col2 = st.columns(2)
            col1.metric("稳定度评分", management_stability.get("stability_score") or "-")
            col2.metric("稳定度等级", management_stability.get("stability_level") or "未知")
            st.write(management_stability.get("summary") or "暂无稳定度摘要。")
        else:
            st.caption("暂无稳定度分析。")

        st.markdown("**近期主要短板**")
        shortcomings = twin_profile.get("current_shortcomings") or []
        if shortcomings:
            for item in shortcomings:
                st.write("• " + str(item))
        else:
            st.caption("暂无明显短板。")

    st.divider()

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        st.markdown("**用户长期偏好**")
        st.write(preferences.get("summary") or "暂无足够记录总结长期偏好。")
        preferred_foods = preferences.get("preferred_foods") or []
        if preferred_foods:
            st.write(f"常见饮食关键词：{'、'.join(preferred_foods[:5])}")
        if preferences.get("preferred_alcohol"):
            st.write(f"常见饮酒类型：{preferences['preferred_alcohol']}")

        st.markdown("**AI 管理助手长期建议**")
        st.write(ai_advice_summary.get("summary") or "暂无明确 AI 管理助手长期建议。")
        if ai_advice_summary.get("source_text"):
            with st.expander("查看原始 AI 管理助手建议", expanded=False):
                st.write(ai_advice_summary["source_text"])

    with top_right:
        st.markdown("**历史发作模式**")
        st.write(attack_patterns.get("summary") or "最近没有记录到明确的痛风发作模式。")
        pattern_items = {
            "近 180 天发作次数": attack_patterns.get("attack_count_180d"),
            "常见部位": attack_patterns.get("common_joint_site"),
            "常见诱因": attack_patterns.get("common_trigger"),
            "平均疼痛评分": attack_patterns.get("average_pain_score"),
            "平均发作间隔": attack_patterns.get("average_interval_days"),
        }
        st.json(pattern_items)

    st.markdown("**最近 7/30/90 天行为画像**")
    portrait_tabs = st.tabs(["近 7 天", "近 30 天", "近 90 天"])
    for tab, key in zip(portrait_tabs, ["7d", "30d", "90d"]):
        portrait = portraits.get(key) or {}
        with tab:
            if not portrait:
                st.info("暂无画像数据。")
                continue
            st.write(portrait.get("summary") or "暂无画像摘要。")
            portrait_view = {
                "记录天数": portrait.get("days_with_logs"),
                "平均饮水 mL/天": portrait.get("average_water_ml"),
                "平均步数": portrait.get("average_steps"),
                "平均运动分钟": portrait.get("average_exercise_minutes"),
                "平均睡眠小时": portrait.get("average_sleep_hours"),
                "饮酒天数": portrait.get("alcohol_days"),
                "疼痛天数": portrait.get("pain_days"),
                "服药完成率 %": portrait.get("medication_taken_rate"),
                "最近尿酸": portrait.get("latest_uric_acid"),
                "发作次数": portrait.get("attack_count"),
            }
            st.json(portrait_view)

    st.markdown("**最近会话记忆**")
    session_memories = context.session_memories or []
    if session_memories:
        session_frame = pd.DataFrame(
            [
                {
                    "时间": item.get("created_at"),
                    "角色": item.get("role"),
                    "内容": item.get("content"),
                    "元数据": item.get("metadata"),
                }
                for item in session_memories
            ]
        )
        st.dataframe(session_frame, use_container_width=True, hide_index=True)
    else:
        st.info("当前还没有沉淀会话记忆。")


def _render_tools_and_skills(orchestrator: AppOrchestrator) -> None:
    st.subheader("工具与服务")
    st.caption("这里汇总当前系统可用的内部工具、真实 MCP 服务能力和技能注册信息。")
    st.markdown("**真实 MCP 工具**")
    tools = pd.DataFrame(orchestrator.describe_tools())
    st.dataframe(tools.rename(columns={"name": "工具", "description": "说明"}), use_container_width=True, hide_index=True)

    st.markdown("**技能注册表**")
    skills = orchestrator.describe_skills()
    if skills:
        summary = pd.DataFrame(
            [
                {
                    "技能": item["name"],
                    "说明": item["description"],
                    "模块": item["module"],
                    "工具数": len(item.get("recommended_tools", [])),
                    "执行步骤数": len(item.get("execution_steps", [])),
                }
                for item in skills
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        with st.expander("查看技能细节", expanded=False):
            for item in skills:
                st.markdown(f"### {item['name']}")
                st.write(item.get("description") or "暂无说明")
                st.caption(f"目录：{item['directory']} | 模块：{item.get('module') or '未绑定'}")
                st.write(f"推荐工具：{'、'.join(item.get('recommended_tools', [])) or '无'}")
                st.write(f"执行步骤：{'；'.join(item.get('execution_steps', [])) or '无'}")
    else:
        st.info("当前没有发现可用的 SKILL.md。")



def _render_profile_form(orchestrator: AppOrchestrator, context, form_key: str) -> None:
    gender_keys = list(GENDER_OPTIONS.keys())
    current_gender = context.profile.get("gender") or "unknown"
    gender_index = gender_keys.index(current_gender) if current_gender in gender_keys else 0
    with st.form(form_key):
        name = st.text_input("姓名", value=context.profile.get("name") or "Demo User")
        gender = st.selectbox("性别", gender_keys, index=gender_index, format_func=lambda x: GENDER_OPTIONS.get(x, x))
        birth_date = st.text_input("出生日期", value=context.profile.get("birth_date") or "")
        height_cm = st.number_input("身高 (cm)", min_value=0.0, max_value=250.0, value=float(context.profile.get("height_cm") or 170.0), step=1.0)
        baseline_weight_kg = st.number_input("基础体重 (kg)", min_value=0.0, max_value=300.0, value=float(context.profile.get("baseline_weight_kg") or 70.0), step=0.5)
        target_uric_acid = st.number_input("目标尿酸", min_value=0.0, max_value=1000.0, value=float(context.profile.get("target_uric_acid") or 360.0), step=10.0)
        has_gout_diagnosis = st.checkbox("已确诊痛风", value=bool(context.profile.get("has_gout_diagnosis")))
        has_hyperuricemia = st.checkbox("高尿酸血症", value=bool(context.profile.get("has_hyperuricemia")))
        has_ckd = st.checkbox("慢性肾病 (CKD)", value=bool(context.profile.get("has_ckd")))
        has_hypertension = st.checkbox("高血压", value=bool(context.profile.get("has_hypertension")))
        has_diabetes = st.checkbox("糖尿病", value=bool(context.profile.get("has_diabetes")))
        allergy_notes = st.text_area("过敏备注", value=context.profile.get("allergy_notes") or "")
        doctor_advice = st.text_area("AI 管理助手长期建议", value=context.profile.get("doctor_advice") or "", help="填写 AI 管理助手给出的长期管理建议或你希望长期参考的注意事项。")
        submitted = st.form_submit_button("保存档案")
    if submitted:
        orchestrator.update_profile(
            {
                "name": name,
                "gender": gender,
                "birth_date": birth_date or None,
                "height_cm": height_cm,
                "baseline_weight_kg": baseline_weight_kg,
                "target_uric_acid": target_uric_acid,
                "has_gout_diagnosis": has_gout_diagnosis,
                "has_hyperuricemia": has_hyperuricemia,
                "has_ckd": has_ckd,
                "has_hypertension": has_hypertension,
                "has_diabetes": has_diabetes,
                "allergy_notes": allergy_notes,
                "doctor_advice": doctor_advice,
            }
        )
        st.success("用户档案已更新，请刷新或切换页面查看最新结果。")


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.mean()), 1)
