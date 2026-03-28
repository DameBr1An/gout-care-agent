from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from gout_agent import data
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

STATUS_OPTIONS = {
    "taken": "已服药",
    "missed": "漏服",
    "skipped": "跳过",
}

BODY_SITE_OPTIONS = [
    "左脚大脚趾",
    "右脚大脚趾",
    "左脚踝",
    "右脚踝",
    "左膝",
    "右膝",
    "左足背",
    "右足背",
    "其他",
]

BODY_HEATMAP_POINTS = [
    ("左膝", 150, 180),
    ("右膝", 210, 180),
    ("左脚踝", 145, 255),
    ("右脚踝", 215, 255),
    ("左足背", 140, 305),
    ("右足背", 220, 305),
    ("左脚大脚趾", 130, 340),
    ("右脚大脚趾", 230, 340),
]


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

        .summary-card {
            background: linear-gradient(135deg, rgba(255,253,248,0.98) 0%, rgba(245,236,221,0.92) 100%);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 12px 28px rgba(98, 76, 52, 0.06);
        }

        .summary-card-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.35rem;
        }

        .summary-card-body {
            color: var(--text-main);
            line-height: 1.65;
            font-size: 0.96rem;
        }

        .bullet-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.75rem;
        }

        .bullet-card-title {
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.35rem;
        }

        .bullet-chip {
            display: inline-block;
            background: rgba(239, 225, 207, 0.9);
            color: var(--accent);
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .assistant-dock {
            position: sticky;
            bottom: 1rem;
            z-index: 30;
            display: flex;
            justify-content: flex-end;
            margin-top: 1rem;
        }

        .assistant-dock-card {
            width: min(100%, 360px);
            background: linear-gradient(135deg, rgba(255,253,248,0.96) 0%, rgba(248,241,229,0.96) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 0.9rem 1rem;
            box-shadow: 0 18px 42px rgba(93, 70, 45, 0.12);
        }

        .assistant-dock-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.15rem;
        }

        .assistant-avatar {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, #b86f3d 0%, #9d5c2f 100%);
            color: white;
            font-size: 1rem;
            box-shadow: 0 10px 18px rgba(157, 92, 47, 0.22);
        }

        .assistant-bubble {
            position: relative;
            width: 16px;
            height: 12px;
            border-radius: 8px;
            background: rgba(255,255,255,0.95);
            display: inline-block;
        }

        .assistant-bubble::after {
            content: "";
            position: absolute;
            left: 2px;
            bottom: -4px;
            width: 7px;
            height: 7px;
            background: rgba(255,255,255,0.95);
            clip-path: polygon(0 0, 100% 0, 0 100%);
        }

        .assistant-bubble::before {
            content: "";
            position: absolute;
            left: 3px;
            top: 4px;
            width: 10px;
            height: 2px;
            border-radius: 999px;
            background: rgba(157, 92, 47, 0.75);
            box-shadow: 0 4px 0 rgba(157, 92, 47, 0.75);
        }

        .assistant-dock-subtitle {
            font-size: 0.86rem;
            color: var(--text-soft);
            margin-bottom: 0.7rem;
        }

        .account-entry {
            background: rgba(255,255,255,0.58);
            border: 1px solid rgba(120, 93, 62, 0.16);
            border-radius: 18px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.8rem;
        }

        .account-entry-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.15rem;
        }

        .account-entry-subtitle {
            color: var(--text-soft);
            font-size: 0.82rem;
        }

        .account-avatar {
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: linear-gradient(180deg, #b86f3d 0%, #9d5c2f 100%);
        }

        .account-person {
            position: relative;
            width: 14px;
            height: 16px;
            display: inline-block;
        }

        .account-person::before {
            content: "";
            position: absolute;
            left: 4px;
            top: 0;
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: rgba(255,255,255,0.96);
        }

        .account-person::after {
            content: "";
            position: absolute;
            left: 1px;
            bottom: 0;
            width: 12px;
            height: 9px;
            border-radius: 8px 8px 4px 4px;
            background: rgba(255,255,255,0.96);
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
        "健康分身": "查看长期模式、部位级影响和你的个人健康分身。",
        "风险概览": "查看当前风险、变化原因和今天最该优先处理的事项。",
        "数据记录": "用最少的输入补充今天的行为、部位变化和用药情况。",
        "报告中心": "查看周期复盘，并上传化验报告作为补充材料。",
    }
    st.markdown(
        f"""
        <div class="app-shell-header">
            <div class="app-shell-chip">{page}</div>
            <div class="app-shell-title">痛风管理分身</div>
            <div class="app-shell-subtitle">{subtitles.get(page, "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_new_user_banner(context) -> None:
    if not _is_new_user_context(context):
        return

    steps = _get_onboarding_steps(context)
    completed = len([step for step in steps if step["done"]])
    total = len(steps)
    next_step = next((step for step in steps if not step["done"]), None)

    body = f"先完成 {total} 个起步动作中的 {completed} 个，系统就会开始形成你的健康分身和风险变化。"
    if next_step:
        body += f" 当前最建议先做：{next_step['label']}。"

    chips = "".join(
        [
            f'<span class="bullet-chip" style="background:{("#dfe9d8" if step["done"] else "rgba(239,225,207,0.9)")};color:{("#3d7b54" if step["done"] else "#9d5c2f")}">{("已完成" if step["done"] else "待完成")} · {step["label"]}</span>'
            for step in steps
        ]
    )
    st.markdown(_summary_card("新用户引导", body), unsafe_allow_html=True)
    st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">起步清单</div>{chips}</div>', unsafe_allow_html=True)


def _severity_color(score: float | int | None) -> str:
    if score is None:
        return "#e6ded0"
    value = max(0.0, min(float(score), 10.0))
    if value < 2:
        return "#dfe9d8"
    if value < 4:
        return "#bcd7a8"
    if value < 6:
        return "#f0cf78"
    if value < 8:
        return "#e79c57"
    return "#cf5d4a"


def _build_body_heatmap(site_pain_patterns: dict[str, dict]) -> str:
    normalized_scores: dict[str, float] = {}
    for site, payload in site_pain_patterns.items():
        score = payload.get("max_pain_score") or payload.get("average_pain_score")
        if score is None:
            continue
        normalized_scores[str(site)] = float(score)

    def pick_score(label: str) -> float | None:
        for site, score in normalized_scores.items():
            if label in site or site in label:
                return score
        return None

    markers = []
    labels = []
    for label, x, y in BODY_HEATMAP_POINTS:
        score = pick_score(label)
        color = _severity_color(score)
        markers.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{color}" stroke="#8d775c" stroke-width="2" />')
        labels.append(f'<text x="{x}" y="{y + 28}" text-anchor="middle" font-size="11" fill="#5c4a38">{label}</text>')
        if score is not None:
            labels.append(f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="10" font-weight="700" fill="#2e261f">{score:.0f}</text>')

    return f"""
    <div style="background: rgba(255,253,248,0.92); border: 1px solid rgba(120,93,62,0.14); border-radius: 20px; padding: 0.8rem 0.8rem 0.4rem 0.8rem;">
      <div style="font-weight: 700; color: #2e261f; margin-bottom: 0.2rem;">部位热力图</div>
      <div style="color: #6b5a49; font-size: 0.88rem; margin-bottom: 0.6rem;">颜色越深表示近期该部位疼痛程度越高。</div>
      <svg viewBox="0 0 360 390" width="100%" style="max-width: 360px; display: block; margin: 0 auto;">
        <circle cx="180" cy="42" r="24" fill="#f0e5d4" stroke="#b89d7d" stroke-width="3" />
        <rect x="152" y="72" width="56" height="78" rx="24" fill="#f0e5d4" stroke="#b89d7d" stroke-width="3" />
        <line x1="152" y1="92" x2="110" y2="152" stroke="#b89d7d" stroke-width="12" stroke-linecap="round" />
        <line x1="208" y1="92" x2="250" y2="152" stroke="#b89d7d" stroke-width="12" stroke-linecap="round" />
        <line x1="168" y1="150" x2="148" y2="252" stroke="#b89d7d" stroke-width="14" stroke-linecap="round" />
        <line x1="192" y1="150" x2="212" y2="252" stroke="#b89d7d" stroke-width="14" stroke-linecap="round" />
        <line x1="148" y1="252" x2="138" y2="328" stroke="#b89d7d" stroke-width="12" stroke-linecap="round" />
        <line x1="212" y1="252" x2="222" y2="328" stroke="#b89d7d" stroke-width="12" stroke-linecap="round" />
        <line x1="138" y1="328" x2="124" y2="350" stroke="#b89d7d" stroke-width="10" stroke-linecap="round" />
        <line x1="222" y1="328" x2="236" y2="350" stroke="#b89d7d" stroke-width="10" stroke-linecap="round" />
        {''.join(markers)}
        {''.join(labels)}
      </svg>
      <div style="display: flex; gap: 0.45rem; justify-content: center; flex-wrap: wrap; margin-top: 0.3rem; font-size: 0.8rem; color: #6b5a49;">
        <span><span style="display:inline-block;width:10px;height:10px;background:#dfe9d8;border-radius:999px;margin-right:4px;"></span>轻</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#bcd7a8;border-radius:999px;margin-right:4px;"></span>低</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#f0cf78;border-radius:999px;margin-right:4px;"></span>中</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#e79c57;border-radius:999px;margin-right:4px;"></span>较高</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#cf5d4a;border-radius:999px;margin-right:4px;"></span>高</span>
      </div>
    </div>
    """


def _clear_auth_state() -> None:
    for key in ["authenticated", "current_user_id", "current_username", "current_display_name"]:
        st.session_state.pop(key, None)


def _render_auth_gate(project_root: Path) -> dict[str, str | int] | None:
    if st.session_state.get("authenticated") and st.session_state.get("current_user_id"):
        return {
            "user_id": int(st.session_state["current_user_id"]),
            "username": str(st.session_state.get("current_username") or ""),
            "display_name": str(st.session_state.get("current_display_name") or ""),
        }

    st.markdown(
        """
        <div class="app-shell-header">
            <div class="app-shell-chip">本地账号</div>
            <div class="app-shell-title">欢迎使用痛风健康分身</div>
            <div class="app-shell-subtitle">请先登录或注册本地账号。每个账号都会拥有自己独立的资料、记录和健康分身。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(f"首次体验可使用演示账号：用户名 `{data.DEFAULT_DEMO_USERNAME}`，密码 `{data.DEFAULT_DEMO_PASSWORD}`。")

    login_tab, register_tab = st.tabs(["登录", "注册"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            login_submitted = st.form_submit_button("登录", use_container_width=True)
        if login_submitted:
            account = data.authenticate_user(project_root, username, password)
            if account:
                st.session_state["authenticated"] = True
                st.session_state["current_user_id"] = int(account["user_id"])
                st.session_state["current_username"] = account["username"]
                st.session_state["current_display_name"] = account["display_name"]
                st.rerun()
            else:
                st.error("用户名或密码不正确。")

    with register_tab:
        with st.form("register_form"):
            display_name = st.text_input("显示名称", placeholder="例如：小李")
            username = st.text_input("设置用户名", placeholder="至少 1 个字符")
            password = st.text_input("设置密码", type="password", placeholder="至少 6 位")
            password_confirm = st.text_input("确认密码", type="password")
            register_submitted = st.form_submit_button("注册并登录", use_container_width=True)
        if register_submitted:
            if password != password_confirm:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    account = data.create_account(project_root, username, password, display_name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user_id"] = int(account["user_id"])
                    st.session_state["current_username"] = account["username"]
                    st.session_state["current_display_name"] = account["display_name"]
                    st.rerun()
    return None


def _render_sidebar_account_settings(project_root: Path, orchestrator: AppOrchestrator, context, current_user: dict[str, str | int]) -> None:
    st.markdown(
        """
        <div class="account-entry">
          <div class="account-entry-title">
            <span class="account-avatar"><span class="account-person"></span></span>
            <span>账户设置</span>
          </div>
          <div class="account-entry-subtitle">登录管理、密码修改和基础资料维护都在这里。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    target = st.popover("打开账户设置", use_container_width=True) if hasattr(st, "popover") else st.expander("打开账户设置", expanded=False)
    with target:
        st.write(f"**当前账号**：{current_user.get('username')}")
        st.write(f"**显示名称**：{current_user.get('display_name')}")

        account_tab, password_tab, profile_tab = st.tabs(["登录管理", "密码修改", "基础资料"])
        with account_tab:
            st.caption("退出后会回到登录页。")
            if st.button("退出登录", key="logout_button", use_container_width=True):
                _clear_auth_state()
                st.rerun()
            st.divider()
            st.caption("账号注销后将无法继续登录当前账号。")
            with st.form("deactivate_account_form"):
                deactivate_password = st.text_input("输入当前密码以确认注销", type="password")
                deactivate_submitted = st.form_submit_button("账号注销", use_container_width=True)
            if deactivate_submitted:
                try:
                    data.deactivate_account(project_root, int(current_user["user_id"]), deactivate_password)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _clear_auth_state()
                    st.success("账号已注销。")
                    st.rerun()

        with password_tab:
            with st.form("change_password_form"):
                current_password = st.text_input("当前密码", type="password")
                new_password = st.text_input("新密码", type="password")
                confirm_password = st.text_input("确认新密码", type="password")
                password_submitted = st.form_submit_button("修改密码", use_container_width=True)
            if password_submitted:
                if new_password != confirm_password:
                    st.error("两次输入的新密码不一致。")
                else:
                    try:
                        data.update_account_password(
                            project_root,
                            user_id=int(current_user["user_id"]),
                            current_password=current_password,
                            new_password=new_password,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("密码已更新。")

        with profile_tab:
            gender_keys = list(GENDER_OPTIONS.keys())
            current_gender = context.profile.get("gender") or "unknown"
            gender_index = gender_keys.index(current_gender) if current_gender in gender_keys else 0
            with st.form("sidebar_profile_form"):
                name = st.text_input("姓名", value=context.profile.get("name") or current_user.get("display_name") or "")
                gender = st.selectbox("性别", gender_keys, index=gender_index, format_func=lambda x: GENDER_OPTIONS.get(x, x))
                birth_date = st.text_input("出生日期", value=context.profile.get("birth_date") or "")
                height_cm = st.number_input("身高 (cm)", min_value=0.0, max_value=250.0, value=float(context.profile.get("height_cm") or 170.0), step=1.0)
                baseline_weight_kg = st.number_input("基础体重 (kg)", min_value=0.0, max_value=300.0, value=float(context.profile.get("baseline_weight_kg") or 70.0), step=0.5)
                target_uric_acid = st.number_input("目标尿酸", min_value=0.0, max_value=1000.0, value=float(context.profile.get("target_uric_acid") or 360.0), step=10.0)
                confirm_profile_update = st.checkbox("我确认更新基础资料", value=False)
                profile_submitted = st.form_submit_button("保存基础资料", use_container_width=True)
            if profile_submitted:
                if not confirm_profile_update:
                    st.error("更新基础资料前，请先确认本次修改。")
                else:
                    orchestrator.update_profile(
                        {
                            "name": name,
                            "gender": gender,
                            "birth_date": birth_date or None,
                            "height_cm": height_cm,
                            "baseline_weight_kg": baseline_weight_kg,
                            "target_uric_acid": target_uric_acid,
                        },
                        audit_meta={"source": "account_settings", "confirmed": True},
                    )
                    st.session_state["current_display_name"] = name or current_user.get("display_name") or current_user.get("username")
                    st.success("基础资料已更新。")

        with st.expander("最近敏感操作", expanded=False):
            audit_logs = orchestrator.get_write_audit_logs(limit=8)
            if audit_logs.empty:
                st.caption("最近还没有敏感写操作记录。")
            else:
                audit_view = audit_logs[["created_at", "tool_name", "source", "status", "confirmed_flag"]].copy()
                audit_view["confirmed_flag"] = audit_view["confirmed_flag"].map(lambda value: "已确认" if bool(value) else "未确认")
                audit_view.columns = ["时间", "操作", "来源", "状态", "确认"]
                st.dataframe(audit_view, use_container_width=True, hide_index=True)


def render_app(project_root: Path) -> None:
    _apply_theme()
    current_user = _render_auth_gate(project_root)
    if current_user is None:
        return

    orchestrator = AppOrchestrator(project_root, user_id=int(current_user["user_id"]))
    context = orchestrator.load_context()
    orchestrator.sync_daily_snapshot(context)
    context = orchestrator.load_context()
    snapshot = orchestrator.get_ui_snapshot(context)

    with st.sidebar:
        st.header("导航")
        page = st.radio(
            "前往页面",
            ["健康分身", "风险概览", "数据记录", "报告中心"],
        )
        st.divider()
        journal_profile = context.user_journal.get("profile", {})
        st.write(f"**用户**：{current_user.get('display_name') or journal_profile.get('name') or '未命名用户'}")
        st.caption(f"账号：{current_user.get('username')}")
        st.write(f"**当前发作风险**：{context.risk_overview.get('attack_risk_label') or snapshot['attack_risk_label']}")
        st.caption("本地模型")
        st.write(f"模型：`{snapshot['llm_status']['model']}`")
        st.write(f"接口：`{snapshot['llm_status']['base_url']}`")
        st.divider()
        _render_sidebar_account_settings(project_root, orchestrator, context, current_user)

    _render_shell_header(page)

    if page == "健康分身":
        _render_profile_hub(orchestrator, context)
    elif page == "风险概览":
        _render_risk_hub(orchestrator, context)
    elif page == "数据记录":
        _render_record_hub(orchestrator, context)
    elif page == "报告中心":
        _render_report_center(orchestrator, context)

    _render_global_assistant(orchestrator, context)


def _render_profile_hub(orchestrator: AppOrchestrator, context) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("健康分身")
    st.caption("这里优先展示你的长期模式、部位变化和个人健康分身。")
    _render_new_user_banner(context)
    _render_memory_portrait(context, nested=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_risk_hub(orchestrator: AppOrchestrator, context) -> None:
    snapshot = orchestrator.get_ui_snapshot(context)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("风险概览")
    st.caption("这里聚焦当前风险、变化原因和今天的管理重点。")
    _render_new_user_banner(context)
    _render_dashboard(orchestrator, context, snapshot)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_record_hub(orchestrator: AppOrchestrator, context) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("数据记录")
    st.caption("这里只保留最常用的三类记录：日常行为、疼痛记录和服药记录。")
    if _is_new_user_context(context):
        _render_empty_guide(
            "建议先完成今天的第一轮记录",
            "先从最简单的日常行为开始，再补充疼痛或服药情况，系统就能逐步形成你的个人模式。",
            [
                "先记录今天的饮水、饮酒和是否服药",
                "如果今天有疼痛，再用一句话描述身体部位和疼痛程度",
                "连续记录几天后，健康分身和风险页会开始变得更丰富",
            ],
        )
    tab1, tab2, tab3 = st.tabs(["日常行为", "疼痛记录", "服药记录"])
    with tab1:
        _render_daily_log(orchestrator, context, nested=True, compact=True)
    with tab2:
        _render_pain_log(orchestrator, context)
    with tab3:
        _render_medication_management(orchestrator, context, nested=True, compact=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_report_center(orchestrator: AppOrchestrator, context) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("报告中心")
    st.caption("这里保留两类核心能力：生成周期报告，以及上传化验报告后做 AI 解读。")
    orchestrator.run_pending_background_jobs(limit=5)
    background_jobs = orchestrator.list_background_jobs(limit=8)
    if _is_new_user_context(context):
        _render_empty_guide(
            "报告会在有连续记录后更有价值",
            "当前也可以先生成一版周期报告，但连续记录几天后，报告里的模式和建议会更准确。",
            [
                "先完成几天的日常行为记录",
                "如果有疼痛或发作，再补充疼痛记录",
                "之后再来看周报、月报或上传化验报告做解读",
            ],
        )

    left, right = st.columns([1.3, 0.7], gap="large")
    with left:
        report_type = st.radio(
            "报告类型",
            ["weekly", "monthly"],
            horizontal=True,
            format_func=lambda x: "周报" if x == "weekly" else "月报",
            key="report_center_type",
        )
        report_label = "周报" if report_type == "weekly" else "月报"
        if st.button(f"生成{report_label}", key=f"submit_{report_type}_job", use_container_width=True):
            orchestrator.submit_background_job("report_generation", {"report_type": report_type})
            st.success(f"{report_label}生成任务已加入队列。")
            st.rerun()

        latest_report_summaries = data.get_report_summaries(
            orchestrator.project_root,
            report_type=report_type,
            limit=1,
            user_id=orchestrator.user_id,
        )
        report_payload = {}
        if not latest_report_summaries.empty:
            report_payload = latest_report_summaries.iloc[0].get("summary_payload") or {}
        active_report_jobs = background_jobs.loc[background_jobs["job_type"] == "report_generation"].head(3) if not background_jobs.empty else pd.DataFrame()
        if not active_report_jobs.empty:
            chips = "".join(
                [
                    f'<span class="bullet-chip">{("周报" if row.get("payload", {}).get("report_type") == "weekly" else "月报")} · {row.get("status")}</span>'
                    for _, row in active_report_jobs.iterrows()
                ]
            )
            st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">报告任务状态</div>{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            _summary_card(
                "报告摘要",
                report_payload.get("summary") or "还没有生成本周期报告。点击上方按钮后，系统会在后台生成并在这里更新摘要。",
            ),
            unsafe_allow_html=True,
        )

        action_plan = report_payload.get("action_plan") or []
        action_text = "；".join(action_plan[:4]) if action_plan else "暂无明确建议，建议继续保持记录。"
        st.markdown(_summary_card("建议", action_text), unsafe_allow_html=True)
        st.markdown(
            _summary_card(
                "AI 解读",
                report_payload.get("explanation") or "报告生成完成后，这里会自动显示结合健康分身和近期行为的解读。",
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(_summary_card("化验报告上传", "上传图片或 PDF 后，可在这里结合当前记录查看 AI 解读。"), unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "上传化验报告",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="lab_report_uploads",
        )
        if uploaded_files:
            upload_rows = [
                {
                    "文件名": file.name,
                    "类型": file.type or "未知",
                    "大小": f"{round(file.size / 1024, 1)} KB",
                }
                for file in uploaded_files
            ]
            st.dataframe(pd.DataFrame(upload_rows), use_container_width=True, hide_index=True)
            lab_payloads = [
                {
                    "name": file.name,
                    "type": file.type or "",
                    "size": file.size,
                    "bytes": file.getvalue(),
                }
                for file in uploaded_files
            ]
            if st.button("开始识别化验报告", key="submit_lab_parse_job", use_container_width=True):
                orchestrator.submit_background_job("lab_report_parse", {"uploaded_files": lab_payloads})
                st.success("化验报告识别任务已加入队列。")
                st.rerun()
        else:
            st.caption("暂未上传化验报告。")

        latest_parse_results = data.get_lab_report_parse_results(orchestrator.project_root, limit=1, user_id=orchestrator.user_id)
        active_lab_jobs = background_jobs.loc[background_jobs["job_type"] == "lab_report_parse"].head(3) if not background_jobs.empty else pd.DataFrame()
        if not active_lab_jobs.empty:
            chips = "".join(
                [
                    f'<span class="bullet-chip">{row.get("status")}</span>'
                    for _, row in active_lab_jobs.iterrows()
                ]
            )
            st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">化验识别任务状态</div>{chips}</div>', unsafe_allow_html=True)

        if latest_parse_results.empty:
            st.markdown(_summary_card("AI 解读", "上传并识别化验报告后，这里会显示结合健康分身、近期行为和历史报告的解读。"), unsafe_allow_html=True)
        else:
            latest_parse = latest_parse_results.iloc[0].get("metrics_payload") or {}
            lab_result = orchestrator.explain_parsed_lab_reports(latest_parse, context)
            if lab_result["source"] == "local_llm":
                st.success("本次化验报告解读已结合本地模型生成。")
            elif lab_result["error"]:
                st.warning(lab_result["error"])
            st.markdown(_summary_card("AI 解读", lab_result["answer"]), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_dashboard(orchestrator: AppOrchestrator, context, snapshot: dict) -> None:
    risk_snapshots = orchestrator.registry.call("获取风险快照", 90)
    journal_records = pd.DataFrame(context.user_journal.get("recent_health_records") or [])
    recent_logs = journal_records.tail(7) if not journal_records.empty else context.logs.tail(7)
    recent_site_history = context.site_history.head(7) if not context.site_history.empty else pd.DataFrame()
    risk_overview = context.risk_overview or {}
    is_new_user = _is_new_user_context(context)

    if is_new_user:
        _render_empty_guide(
            "还没有形成稳定风险视图",
            "先连续补充几天记录，系统就会开始生成风险变化、风险因素和管理建议。",
            [
                "先在“数据记录”里记录饮水、饮酒和今天的疼痛情况",
                "如果有明显疼痛或发作，再补充部位和诱因",
                "连续记录后，这里会开始显示趋势、变化原因和重点建议",
            ],
        )
        return

    st.metric("当前风险", risk_overview.get("attack_risk_label") or snapshot["attack_risk_label"])

    st.subheader("变化原因")
    st.markdown(
        _summary_card(
            "为什么会这样",
            _summarize_risk_change(risk_snapshots if not risk_snapshots.empty else pd.DataFrame(), recent_logs, recent_site_history),
        ),
        unsafe_allow_html=True,
    )
    trigger_summary = risk_overview.get("trigger_summary") or snapshot["trigger_summary"]
    if trigger_summary:
        factor_chips = "".join([f'<span class="bullet-chip">{item.get("label")} · {item.get("count")}次</span>' for item in trigger_summary[:5]])
        st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">最近最常出现的风险因素</div>{factor_chips}</div>', unsafe_allow_html=True)
    else:
        st.info("近期还没有识别到明显诱因。")

    st.subheader("今天怎么做")
    st.markdown(
        _summary_card(
            "今天优先做什么",
            "；".join(
                [
                    f"饮水：{risk_overview.get('hydration_advice') or snapshot['hydration_advice']}",
                    f"饮食：{risk_overview.get('diet_advice') or snapshot['diet_advice']}",
                    f"运动：{risk_overview.get('exercise_advice') or snapshot['exercise_advice']}",
                    f"目标：{risk_overview.get('behavior_goal') or snapshot['behavior_goal']}",
                ]
            ),
        ),
        unsafe_allow_html=True,
    )

    abnormal_items = risk_overview.get("abnormal_items") or snapshot["abnormal_items"]
    if abnormal_items:
        chips = "".join([f'<span class="bullet-chip">{item}</span>' for item in abnormal_items[:4]])
        st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">需要关注</div>{chips}</div>', unsafe_allow_html=True)
    else:
        st.success("最近一次数据中未发现明显异常提醒。")

def _render_daily_log(orchestrator: AppOrchestrator, context, nested: bool = False, compact: bool = False) -> None:
    if not nested:
        st.subheader("日常行为")
        st.caption("记录今天最常用的行为与状态信息。")
    with st.form("daily_health_form"):
        log_date = st.date_input("日期", value=date.today())
        water_ml = st.number_input("饮水量 (mL)", min_value=0, max_value=6000, value=2000 if compact else 1800, step=100)
        alcohol_key = st.selectbox("饮酒情况", list(ALCOHOL_OPTIONS.keys()), format_func=lambda x: ALCOHOL_OPTIONS.get(x, x))
        diet_notes = st.text_input("饮食备注", placeholder="例如：海鲜、火锅、烧烤")
        pain_score = st.slider("疼痛评分", min_value=0, max_value=10, value=0)
        joint_pain_flag = st.checkbox("今天有关节痛", value=False)
        medication_taken_flag = st.checkbox("今天已服药", value=True)
        if compact:
            st.caption("推荐先填：饮水 2000 mL、是否饮酒、今天疼不疼、有没有按时服药。")
            symptom_notes = ""
            free_text = ""
        else:
            symptom_notes = st.text_input("部位不适说明（选填）", placeholder="例如：左脚大脚趾轻微疼痛")
            free_text = st.text_area("补充说明（选填）", placeholder="今天还有什么想补充的？")
        submitted = st.form_submit_button("保存日常记录")
    if submitted:
        orchestrator.save_daily_log(
            {
                "log_date": str(log_date),
                "water_ml": water_ml,
                "alcohol_intake": alcohol_key,
                "diet_notes": diet_notes,
                "symptom_notes": symptom_notes,
                "pain_score": pain_score,
                "joint_pain_flag": joint_pain_flag,
                "medication_taken_flag": medication_taken_flag,
                "free_text": free_text,
            }
        )
        st.success("已保存今天的日常记录。")


def _render_site_and_attack_log(orchestrator: AppOrchestrator, context, nested: bool = False, compact: bool = False) -> None:
    if not nested:
        st.subheader("部位症状与发作")
    st.caption("先记录部位症状；如属于明确发作，再补充发作信息。")
    if _is_new_user_context(context):
        st.info("如果今天有明显疼痛，先记录部位和疼痛程度；只有属于一次明确发作时，再补充持续时间和诱因。")

    with st.form("site_attack_form"):
        record_date = st.date_input("记录日期", value=date.today())
        body_site = st.selectbox("身体部位", BODY_SITE_OPTIONS)
        pain_score = st.slider("疼痛程度", min_value=0, max_value=10, value=3 if compact else 4)
        col1, col2, col3 = st.columns(3)
        with col1:
            swelling_flag = st.checkbox("红肿", value=False)
        with col2:
            redness_flag = st.checkbox("发红", value=False)
        with col3:
            stiffness_flag = st.checkbox("僵硬", value=False)
        symptom_notes = st.text_area("症状备注", placeholder="例如：右脚大脚趾下午开始酸痛，走路时更明显")

        st.markdown("**如属于一次明确发作，可补充以下信息**")
        is_attack = st.checkbox("这是一次发作", value=False)
        if compact:
            duration_hours = 24.0
            suspected_trigger = ""
            resolved_flag = False
            if is_attack:
                st.caption("推荐先记录部位、疼痛和是否属于发作；持续时间默认按 24 小时保存，诱因可稍后补充。")
        else:
            duration_hours = st.number_input("持续时间 (小时)", min_value=0.0, max_value=240.0, value=24.0, step=1.0, disabled=not is_attack)
            suspected_trigger = st.text_input("诱因备注", placeholder="例如：啤酒、海鲜、饮水不足", disabled=not is_attack)
            resolved_flag = st.checkbox("是否已缓解", value=False, disabled=not is_attack)

        submitted = st.form_submit_button("保存部位记录")

    if submitted:
        orchestrator.save_joint_symptom(
            {
                "log_date": str(record_date),
                "body_site": body_site,
                "pain_score": pain_score,
                "swelling_flag": swelling_flag,
                "redness_flag": redness_flag,
                "stiffness_flag": stiffness_flag,
                "symptom_notes": symptom_notes,
            }
        )
        if is_attack:
            orchestrator.save_attack(
                {
                    "attack_date": str(record_date),
                    "joint_site": body_site,
                    "pain_score": pain_score,
                    "swelling_flag": swelling_flag,
                    "redness_flag": redness_flag,
                    "duration_hours": duration_hours,
                    "suspected_trigger": suspected_trigger,
                    "resolved_flag": resolved_flag,
                    "notes": symptom_notes,
                }
            )
            st.success("已保存部位症状和发作记录。")
        else:
            st.success("已保存部位症状。")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.subheader("近期部位变化")
        if context.site_history.empty:
            st.info("还没有部位变化记录。")
        else:
            site_display = context.site_history[["event_date", "event_type", "site", "pain_score", "swelling_flag", "redness_flag", "stiffness_flag"]].copy()
            site_display["event_date"] = pd.to_datetime(site_display["event_date"], errors="coerce").dt.date
            site_display["event_type"] = site_display["event_type"].map(lambda x: "发作" if str(x) == "attack" else "症状")
            site_display["swelling_flag"] = site_display["swelling_flag"].map(lambda x: "是" if bool(x) else "否")
            site_display["redness_flag"] = site_display["redness_flag"].map(lambda x: "是" if bool(x) else "否")
            site_display["stiffness_flag"] = site_display["stiffness_flag"].map(lambda x: "是" if bool(x) else "否")
            site_display.columns = ["日期", "类型", "部位", "疼痛程度", "红肿", "发红", "僵硬"]
            st.dataframe(site_display.head(10), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("近期发作记录")
        attack_history = context.site_history.loc[context.site_history["event_type"] == "attack"].copy() if not context.site_history.empty else pd.DataFrame()
        if attack_history.empty:
            st.info("还没有明确发作记录。")
        else:
            display = attack_history[["event_date", "site", "pain_score", "trigger_notes", "resolved_flag"]].copy()
            display["event_date"] = pd.to_datetime(display["event_date"], errors="coerce").dt.date
            display["resolved_flag"] = display["resolved_flag"].map(lambda x: "是" if bool(x) else "否")
            display.columns = ["日期", "部位", "疼痛评分", "诱因备注", "是否缓解"]
            st.dataframe(display, use_container_width=True, hide_index=True)


def _render_pain_log(orchestrator: AppOrchestrator, context) -> None:
    st.subheader("疼痛记录")
    st.caption("用一句话描述今天的不适，系统会自动解析部位、疼痛程度和是否属于一次发作。")
    if _is_new_user_context(context):
        st.info("例如：今天右脚大脚趾疼 6 分，有点红肿，像一次发作。")

    with st.form("pain_text_form", clear_on_submit=True):
        record_date = st.date_input("记录日期", value=date.today(), key="pain_text_date")
        pain_text = st.text_area(
            "不适描述",
            placeholder="例如：今天右脚大脚趾疼 6 分，还有点红肿，像一次发作",
            height=100,
        )
        submitted = st.form_submit_button("解析并保存")

    if submitted:
        candidate = _build_assistant_writeback_candidate(orchestrator, context, pain_text)
        if not candidate or candidate.get("type") != "symptom":
            st.warning("这段描述里还没有识别到明确的疼痛或部位信息，建议补充疼痛程度和身体部位。")
        else:
            symptom_payload = dict(candidate["symptom_payload"])
            symptom_payload["log_date"] = str(record_date)
            orchestrator.save_joint_symptom(symptom_payload)

            if candidate.get("attack_payload"):
                attack_payload = dict(candidate["attack_payload"])
                attack_payload["attack_date"] = str(record_date)
                orchestrator.save_attack(attack_payload)
                st.success("已根据描述自动写入疼痛记录和发作记录。")
            else:
                st.success("已根据描述自动写入疼痛记录。")
            st.caption(candidate.get("summary") or "系统已完成解析并写入。")


def _render_profile_management(orchestrator: AppOrchestrator, context) -> None:
    st.subheader("基础资料")
    profile = context.user_journal.get("profile", {}) or orchestrator.get_profile()

    col1, col2 = st.columns([0.58, 0.42], gap="large")
    with col1:
        st.caption("这部分用于补充基础背景信息。")
        basic_items = {
            "姓名": profile.get("name") or "未填写",
            "性别": GENDER_OPTIONS.get(profile.get("gender") or "unknown", "未知"),
            "出生日期": profile.get("birth_date") or "未填写",
            "身高": f"{float(profile['height_cm']):.0f} cm" if profile.get("height_cm") not in (None, "") else "未填写",
            "基础体重": f"{float(profile['baseline_weight_kg']):.0f} kg" if profile.get("baseline_weight_kg") not in (None, "") else "未填写",
        }
        basic_items = [{"项目": key, "内容": value} for key, value in basic_items.items()]
        st.dataframe(pd.DataFrame(basic_items), use_container_width=True, hide_index=True)
    with col2:
        with st.expander("编辑基础资料", expanded=False):
            _render_profile_form(orchestrator, context, form_key="profile_management_form")


def _render_risk_monitor(orchestrator: AppOrchestrator, context, snapshot: dict, embedded: bool = False) -> None:
    if not embedded:
        st.subheader("风险分析")
    risk_snapshots = orchestrator.registry.call("获取风险快照", 90)
    risk_overview = context.risk_overview or {}

    col1, col2 = st.columns(2)
    col1.metric("当前风险", risk_overview.get("attack_risk_label") or snapshot["attack_risk_label"])
    col2.metric("综合评分", risk_overview.get("overall_risk_score") or snapshot["overall_risk_score"])

    st.write(risk_overview.get("explanation") or snapshot["explanation"])

    if not risk_snapshots.empty:
        frame = risk_snapshots.copy()
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        fig = px.line(frame, x="snapshot_date", y="overall_risk_score", markers=True, title="风险评分趋势")
        st.plotly_chart(fig, use_container_width=True)

    st.write("**需要关注的情况**")
    abnormal_items = risk_overview.get("abnormal_items") or snapshot["abnormal_items"]
    if abnormal_items:
        for item in abnormal_items:
            st.warning(item)
    else:
        st.success("暂无明显异常提醒。")

    st.write("**诱因回顾**")
    trigger_summary = risk_overview.get("trigger_summary") or snapshot["trigger_summary"]
    if trigger_summary:
        trigger_frame = pd.DataFrame(trigger_summary)
        st.dataframe(
            trigger_frame[["label", "count"]].rename(columns={"label": "诱因", "count": "次数"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("最近暂无明显诱因。")


def _render_medication_management(orchestrator: AppOrchestrator, context, nested: bool = False, compact: bool = False) -> None:
    if not nested:
        st.subheader("用药管理")
        st.caption("在这里维护药物方案和服药记录。")
    else:
        st.subheader("服药记录")
        st.caption("先记录今天有没有按时服药；如还没有药物方案，再补充最基本的信息。")
    left, right = st.columns(2, gap="large")

    with left:
        with st.expander("如需补充药物方案", expanded=False):
            with st.form("medication_form"):
                medication_name = st.text_input("药物名称", value="Allopurinol")
                dose = st.text_input("剂量", value="100 mg")
                frequency = st.text_input("频率", value="每日一次")
                if compact:
                    st.caption("先补充药物名称、剂量和频率即可。")
                    start_date = date.today()
                    end_date = date.today() + timedelta(days=30)
                    purpose = "降尿酸"
                    active_flag = True
                else:
                    start_date = st.date_input("开始日期", value=date.today())
                    end_date = st.date_input("结束日期", value=date.today() + timedelta(days=30))
                    purpose = st.text_input("用途", value="降尿酸")
                    active_flag = st.checkbox("启用中", value=True)
                confirm_medication_add = st.checkbox("我确认新增这条药物方案", value=False)
                med_submitted = st.form_submit_button("添加药物")
            if med_submitted:
                if not confirm_medication_add:
                    st.error("新增药物方案前，请先确认本次写入。")
                else:
                    orchestrator.add_medication(
                        {
                            "medication_name": medication_name,
                            "dose": dose,
                            "frequency": frequency,
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "purpose": purpose,
                            "active_flag": active_flag,
                        },
                        audit_meta={"source": "medication_form", "confirmed": True},
                    )
                    st.success("药物已添加，请刷新或切换页面查看最新结果。")

    with right:
        st.write("**当前用药方案**")
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
            selected_label = st.selectbox("选择药物", list(medication_choices.keys())) if medication_choices else None
            status = st.selectbox("服药状态", status_keys, format_func=lambda x: STATUS_OPTIONS.get(x, x)) if medication_choices else None
            if medication_choices and st.button("保存服药状态"):
                taken_time = datetime.now().replace(microsecond=0).isoformat(sep=" ") if status == "taken" else None
                orchestrator.log_medication_taken(medication_choices[selected_label], status, taken_time)
                st.success("服药状态已保存，请刷新或切换页面查看最新结果。")

        adherence = orchestrator.registry.call("获取服药依从性", 30)
        if not adherence.empty:
            st.write("**近 30 天服药记录**")
            adherence_view = adherence[["medication_name", "status", "taken_time", "created_at"]].copy()
            adherence_view["status"] = adherence_view["status"].map(lambda x: STATUS_OPTIONS.get(x, x))
            adherence_view.columns = ["药物名称", "状态", "服药时间", "记录时间"]
            st.dataframe(adherence_view, use_container_width=True, hide_index=True)


def _run_assistant_question(orchestrator: AppOrchestrator, context, question: str) -> None:
    cleaned = (question or "").strip()
    if not cleaned:
        return
    st.session_state["assistant_last_question"] = cleaned
    st.session_state["assistant_last_result"] = orchestrator.answer_coach_question(cleaned, context)


def _render_global_assistant(orchestrator: AppOrchestrator, context) -> None:
    st.session_state.setdefault("assistant_last_question", "")
    st.session_state.setdefault("assistant_last_result", None)
    st.session_state.setdefault("assistant_writeback_notice", "")

    st.markdown(
        """
        <div class="assistant-dock">
          <div class="assistant-dock-card">
            <div class="assistant-dock-title">
              <span class="assistant-avatar"><span class="assistant-bubble"></span></span>
            <span>智能问答助手</span>
          </div>
            <div class="assistant-dock-subtitle">随时提问，结合你的全局记录和健康分身给出分析解答。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dock_left, dock_right = st.columns([4.2, 1.8], gap="large")
    with dock_right:
        if hasattr(st, "popover"):
            with st.popover("智能问答助手", use_container_width=True):
                _render_assistant_panel(orchestrator, context)
        else:
            with st.expander("智能问答助手", expanded=False):
                _render_assistant_panel(orchestrator, context)


def _render_assistant_panel(orchestrator: AppOrchestrator, context) -> None:
    st.caption("这是一个专注问答和分析的对话助手，会结合你的全局记录和健康分身回答问题。")
    notice = st.session_state.get("assistant_writeback_notice")
    if notice:
        st.success(notice)
        st.session_state["assistant_writeback_notice"] = ""

    with st.form("assistant_question_form", clear_on_submit=True):
        question = st.text_input("想问什么？", placeholder="例如：为什么最近右脚大脚趾更容易疼？")
        submitted = st.form_submit_button("发送")
    if submitted:
        _run_assistant_question(orchestrator, context, question)

    st.caption("快捷问题")
    quick_cols = st.columns(3)
    quick_questions = ["今天该注意什么", "为什么风险变了", "今晚该避开什么"]
    for col, quick_question in zip(quick_cols, quick_questions):
        if col.button(quick_question, key=f"assistant_quick_{quick_question}"):
            _run_assistant_question(orchestrator, context, quick_question)

    last_result = st.session_state.get("assistant_last_result")
    if last_result:
        if last_result["source"] == "local_llm":
            st.success("本次回答已结合本地模型生成。")
        else:
            st.warning(last_result["error"] or "本地模型暂时不可用，系统已自动切换为规则引擎回答。")
        st.caption(f"当前处理技能：{last_result['skill']}")
        st.markdown("**分析结果**")
        st.write(last_result["answer"])
        _render_assistant_writeback(orchestrator, context, st.session_state.get("assistant_last_question") or "")


def _render_assistant_writeback(orchestrator: AppOrchestrator, context, question: str) -> None:
    candidate = _build_assistant_writeback_candidate(orchestrator, context, question)
    if not candidate:
        return

    st.markdown("**快捷写入**")
    st.caption("识别到你可能在描述一条记录，如需保存，可以直接在这里确认。")

    if candidate["type"] == "medication":
        medications = context.medications.copy()
        if medications.empty:
            st.info("当前还没有药物方案，先到“数据记录”里补充药物后，再从小助手里快捷打卡。")
            return
        active = medications.loc[medications["active_flag"] == 1] if "active_flag" in medications.columns else medications
        if active.empty:
            active = medications
        options = {f"{row['medication_name']} ({row['dose']})": int(row["id"]) for _, row in active.iterrows()}
        selected = st.selectbox("选择药物", list(options.keys()), key="assistant_writeback_medication")
        button_label = "记录已服药" if candidate["status"] == "taken" else "记录漏服"
        if st.button(button_label, key="assistant_writeback_medication_button", use_container_width=True):
            taken_time = datetime.now().replace(microsecond=0).isoformat(sep=" ") if candidate["status"] == "taken" else None
            orchestrator.log_medication_taken(options[selected], candidate["status"], taken_time)
            st.session_state["assistant_writeback_notice"] = _build_assistant_after_writeback_message(
                orchestrator,
                action_type="medication",
                medication_status=candidate["status"],
            )
            st.rerun()
        return

    st.caption(candidate["summary"])
    action_cols = st.columns(2 if candidate.get("attack_payload") else 1)
    if action_cols[0].button("写入部位症状", key="assistant_writeback_symptom", use_container_width=True):
        orchestrator.save_joint_symptom(candidate["symptom_payload"])
        st.session_state["assistant_writeback_notice"] = _build_assistant_after_writeback_message(
            orchestrator,
            action_type="symptom",
            body_site=candidate["symptom_payload"].get("body_site"),
        )
        st.rerun()
    if candidate.get("attack_payload") and len(action_cols) > 1:
        if action_cols[1].button("写入为发作记录", key="assistant_writeback_attack", use_container_width=True):
            orchestrator.save_joint_symptom(candidate["symptom_payload"])
            orchestrator.save_attack(candidate["attack_payload"])
            st.session_state["assistant_writeback_notice"] = _build_assistant_after_writeback_message(
                orchestrator,
                action_type="attack",
                body_site=candidate["attack_payload"].get("joint_site"),
            )
            st.rerun()


def _build_assistant_writeback_candidate(orchestrator: AppOrchestrator, context, question: str) -> dict | None:
    cleaned = (question or "").strip()
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if any(keyword in cleaned for keyword in ["已服药", "刚吃药", "吃药了", "服药了", "漏服", "忘记吃药", "没服药"]):
        status = "missed" if any(keyword in cleaned for keyword in ["漏服", "忘记吃药", "没服药"]) else "taken"
        return {"type": "medication", "status": status}

    parsed = orchestrator.parse_intake_text(cleaned)
    body_site = _extract_body_site_from_text(cleaned) or "其他"
    has_symptom = bool(parsed.get("joint_pain_flag") or parsed.get("pain_score") or parsed.get("symptom_notes"))
    if not has_symptom:
        return None

    log_date = parsed.get("log_date") or str(date.today())
    pain_score = int(parsed.get("pain_score") or 0)
    swelling_flag = any(keyword in cleaned for keyword in ["红肿", "肿", "肿胀"])
    redness_flag = any(keyword in cleaned for keyword in ["发红", "红"])
    stiffness_flag = any(keyword in cleaned for keyword in ["僵", "僵硬"])
    symptom_notes = parsed.get("symptom_notes") or cleaned
    symptom_payload = {
        "log_date": log_date,
        "body_site": body_site,
        "pain_score": pain_score,
        "swelling_flag": swelling_flag,
        "redness_flag": redness_flag,
        "stiffness_flag": stiffness_flag,
        "symptom_notes": symptom_notes,
    }
    attack_like = ("发作" in cleaned) or pain_score >= 6 or swelling_flag or redness_flag
    attack_payload = None
    if attack_like:
        attack_payload = {
            "attack_date": log_date,
            "joint_site": body_site,
            "pain_score": pain_score,
            "swelling_flag": swelling_flag,
            "redness_flag": redness_flag,
            "duration_hours": 24.0,
            "suspected_trigger": _extract_trigger_hint(cleaned),
            "resolved_flag": False,
            "notes": symptom_notes,
        }

    summary = f"识别到你在描述{body_site}的不适，疼痛约 {pain_score}/10。"
    if attack_like:
        summary += " 如果这是一次明确发作，也可以直接写入为发作记录。"
    return {"type": "symptom", "summary": summary, "symptom_payload": symptom_payload, "attack_payload": attack_payload}


def _build_assistant_after_writeback_message(
    orchestrator: AppOrchestrator,
    action_type: str,
    body_site: str | None = None,
    medication_status: str | None = None,
) -> str:
    refreshed_context = orchestrator.load_context()
    risk_overview = refreshed_context.risk_overview or {}
    risk_label = risk_overview.get("attack_risk_label") or "未知"
    hydration_advice = risk_overview.get("hydration_advice")
    diet_advice = risk_overview.get("diet_advice")
    behavior_goal = risk_overview.get("behavior_goal")

    if action_type == "medication":
        prefix = "已通过小助手写入已服药记录。" if medication_status == "taken" else "已通过小助手写入漏服记录。"
        tips: list[str] = [f"当前发作风险为{risk_label}"]
        if medication_status == "missed":
            tips.append("今天建议尽快按医嘱处理漏服情况，并留意疼痛或红肿是否加重")
        elif behavior_goal:
            tips.append(f"今天优先做到：{behavior_goal}")
        if hydration_advice:
            tips.append(f"饮水方面：{hydration_advice}")
        return f"{prefix} {'；'.join(tips[:3])}。"

    site_label = body_site or "该部位"
    if action_type == "attack":
        prefix = f"已通过小助手写入{site_label}的发作记录。"
        tips = [f"当前发作风险为{risk_label}"]
        if hydration_advice:
            tips.append(f"先做：{hydration_advice}")
        if diet_advice:
            tips.append(f"饮食上：{diet_advice}")
        tips.append("如果疼痛、红肿持续加重，建议尽快线下就医")
        return f"{prefix} {'；'.join(tips[:4])}。"

    prefix = f"已通过小助手写入{site_label}的部位症状记录。"
    tips = [f"当前发作风险为{risk_label}"]
    if behavior_goal:
        tips.append(f"今天优先：{behavior_goal}")
    elif hydration_advice:
        tips.append(f"建议先做：{hydration_advice}")
    if diet_advice:
        tips.append(f"另外注意：{diet_advice}")
    return f"{prefix} {'；'.join(tips[:3])}。"


def _extract_body_site_from_text(text: str) -> str | None:
    aliases = {
        "左脚大脚趾": ["左脚大脚趾", "左大脚趾", "左脚趾"],
        "右脚大脚趾": ["右脚大脚趾", "右大脚趾", "右脚趾"],
        "左脚踝": ["左脚踝", "左踝"],
        "右脚踝": ["右脚踝", "右踝"],
        "左膝": ["左膝", "左膝盖"],
        "右膝": ["右膝", "右膝盖"],
        "左足背": ["左足背", "左脚背"],
        "右足背": ["右足背", "右脚背"],
    }
    for canonical, candidates in aliases.items():
        if any(candidate in text for candidate in candidates):
            return canonical
    return None


def _extract_trigger_hint(text: str) -> str:
    trigger_keywords = ["啤酒", "海鲜", "火锅", "烧烤", "饮水不足", "熬夜", "烈酒", "葡萄酒"]
    hits = [keyword for keyword in trigger_keywords if keyword in text]
    return "、".join(hits) if hits else ""


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

    st.markdown(_summary_card("报告概要", report.get("executive_summary") or report.get("summary") or "暂无摘要。"), unsafe_allow_html=True)

    personal_pattern_summary = report.get("personal_pattern_summary")
    digital_twin_profile = report.get("digital_twin_profile") or {}
    if personal_pattern_summary:
        st.markdown(_summary_card("个人模式概览", personal_pattern_summary), unsafe_allow_html=True)
        site_pain_patterns = digital_twin_profile.get("site_pain_patterns") or {}
        if site_pain_patterns:
            st.markdown(_build_body_heatmap(site_pain_patterns), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    latest_risk = report.get("latest_risk") or {}
    c1.metric("发作风险", latest_risk.get("attack_risk_level_cn") or latest_risk.get("attack_risk_level") or "未知")
    c2.metric("整体风险评分", latest_risk.get("overall_risk_score") or "-")
    c3.metric("记录条数", report.get("entries") or 0)

    key_findings = report.get("key_findings") or []
    if key_findings:
        chips = "".join([f'<span class="bullet-chip">{item}</span>' for item in key_findings[:5]])
        st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">关键结论</div>{chips}</div>', unsafe_allow_html=True)
    else:
        st.caption("本期暂时没有足够记录形成关键结论。")

    action_plan = report.get("action_plan") or []
    if action_plan:
        chips = "".join([f'<span class="bullet-chip">{item}</span>' for item in action_plan[:5]])
        st.markdown(f'<div class="bullet-card"><div class="bullet-card-title">后续建议</div>{chips}</div>', unsafe_allow_html=True)
    else:
        st.caption("本期暂时没有形成明确行动建议。")

    st.markdown(_summary_card("就医建议", report.get("medical_notice") or "如症状持续或加重，请及时线下就医。"), unsafe_allow_html=True)

    with st.expander("查看原始报告内容", expanded=False):
        st.json(report)


def _render_memory_portrait(context, nested: bool = False) -> None:
    if not nested:
        st.subheader("健康分身")
        st.caption("这里展示你的部位变化、近期行为和个人痛风模式。")

    memory_payload = context.twin_state or {}
    portraits = memory_payload.get("behavior_portraits") or {}
    twin_profile = memory_payload.get("digital_twin_profile") or {}
    site_pain_patterns = twin_profile.get("site_pain_patterns") or {}
    is_new_user = _is_new_user_context(context)

    st.markdown("**个人健康分身**")
    if is_new_user:
        _render_empty_guide(
            "你的健康分身还在形成中",
            "分身会根据你的日常行为、部位症状和发作记录逐步学习你的个人模式。",
            [
                "先连续记录几天饮水、饮酒、疼痛和服药情况",
                "有明显不适时补充部位症状，发作时再补充持续时间和诱因",
                "连续记录后，这里会开始显示部位热力图、近期行为和你的痛风模式",
            ],
        )
        return

    st.markdown(_summary_card("我的痛风模式", twin_profile.get("summary") or "当前还没有形成稳定的个人痛风模式，请继续补充记录。"), unsafe_allow_html=True)

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        st.markdown("**部位热力图**")
        st.markdown(_build_body_heatmap(site_pain_patterns), unsafe_allow_html=True)

    with top_right:
        st.markdown("**近期行为**")
        st.caption("用最近 7 / 30 / 90 天的行为变化，观察近期管理状态。")
        portrait_tabs = st.tabs(["近 7 天", "近 30 天", "近 90 天"])
        for tab, key in zip(portrait_tabs, ["7d", "30d", "90d"]):
            portrait = portraits.get(key) or {}
            with tab:
                if not portrait:
                    st.info("暂无画像数据。")
                    continue
                st.markdown(_summary_card("阶段摘要", portrait.get("summary") or "暂无画像摘要。"), unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("记录天数", portrait.get("days_with_logs") or 0)
                m2.metric("平均饮水", f"{float(portrait['average_water_ml']):.0f} mL" if portrait.get("average_water_ml") is not None else "-")
                m3.metric("饮酒天数", portrait.get("alcohol_days") or 0)
                m4.metric("发作次数", portrait.get("attack_count") or 0)
                extra_bits = []
                if portrait.get("pain_days"):
                    extra_bits.append(f"记录到 {int(portrait['pain_days'])} 天疼痛")
                if portrait.get("medication_taken_rate") is not None:
                    extra_bits.append(f"服药完成率约 {float(portrait['medication_taken_rate']):.0f}%")
                if portrait.get("symptom_log_count"):
                    extra_bits.append(f"补充了 {int(portrait['symptom_log_count'])} 条部位症状")
                if extra_bits:
                    st.caption("；".join(extra_bits))

    st.markdown("**我的痛风模式**")
    mode_lines: list[str] = []
    trigger_items = twin_profile.get("top_triggers") or []
    if trigger_items:
        trigger_text = "、".join([item.get("label") or "未知因素" for item in trigger_items[:3]])
        mode_lines.append(f"近期最常见的影响因素包括：{trigger_text}。")
    if site_pain_patterns:
        sorted_patterns = sorted(
            site_pain_patterns.items(),
            key=lambda item: ((item[1].get("attack_count") or 0), (item[1].get("average_pain_score") or 0)),
            reverse=True,
        )
        site, payload = sorted_patterns[0]
        mode_lines.append(
            f"{site}是近期最需要关注的部位，平均疼痛约 {float(payload.get('average_pain_score') or 0):.1f} 分，近期发作 {int(payload.get('attack_count') or 0)} 次。"
        )
    management_stability = twin_profile.get("management_stability") or {}
    if management_stability.get("summary"):
        mode_lines.append(str(management_stability.get("summary")))
    shortcomings = twin_profile.get("current_shortcomings") or []
    if shortcomings:
        mode_lines.append(f"接下来优先改进：{'、'.join(shortcomings[:3])}。")

    st.markdown(
        _summary_card(
            "长期模式解读",
            " ".join(mode_lines) or "继续补充一段时间的行为、症状和发作记录后，系统会逐步形成更稳定的个人痛风模式。",
        ),
        unsafe_allow_html=True,
    )


def _render_empty_guide(title: str, body: str, steps: list[str]) -> None:
    st.info(body)
    st.markdown(f"**{title}**")
    for index, step in enumerate(steps, start=1):
        st.write(f"{index}. {step}")


def _get_onboarding_steps(context) -> list[dict[str, Any]]:
    profile = context.user_journal.get("profile", {}) if getattr(context, "user_journal", None) else {}
    has_profile = bool(profile.get("name") or profile.get("birth_date") or profile.get("height_cm"))
    has_logs = not context.logs.empty
    has_pain = not context.symptom_logs.empty or not context.attacks.empty
    has_reports = len(context.logs) >= 3 or len(context.attacks) >= 1
    return [
        {"label": "完善基础资料", "done": has_profile},
        {"label": "记录日常行为", "done": has_logs},
        {"label": "补充疼痛记录", "done": has_pain},
        {"label": "查看第一份报告", "done": has_reports},
    ]


def _summary_card(title: str, body: str) -> str:
    return f"""
    <div class="summary-card">
      <div class="summary-card-title">{title}</div>
      <div class="summary-card-body">{body}</div>
    </div>
    """


def _is_new_user_context(context) -> bool:
    has_logs = not context.logs.empty
    has_symptoms = not context.symptom_logs.empty
    has_attacks = not context.attacks.empty
    twin_profile = (context.twin_state or {}).get("digital_twin_profile") or {}
    has_twin = bool((twin_profile.get("top_triggers") or []) or (twin_profile.get("site_pain_patterns") or {}) or twin_profile.get("summary"))
    return not any([has_logs, has_symptoms, has_attacks, has_twin])


def _summarize_risk_change(risk_snapshots: pd.DataFrame, recent_logs: pd.DataFrame, recent_site_history: pd.DataFrame) -> str:
    change_text = "当前数据量还不够，建议先连续记录几天观察变化。"
    reasons: list[str] = []

    if not risk_snapshots.empty and "overall_risk_score" in risk_snapshots.columns:
        clean = risk_snapshots.dropna(subset=["overall_risk_score"]).copy()
        clean = clean.sort_values("snapshot_date")
        if len(clean) >= 2:
            current_score = float(clean.iloc[-1]["overall_risk_score"])
            previous_score = float(clean.iloc[-2]["overall_risk_score"])
            delta = current_score - previous_score
            if abs(delta) < 1:
                change_text = "整体风险与上一次相比基本持平。"
            elif delta > 0:
                change_text = f"整体风险较上一次上升了 {delta:.0f} 分。"
            else:
                change_text = f"整体风险较上一次下降了 {abs(delta):.0f} 分。"

    water_avg = _mean_numeric(recent_logs, "water_ml")
    if water_avg is not None and water_avg < 1800:
        reasons.append("近 7 天饮水仍然偏低")
    if not recent_logs.empty and "alcohol_intake" in recent_logs.columns:
        alcohol_days = int((recent_logs["alcohol_intake"].fillna("").astype(str).str.lower() != "none").sum())
        if alcohol_days:
            reasons.append(f"近 7 天记录到 {alcohol_days} 天饮酒")
    pain_avg = _mean_numeric(recent_logs, "pain_score")
    if pain_avg is not None and pain_avg > 0:
        reasons.append(f"近期平均疼痛约 {pain_avg:.1f}/10")
    if not recent_site_history.empty:
        attack_count = int((recent_site_history["event_type"].astype(str) == "attack").sum())
        if attack_count:
            reasons.append(f"近期有 {attack_count} 次明确发作记录")

    if reasons:
        return f"{change_text} 主要与{ '、'.join(reasons[:3]) }有关。"
    return change_text


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
        doctor_advice = st.text_area("长期管理建议", value=context.profile.get("doctor_advice") or "", help="填写需要长期参考的管理建议或注意事项。")
        confirm_profile_update = st.checkbox("我确认更新基础资料", value=False)
        submitted = st.form_submit_button("保存基础资料")
    if submitted:
        if not confirm_profile_update:
            st.error("更新基础资料前，请先确认本次修改。")
        else:
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
                },
                audit_meta={"source": "profile_form", "confirmed": True},
            )
            st.success("基础资料已更新，请刷新或切换页面查看最新结果。")


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.mean()), 1)
