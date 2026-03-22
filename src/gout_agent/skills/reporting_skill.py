from __future__ import annotations


def explain_report(report_payload: dict, context: dict) -> str:
    latest = context["risk_result"]

    parts = [
        "本期报告覆盖 {period}。期间共记录 {entries} 条健康数据，平均饮水 {water} mL，当前发作风险为 {attack}。".format(
            period=report_payload.get("period"),
            entries=report_payload.get("entries"),
            water=report_payload.get("mean_water_ml") if report_payload.get("mean_water_ml") is not None else "暂无数据",
            attack=latest["attack_risk_level_cn"],
        )
    ]

    parts.append("建议本周继续把重点放在规律补水、避免已识别诱因和按时用药上。")
    parts.append("如果疼痛、红肿或异常指标持续加重，请及时线下就医。")
    return " ".join(parts)
