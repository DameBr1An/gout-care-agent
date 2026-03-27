from __future__ import annotations


def explain_report(report_payload: dict, context: dict) -> str:
    latest = context["risk_result"]
    parts = [
        (
            f"本期报告覆盖 {report_payload.get('period')}。期间共记录 {report_payload.get('entries')} 条健康数据，"
            f"平均饮水 {report_payload.get('mean_water_ml') if report_payload.get('mean_water_ml') is not None else '暂无数据'} mL，"
            f"当前发作风险为 {latest['attack_risk_level_cn']}。"
        ),
        "建议本周继续把重点放在规律补水、避免已识别诱因和按时用药上。",
        "如果疼痛、红肿或异常指标持续加重，请及时线下就医。",
    ]
    return " ".join(parts)
