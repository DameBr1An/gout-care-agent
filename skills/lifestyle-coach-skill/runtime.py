from __future__ import annotations


FOOD_HINTS = {
    "啤酒": "当前阶段建议尽量避免，尤其在发作风险偏高时更应严格控制。",
    "海鲜": "属于高嘌呤食物，近期风险较高时建议暂时避免或明显减少摄入。",
    "烧烤": "常伴随高脂和高嘌呤饮食，建议减少频率，并同步增加饮水量。",
    "火锅": "如果包含海鲜、动物内脏或大量红肉，发作风险往往会进一步升高。",
    "鸡蛋": "通常属于相对友好的低嘌呤选择，可作为更稳妥的蛋白来源。",
    "牛奶": "低脂奶制品通常是更友好的选择，可作为日常饮食中的补充。",
    "豆腐": "多数情况下可以适量食用，但建议结合当天整体风险控制总量。",
}


def build_daily_lifestyle_guidance(context: dict) -> str:
    risk_data = context["risk_result"]
    return (
        "今日管理建议：\n"
        f"饮水建议：{risk_data['hydration_advice']}\n"
        f"饮食建议：{risk_data['diet_advice']}\n"
        f"运动建议：{risk_data['exercise_advice']}\n"
        f"今日重点：{risk_data['behavior_goal']}"
    )


def answer_food_question(question: str, context: dict) -> str:
    for keyword, answer in FOOD_HINTS.items():
        if keyword in question:
            return (
                f"{keyword}：{answer} 当前发作风险为 {context['risk_result']['attack_risk_level_cn']}。"
                " 建议结合今天的症状、饮水和尿酸情况谨慎选择。"
            )
    return build_daily_lifestyle_guidance(context)
