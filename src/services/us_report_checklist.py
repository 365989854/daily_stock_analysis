"""The six-dimension checklist contract for implicit US trend reports only."""

from copy import deepcopy
import json
import math
import re

from data_provider.us_index_mapping import is_us_stock_code


TITLES = ("多头排列", "乖离率合理", "量能配合", "无重大利空", "成交量价格结构（Volume Profile）", "PE估值合理")
MODE = "us_default_trend"
NO_PROFILE = "暂无可用 Volume Profile 数据，未作判断"
CONTRACT = """## 默认美股趋势报告检查清单
dashboard.battle_plan.action_checklist 必须恰好六项，按检查项1至6排列，不得遗漏或合并：
1. 多头排列
2. 乖离率合理
3. 量能配合
4. 无重大利空
5. 成交量价格结构（Volume Profile）
6. PE估值合理
第5项只评价提供的 POC、VAH、VAL、当前价格所在价值区和历史成交量密集区。
不得使用筹码健康、筹码缺失、套牢盘、主力成本、真实持仓成本、获利比例等概念；不得生成 profit_ratio、avg_cost、concentration。
无数据仍保留第5项，写“暂无可用 Volume Profile 数据，未作判断”，不编造正面或负面结论。
使用既有 Volume Profile，不重新计算指标；数据不足不等于检查通过。价格使用美元 $。
"""


def default_us_prompt(prompt):
    """Adapt a selected default template, without changing shared CN/skill templates."""
    prompt = re.sub(
        r"### 3\. 效率优先（筹码结构）.*?(?=### 4\.)",
        "### 3. 成交量价格结构（Volume Profile）\n仅评价历史 OHLCV 成交量价格分布。\n\n",
        prompt, flags=re.S,
    )
    prompt = re.sub(r'"chip_structure":\s*\{[^{}]*\}', '"chip_structure": {}', prompt)
    examples = [f"✅/⚠️/❌ 检查项{i}：{title}" for i, title in enumerate(TITLES, 1)]
    prompt = re.sub(r'"action_checklist":\s*\[[^\]]*\]',
                    lambda _: '"action_checklist": ' + json.dumps(examples, ensure_ascii=False), prompt)
    prompt = prompt.replace("**第二阶段 · 技术与筹码**", "**第二阶段 · 技术与成交量价格结构**")
    prompt = prompt.replace("- `get_chip_distribution` 获取筹码分布", "- 使用上下文中的 Volume Profile；不可用时明确未作判断")
    prompt = prompt.replace("- ✅ 筹码集中健康", "- 成交量价格结构仅作为历史分布参考，不作为独立买入条件")
    prompt = prompt.replace("量能/筹码", "量能/成交量价格结构")
    return prompt + "\n" + CONTRACT


def bind_default_us_checklist(result, *, legacy, volume_profile):
    """Persist host-selected mode and source facts; never infer mode from item number."""
    if legacy is not True or not is_us_stock_code(getattr(result, "code", "")):
        return
    if not isinstance(getattr(result, "market_snapshot", None), dict):
        result.market_snapshot = {}
    result.market_snapshot["checklist_context"] = {
        "mode": MODE,
        "volume_profile": deepcopy(volume_profile) if isinstance(volume_profile, dict) else None,
    }
    result.market_snapshot["chip_status"] = "not_supported"
    if isinstance(getattr(result, "dashboard", None), dict):
        perspective = result.dashboard.get("data_perspective") or {}
        result.dashboard["data_perspective"] = perspective
        if isinstance(volume_profile, dict) and volume_profile:
            perspective["volume_profile"] = deepcopy(volume_profile)
        else:
            perspective.pop("volume_profile", None)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _profile_description(profile):
    """Describe existing levels, not ownership or a fabricated pass/fail judgment."""
    if not isinstance(profile, dict):
        return None
    levels = {key: _number(profile.get(key)) for key in ("poc", "vah", "val", "current_price")}
    if any(value is None for value in levels.values()) or not levels["val"] <= levels["poc"] <= levels["vah"]:
        return None
    position = {
        "above_value_area": "价格位于价值区上方",
        "below_value_area": "价格位于价值区下方",
        "inside_value_area_above_poc": "价格位于价值区内、POC上方",
        "inside_value_area_below_poc": "价格位于价值区内、POC下方",
    }.get(profile.get("position"), "价格位置未确认")
    text = (f"{position}；POC ${levels['poc']:.2f}，VAH ${levels['vah']:.2f}，"
            f"VAL ${levels['val']:.2f}，当前价 ${levels['current_price']:.2f}")
    nodes = []
    for node in (profile.get("high_volume_nodes") or [])[:3]:
        if isinstance(node, dict):
            low, high = _number(node.get("low")), _number(node.get("high"))
            if low is not None and high is not None and low <= high:
                nodes.append(f"${low:.2f}–${high:.2f}")
    if nodes:
        text += "；历史高成交量密集区 " + "、".join(nodes)
    return text


def restore_default_us_checklist(result):
    """Keep six dimensions after semantic validation; explicit skills are untouched."""
    snapshot = getattr(result, "market_snapshot", None) or {}
    context = snapshot.get("checklist_context") or {}
    if context.get("mode") != MODE or not is_us_stock_code(getattr(result, "code", "")):
        return
    battle = result.dashboard.get("battle_plan") or {}
    result.dashboard["battle_plan"] = battle
    checks = battle.get("action_checklist") or []
    by_number = {}
    for item in checks:
        if isinstance(item, str):
            match = re.search(r"检查项\s*([1-6])\s*[：:]", item)
            if match:
                by_number.setdefault(int(match[1]), item)
    output = []
    for number, title in enumerate(TITLES, 1):
        item = by_number.get(number, "")
        if number == 5:
            description = _profile_description(context.get("volume_profile"))
            if description is None:
                item = f"⚪ 检查项5：{title}：{NO_PROFILE}"
            elif (title not in item or NO_PROFILE in item or re.search(
                r"筹码|套牢盘|主力成本|持仓成本|获利比例|profit_ratio|avg_cost|concentration", item,
            )):
                item = f"ℹ️ 检查项5：{title}：{description}（仅描述历史成交量价格分布）"
        elif title not in item:
            item = f"⚪ 检查项{number}：{title}（模型未提供有效评价，未作判断）"
        output.append(item)
    battle["action_checklist"] = output
