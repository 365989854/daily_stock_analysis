"""Reconcile model reports with source facts; keep raw responses for auditing."""

from copy import deepcopy
import math
import re

from data_provider.us_index_mapping import is_us_stock_code
from src.report_language import get_report_volume_analysis, normalize_report_language


def reconcile_report(result):
    """Validate structured model output before persistence and again on display."""
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        return result
    perspective = dashboard.get("data_perspective") or {}
    dashboard["data_perspective"] = perspective
    volume = get_report_volume_analysis(result, getattr(result, "report_language", "zh"))
    if volume:
        perspective["volume_analysis"] = volume
    snapshot = getattr(result, "market_snapshot", None) or {}
    overview = getattr(result, "analysis_context_pack_overview", None) or {}
    blocks = overview.get("blocks", [])
    if is_us_stock_code(getattr(result, "code", "")) and volume and volume.get("volume_ratio") is None:
        # Daily totals support a comparison, not claims about intraday intention.
        result.volume_analysis = volume.get("volume_meaning", "")

        def remove_intraday_inference(value):
            if isinstance(value, dict):
                return {k: remove_intraday_inference(v) for k, v in value.items()}
            if isinstance(value, list):
                return [v for item in value if not isinstance(v := remove_intraday_inference(item), str) or v.strip()]
            if isinstance(value, str):
                value = re.sub(r"[^，,。；;]*(?:追高意愿|盘中成交意愿)[^，,。；;]*[，,。；;]?", "", value)
                return re.sub(r"[^，,。；;]*量比\s*(?:为|仅为|=|：|:)?\s*\d[^，,。；;]*[，,。；;]?", "", value)
            return value

        dashboard = remove_intraday_inference(dashboard)
        result.dashboard = dashboard
        perspective = dashboard["data_perspective"]
        for key in ("analysis_summary", "risk_warning", "guardrail_reason", "short_term_outlook"):
            value = getattr(result, key, None)
            if isinstance(value, str):
                setattr(result, key, remove_intraday_inference(value))
    unsupported = snapshot.get("chip_status") == "not_supported" or any(
        b.get("key") == "chip" and b.get("status") == "not_supported"
        for b in blocks if isinstance(b, dict)
    )
    if is_us_stock_code(getattr(result, "code", "")) and unsupported:
        perspective.pop("chip_structure", None)
        perspective["chip_unavailable_reason"] = {
            "zh": "不适用（美股不适用 A 股口径筹码分布）",
            "en": "Not applicable (A-share chip distribution does not apply to US stocks)",
            "ko": "해당 없음 (미국 주식에는 A주 매물 분포가 적용되지 않음)",
        }[normalize_report_language(getattr(result, "report_language", "zh"))]
        perspective["chip_not_applicable"] = True

        def clean_clause(match):
            clause = match[0]
            # Chip health is inapplicable even when the model did not say
            # "missing". Keep the explicit applicability explanation.
            if "筹码" in clause and not re.search(r"不适用|不支持", clause):
                return ""
            for claim in re.finditer(r"套牢盘|主力成本|(?:真实)?(?:投资者)?持仓成本", clause):
                # Preserve disclaimers such as '不是主力成本' / '不代表真实
                # 投资者持仓成本', but not an affirmative ownership inference.
                prefix = re.split(r"但|然而|而是|却", clause[:claim.start()])[-1]
                if not re.search(
                    r"不是|并非|不代表|不等于|(?:不能|不应|无法)(?:据此)?(?:直接)?"
                    r"(?:推导|推断|确定|判断|视为|理解为|当作|作为|代表|等同)", prefix,
                ):
                    return ""
            return clause

        def clean(value, *, checklist=False):
            if isinstance(value, dict):
                return {k: clean(v, checklist=(k == "action_checklist")) for k, v in value.items()}
            if isinstance(value, list):
                if checklist:
                    # A checklist item is one assertion, not independent clauses.
                    # If any clause violates the chip contract, discard the whole
                    # item; keep valid items verbatim, including their numbering.
                    return [item for item in value if not isinstance(item, str) or (item.strip() and clean(item) == item)]
                return [v for item in value if not isinstance(v := clean(item), str) or v.strip()]
            if isinstance(value, str):
                # Keep normal VP price/distribution clauses and all numeric data.
                return re.sub(
                    r"[^，,；;。\n]+[，,；;。]?",
                    clean_clause,
                    value,
                )
            return value

        result.dashboard = clean(dashboard)
        for key in ("analysis_summary", "risk_warning", "guardrail_reason", "technical_analysis",
                    "trend_analysis", "ma_analysis", "volume_analysis", "pattern_analysis", "short_term_outlook",
                    "medium_term_outlook", "key_points", "buy_reason"):
            value = getattr(result, key, None)
            if isinstance(value, str):
                setattr(result, key, clean(value))
    from src.services.us_report_checklist import restore_default_us_checklist
    restore_default_us_checklist(result)
    return result


def report_display_result(result):
    """Prepare a non-mutating currency-aware view of a stored result."""
    result = reconcile_report(deepcopy(result))
    if not is_us_stock_code(getattr(result, "code", "")):
        return result
    price_keys = {"price", "current_price", "close", "prev_close", "open", "high", "low",
                  "change_amount", "ma5", "ma10", "ma20", "support_level", "resistance_level",
                  "poc", "vah", "val", "ideal_buy", "secondary_buy", "stop_loss", "take_profit"}

    def currency(value, key=""):
        if key == "checklist_context":
            return value  # Source facts and mode metadata are not presentation fields.
        if isinstance(value, dict):
            return {k: currency(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [currency(v) for v in value]
        if key in price_keys and value is not None:
            try:
                number = float(value)
                if math.isfinite(number) and not isinstance(value, bool):
                    return f"${number:.2f}"
            except (ValueError, TypeError):
                pass
        if isinstance(value, str):
            value = re.sub(r"(?<![\d.$A-Za-z])\$?(\d+(?:\.\d+)?(?:\s*[-~至]\s*\d+(?:\.\d+)?)?)\s*([万亿]?)元", r"$\1\2", value)
            value = re.sub(r"(MA\d+\s*[(（])([\d.]+)(?=[)）])", r"\1$\2", value)
            if key in price_keys and "$" not in value:
                # A price field can also contain 'wait for MA10' or 'stop at -5%'.
                # Only prefix an initial price, never an indicator period or percent.
                value = re.sub(r"^(\s*)(\d+(?:\.\d+)?)(?![\d.%])", r"\1$\2", value)
            return value
        return value

    for key, value in vars(result).items():
        if key not in {"raw_response", "analysis_context_pack_overview", "market_phase_summary"}:
            setattr(result, key, currency(value, key))
    return result
