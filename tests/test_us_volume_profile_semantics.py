"""Replay the user-supplied Actions report dated 2026-09-26 07:16:16.

The fixture is the complete rendered report, not an original model JSON.
Reconstruct its relevant model fields verbatim to exercise parsing, validation
and both report renderers without calling a live model or sending notifications.
"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.notification import NotificationService
from src.services.report_validation import reconcile_report


REPORT = (Path(__file__).parent / "fixtures/avgo_report_volume_profile.txt").read_text(encoding="utf-8-sig")
BAD_CHECK = "⚠️ 检查项5：筹码健康（价格处于POC下方，上方存在套牢盘压制）"
POSITION = "价格位于价值区内、POC下方"
DISCLAIMER = next(line.removeprefix("注：") for line in REPORT.splitlines() if line.startswith("注：Volume Profile"))


def _response(correct):
    checklist = [line[2:] for line in REPORT.split("✅ 检查清单", 1)[1].split("🎯 信号归因分析", 1)[0].splitlines() if line.startswith("- ")]
    assert BAD_CHECK in checklist
    if correct:
        checklist[checklist.index(BAD_CHECK)] = "✅ Volume Profile：" + POSITION
    profile_section = REPORT.split("Volume Profile（历史成交量价格分布）", 1)[1]
    values = next(line for line in profile_section.splitlines() if line.startswith("| $"))
    prices = [float(value.strip().removeprefix("$")) for value in values.strip("|").split("|")]
    profile = dict(zip(("poc", "vah", "val", "current_price"), prices))
    profile.update(position="inside_value_area_below_poc", lookback=30, analysis=POSITION,
                   explanation=DISCLAIMER, high_volume_nodes=[
                       {"low": 352.42, "high": 369.65, "volume_share": 0.599},
                       {"low": 348.60, "high": 350.51, "volume_share": 0.044},
                   ])
    return {
        "sentiment_score": 56, "operation_advice": "观望", "trend_prediction": "震荡",
        "analysis_summary": "等待趋势确认", "confidence_level": "中",
        "dashboard": {
            "data_perspective": {"volume_profile": profile, "volume_analysis": {
                "volume_ratio": None, "turnover_rate": None,
                "volume_meaning": "较前一交易日成交量为 0.77 倍。",
            }},
            "battle_plan": {"action_checklist": checklist,
                            "sniper_points": {"ideal_buy": "$351.06（MA10支撑位）"}},
        },
    }


@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("template", [True, False])
def test_real_report_model_response_to_report(correct, template):
    response = _response(correct)
    raw = json.dumps(response, ensure_ascii=False)
    context = {
        "code": "AVGO", "stock_name": "Broadcom", "date": "2026-09-25",
        "today": {"close": 352.81}, "realtime": {
            "price": 352.81, "volume_ratio": None, "turnover_rate": None, "source": "yfinance",
        },
        "volume_change_ratio": 0.77,
        "volume_profile": response["dashboard"]["data_perspective"]["volume_profile"],
        "analysis_context_pack": {"blocks": {"chip": {"status": "not_supported"}}},
    }
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._config_override = Config(stock_list=[], gemini_request_delay=0, report_integrity_enabled=False)
    with patch.object(analyzer, "get_generation_backend_config_error", return_value=None), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_get_skill_prompt_sections", return_value=(None, None, True)), \
         patch.object(analyzer, "_call_litellm", return_value=(raw, "test-model", {})):
        result = analyzer.analyze(context)
    assert result.success, result.error_message
    assert result.raw_response == raw
    profile = response["dashboard"]["data_perspective"]["volume_profile"]
    assert result.dashboard["data_perspective"]["volume_profile"] == profile
    checklist = result.dashboard["battle_plan"]["action_checklist"]
    if correct:
        assert checklist == response["dashboard"]["battle_plan"]["action_checklist"]
    else:
        assert checklist == [item for item in response["dashboard"]["battle_plan"]["action_checklist"] if item != BAD_CHECK]
    assert all("筹码健康" not in item and "套牢盘" not in item for item in checklist)
    before = deepcopy(result.to_dict())
    config = Config(stock_list=[], report_renderer_enabled=template)
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        output = NotificationService().generate_dashboard_report([result])
    assert "筹码健康" not in output and "套牢盘" not in output
    assert POSITION in output and DISCLAIMER in output
    for price in ("$364.86", "$371.56", "$348.60", "$352.81", "$352.42", "$369.65", "$351.06"):
        assert price in output
    assert "高成交量密集区" in output
    assert output.count("量比 暂无数据") == 1
    assert "换手率：暂无数据" in output
    assert "较前一交易日成交量为 0.77 倍" in output
    assert "不适用（美股不适用 A 股口径筹码分布）" in output
    assert result.to_dict() == before


@pytest.mark.parametrize("wrong", ["筹码健康", "筹码缺失", "上方存在套牢盘压制", "POC代表主力成本", "POC是真实投资者持仓成本", "价格不能突破套牢盘"])
def test_false_claims_removed_without_losing_profile_facts(wrong):
    payload = _response(True)
    result = GeminiAnalyzer.__new__(GeminiAnalyzer)._parse_response(json.dumps(payload), "AVGO", "Broadcom")
    result.market_snapshot = {"chip_status": "not_supported"}
    profile = result.dashboard["data_perspective"]["volume_profile"]
    profile["analysis"] = POSITION + "；" + wrong + "。"
    result.dashboard["battle_plan"]["action_checklist"].append(wrong)
    result.technical_analysis = wrong
    reconcile_report(result)
    assert wrong not in str(result.dashboard["battle_plan"]["action_checklist"])
    assert wrong not in result.technical_analysis
    assert result.dashboard["data_perspective"]["volume_profile"]["analysis"] == POSITION + "；"
    assert result.dashboard["data_perspective"]["volume_profile"]["explanation"] == DISCLAIMER
    before = deepcopy(result.to_dict())
    reconcile_report(result)
    assert result.to_dict() == before


@pytest.mark.parametrize("code,status", [("600519", "not_supported"), ("AVGO", "available"), ("AVGO", None)])
def test_cleanup_requires_us_and_explicit_unsupported_chip(code, status):
    result = GeminiAnalyzer.__new__(GeminiAnalyzer)._parse_response(json.dumps(_response(False)), code, "test")
    result.market_snapshot = {"chip_status": status}
    reconcile_report(result)
    assert BAD_CHECK in result.dashboard["battle_plan"]["action_checklist"]


def test_overview_status_and_correct_disclaimers_preserve_profile():
    result = GeminiAnalyzer.__new__(GeminiAnalyzer)._parse_response(json.dumps(_response(True)), "AVGO", "Broadcom")
    result.analysis_context_pack_overview = {"blocks": [{"key": "chip", "status": "not_supported"}]}
    profile = result.dashboard["data_perspective"]["volume_profile"]
    profile["explanation"] = "Volume Profile不能据此推导为套牢盘、主力成本或真实投资者持仓成本。"
    expected = deepcopy(profile)
    result.dashboard["battle_plan"]["action_checklist"].append(BAD_CHECK)
    reconcile_report(result)
    assert result.dashboard["data_perspective"]["volume_profile"] == expected
    assert BAD_CHECK not in result.dashboard["battle_plan"]["action_checklist"]
