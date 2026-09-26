"""#58: yesterday's volume multiple must not populate the realtime ratio."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.analyzer import GeminiAnalyzer, normalize_volume_ratio
from src.config import Config
from src.market_phase_prompt import format_market_phase_prompt_section
from src.notification import NotificationService
from src.services.report_renderer import render
from tests.test_report_renderer import _make_result


def _wrong_dashboard():
    return {"data_perspective": {"volume_analysis": {
        "volume_ratio": 0.77, "volume_status": "缩量",
        "volume_meaning": "量比0.77，追高意愿不足、易回落", "turnover_rate": None,
    }}}


def _context(realtime):
    return {
        "code": "AVGO", "stock_name": "Broadcom", "date": "2026-09-25",
        "today": {"close": 210, "volume": 770000},
        "yesterday": {"close": 200, "volume": 1000000},
        "realtime": realtime, "volume_change_ratio": 0.77,
    }


@pytest.mark.parametrize("realtime", [{}, {"volume_ratio": None}, {"volume_ratio": "N/A"}])
def test_analyze_corrects_model_substitution_after_parsing(realtime):
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._config_override = Config(stock_list=[], gemini_request_delay=0, report_integrity_enabled=False)
    response = json.dumps({
        "sentiment_score": 50, "trend_prediction": "震荡", "operation_advice": "观望",
        "analysis_summary": "等待开盘", "dashboard": _wrong_dashboard(),
    })
    context = _context(realtime)
    with patch.object(analyzer, "get_generation_backend_config_error", return_value=None), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_get_skill_prompt_sections", return_value=(None, None, True)), \
         patch.object(analyzer, "_call_litellm", return_value=(response, "test-model", {})) as call:
        result = analyzer.analyze(context)
    assert result.success, result.error_message
    volume = result.dashboard["data_perspective"]["volume_analysis"]
    assert volume["volume_ratio"] is None
    assert volume["volume_meaning"] == "较前一交易日成交量为 0.77 倍。"
    assert "追高意愿不足" not in volume["volume_meaning"]
    prompt = call.call_args.args[0]
    assert "较前一交易日成交量倍数（volume_change_ratio，非量比）：0.77倍" in prompt
    assert "缺失时填 null" in prompt
    assert context["volume_change_ratio"] == 0.77  # No recomputation or formula change.


@pytest.mark.parametrize("quote_ratio", [0, 0.77, 1.25])
def test_genuine_realtime_ratio_is_preserved_even_when_model_confuses_fields(quote_ratio):
    result = _make_result(code="AVGO", dashboard=_wrong_dashboard())
    normalize_volume_ratio(result, SimpleNamespace(volume_ratio=quote_ratio))
    assert result.dashboard["data_perspective"]["volume_analysis"]["volume_ratio"] == quote_ratio
    assert result.market_snapshot["volume_ratio"] == quote_ratio


@pytest.mark.parametrize("use_template", [False, True])
@pytest.mark.parametrize("quote_ratio,expected", [(None, "暂无数据"), ("N/A", "暂无数据"), (1.25, "1.25"), (0, "0")])
def test_both_renderers_prefer_snapshot_over_wrong_model_field(use_template, quote_ratio, expected):
    # Also exercises rerendering a historical result whose stored model output is wrong.
    result = _make_result(code="AVGO", name="Broadcom", dashboard=_wrong_dashboard())
    result.market_snapshot = {"price": 210, "volume_ratio": quote_ratio}
    before = deepcopy(result.dashboard)
    config = Config(stock_list=[], report_renderer_enabled=use_template)
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        output = render("markdown", [result]) if use_template else NotificationService().generate_dashboard_report([result])
    assert output is not None
    assert f"量比 {expected}" in output
    assert "| $210.00 |" in output
    assert "量比 0.77" not in output
    assert "0.77" not in output
    if quote_ratio in (None, "N/A"):
        assert "追高意愿不足" not in output
    assert result.dashboard == before  # Rendering does not mutate historical storage.


def test_agent_phase_prompt_distinguishes_the_two_fields():
    for language in ("zh", "en"):
        prompt = format_market_phase_prompt_section(
            {"market": "us", "phase": "non_trading"}, report_language=language,
        )
        assert "volume_analysis.volume_ratio" in prompt
        assert "volume_change_ratio" in prompt
        assert "null" in prompt
