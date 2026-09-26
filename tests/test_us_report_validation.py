"""Regression from Actions #59, run 36223276789 (2026-09-26).

The fixture is the model JSON from analysis-reports-59's debug log. Quote
facts below come from the same run's full prompt, not the model response.
"""

from copy import deepcopy
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.notification import NotificationService
from src.services.report_renderer import render
from src.services.report_validation import reconcile_report, report_display_result


def analyze_fixture(correct=False, turnover=None):
    response = json.loads((Path(__file__).parent / "fixtures/us_report_model_response.json").read_text(encoding="utf-8"))
    if correct:
        dp = response["dashboard"]["data_perspective"]
        dp["volume_analysis"].update(volume_ratio=None, turnover_rate=None, volume_status="",
                                     volume_meaning="较前一交易日成交量为 0.77 倍。")
        dp["chip_structure"] = {}
        response["dashboard"]["phase_decision"]["data_limitations"] = ["实时量比数据缺失"]
        response["dashboard"]["battle_plan"]["action_checklist"] = ["✅ 均线数据完整"]
        response = json.loads(json.dumps(response, ensure_ascii=False).replace("351.06元", "$351.06"))
    elif correct is False:
        response["dashboard"]["data_perspective"]["volume_analysis"].update(
            volume_ratio=0.77, turnover_rate=0,
            volume_meaning="实时量比0.77，追高意愿不足、易回落",
        )
    context = {
        "code": "AVGO", "stock_name": "Broadcom", "date": "2026-09-25",
        "today": {"close": 352.81, "open": 352.62, "high": 354.43, "low": 349.43},
        "yesterday": {"close": 350.36}, "volume_change_ratio": 0.77,
        "realtime": {"price": 352.81, "volume_ratio": None, "turnover_rate": turnover, "source": "yfinance"},
        "analysis_context_pack": {"blocks": {"chip": {"status": "not_supported"}}},
    }
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._config_override = Config(stock_list=[], gemini_request_delay=0, report_integrity_enabled=False)
    raw = json.dumps(response, ensure_ascii=False)
    with patch.object(analyzer, "get_generation_backend_config_error", return_value=None), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_get_skill_prompt_sections", return_value=(None, None, True)), \
         patch.object(analyzer, "_call_litellm", return_value=(raw, "test-model", {})):
        result = analyzer.analyze(context)
    assert result.success, result.error_message
    assert result.raw_response == raw
    return result


@pytest.mark.parametrize("correct", [None, True, False])
@pytest.mark.parametrize("template", [True, False])
def test_correct_and_hostile_models_render_source_facts(correct, template):
    result = analyze_fixture(correct)
    before = deepcopy(result.to_dict())
    volume = result.dashboard["data_perspective"]["volume_analysis"]
    assert volume["turnover_rate"] is None
    assert volume["volume_ratio"] is None
    assert volume["volume_meaning"] == "较前一交易日成交量为 0.77 倍。"
    config = Config(stock_list=[], report_renderer_enabled=template)
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        output = NotificationService().generate_dashboard_report([result])
    assert output.count("量比 暂无数据") == 1
    assert "(暂无数据)" not in output
    assert "换手率：暂无数据" in output
    assert "换手率：0%" not in output
    assert "筹码健康" not in output
    assert "筹码分布数据缺失" not in output
    assert "不适用" in output
    assert "较前一交易日成交量为 0.77 倍" in output
    assert "量比 0.77" not in output
    assert "追高意愿" not in output
    assert "$352.81" in output and "$351.06" in output
    assert "351元" not in output and "348元" not in output
    assert result.to_dict() == before


@pytest.mark.parametrize("turnover", [0, 1.25])
def test_real_turnover_zero_and_positive_are_retained(turnover):
    result = analyze_fixture(turnover=turnover)
    assert result.dashboard["data_perspective"]["volume_analysis"]["turnover_rate"] == turnover
    assert f"换手率：{turnover:g}%" in render("markdown", [result])


def test_chip_cleanup_preserves_other_limits_and_is_idempotent():
    result = analyze_fixture()
    result.dashboard["phase_decision"]["data_limitations"] = ["筹码缺失；日线未完成", "筹码缺失，估值偏高"]
    reconcile_report(result)
    assert result.dashboard["phase_decision"]["data_limitations"] == ["日线未完成", "估值偏高"]
    before = deepcopy(result.to_dict())
    reconcile_report(result)
    assert result.to_dict() == before
    assert result.confidence_level == "中"


@pytest.mark.parametrize("value,expected", [
    ("等待 MA10 确认", "等待 MA10 确认"),
    ("回撤5%止损", "回撤5%止损"),
    ("5%", "5%"),
    ("$351.06元", "$351.06"),
    ("351.06（MA10）", "$351.06（MA10）"),
    ("351.06元（MA10）", "$351.06（MA10）"),
])
def test_currency_does_not_confuse_indicator_periods_or_percentages(value, expected):
    result = analyze_fixture()
    result.dashboard["battle_plan"]["sniper_points"]["ideal_buy"] = value
    view = report_display_result(result)
    assert view.dashboard["battle_plan"]["sniper_points"]["ideal_buy"] == expected
    assert report_display_result(view).dashboard == view.dashboard


def test_invalid_volume_claim_does_not_remove_real_price_risk():
    result = analyze_fixture()
    result.risk_warning = "跌破支撑应减仓，量比0.77，追高意愿不足、易回落。"
    reconcile_report(result)
    assert result.risk_warning == "跌破支撑应减仓，"


def test_cn_prices_and_applicable_chip_limits_are_not_rewritten():
    result = analyze_fixture()
    result.code = "600519"
    result.dashboard["phase_decision"]["data_limitations"] = ["筹码缺失"]
    result.dashboard["battle_plan"]["sniper_points"]["ideal_buy"] = "351元"
    view = report_display_result(result)
    assert view.dashboard["battle_plan"]["sniper_points"]["ideal_buy"] == "351元"
    assert view.dashboard["phase_decision"]["data_limitations"] == ["筹码缺失"]


@pytest.mark.parametrize("agent", [False, True])
def test_pipeline_corrects_fields_before_saving_history(agent):
    from tests.test_pipeline_market_phase_context import _make_pipeline
    from src.agent.executor import AgentResult
    from src.enums import ReportType

    pipeline = _make_pipeline(agent_mode=agent)
    pipeline._ensure_agent_history = MagicMock()
    pipeline.fetcher_manager.get_stock_name.return_value = "Broadcom"
    pipeline.db.get_analysis_context.return_value = {
        "code": "AVGO", "stock_name": "Broadcom", "date": "2026-09-25",
        "today": {"date": "2026-09-25", "close": 352.81, "volume": 770},
        "yesterday": {"date": "2026-09-24", "close": 350.36, "volume": 1000},
        "volume_change_ratio": 0.77,
    }
    payload = json.loads((Path(__file__).parent / "fixtures/us_report_model_response.json").read_text(encoding="utf-8"))
    # Normal and Agent paths receive the uncorrected model output.
    result = GeminiAnalyzer.__new__(GeminiAnalyzer)._parse_response(json.dumps(payload), "AVGO", "Broadcom")
    pipeline.analyzer.analyze.return_value = result
    executor = MagicMock()
    executor.run.return_value = AgentResult(success=True, content=json.dumps(payload), dashboard=payload, provider="test")
    with patch("src.agent.factory.build_agent_executor", return_value=executor):
        output = pipeline.analyze_stock("AVGO", ReportType.SIMPLE, "regression-59",
                                        current_time=datetime(2026, 9, 26, 6, 18, tzinfo=timezone.utc))
    assert output is not None
    saved = pipeline.db.save_analysis_history.call_args.kwargs["result"]
    volume = saved.dashboard["data_perspective"]["volume_analysis"]
    assert volume["turnover_rate"] is None and volume["volume_ratio"] is None
    assert "0.77 倍" in volume["volume_meaning"]
    assert saved.dashboard["data_perspective"]["chip_not_applicable"]
    assert "筹码健康" not in str(saved.dashboard["battle_plan"]["action_checklist"])
