"""US session -> technical quality -> report rendering regression tests."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.core.pipeline import StockAnalysisPipeline
from src.core.trading_calendar import build_market_phase_context
from src.market_phase_prompt import format_market_phase_prompt_section
from src.notification import NotificationService
from src.services.analysis_context_builder import AnalysisContextBuilder
from src.services.report_renderer import render
from src.stock_analyzer import StockTrendAnalyzer
from tests.test_analysis_context_builder import _artifacts
from tests.test_report_renderer import _make_result
from tests.test_us_report_data_quality import _fetch_quote


def _pipeline():
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(enable_realtime_technical_indicators=True, report_language="zh")
    pipeline.search_service = SimpleNamespace(news_window_days=3)
    pipeline.fetcher_manager = SimpleNamespace(build_failed_fundamental_context=lambda *args, **kwargs: {})
    return pipeline


def _daily(last_date):
    dates = pd.bdate_range(end=last_date, periods=30)
    return pd.DataFrame({
        "date": dates.date, "open": 200.0, "high": 212.0, "low": 198.0,
        "close": 210.0, "volume": 1000000, "amount": 0.0, "pct_chg": 0.0,
    })


@pytest.mark.parametrize("instant,expected_phase,last_date,overlay", [
    ("2026-09-26T12:00:00+08:00", "non_trading", "2026-09-25", False),
    ("2026-09-28T09:00:00+08:00", "non_trading", "2026-09-25", False),
    ("2026-09-07T22:00:00+08:00", "non_trading", "2026-09-04", False),
    ("2026-09-28T20:00:00+08:00", "premarket", "2026-09-25", False),
    ("2026-09-28T22:00:00+08:00", "intraday", "2026-09-25", True),
    # Hong Kong Saturday, but New York is still trading on Friday.
    ("2026-09-26T02:00:00+08:00", "intraday", "2026-09-24", True),
    ("2026-09-29T05:00:00+08:00", "postmarket", "2026-09-28", False),
    # Winter time and the early close after Thanksgiving use the calendar too.
    ("2026-12-07T23:00:00+08:00", "intraday", "2026-12-04", True),
    ("2026-11-28T03:00:00+08:00", "postmarket", "2026-11-27", False),
])
def test_us_session_controls_both_bar_generation_and_report_quality(
    instant, expected_phase, last_date, overlay,
):
    phase = build_market_phase_context(market="us", current_time=datetime.fromisoformat(instant)).to_dict()
    assert phase["phase"] == expected_phase
    pipeline = _pipeline()
    quote = _fetch_quote("AVGO")
    original = _daily(last_date)
    market_time = datetime.fromisoformat(phase["market_local_time"])
    with patch("src.core.pipeline.get_market_now", return_value=market_time):
        daily = pipeline._augment_historical_with_realtime(
            original, quote, "AVGO", market="us", market_phase_context=phase,
        )
        trend = StockTrendAnalyzer().analyze(daily, "AVGO")
        base = {"code": "AVGO", "date": last_date,
                "today": original.iloc[-1].to_dict(), "yesterday": original.iloc[-2].to_dict()}
        enhanced = pipeline._enhance_context(base, quote, None, trend, market_phase_context=phase)
    pack = AnalysisContextBuilder.build(_artifacts(
        code="AVGO", market="us", phase=phase, chip_data=None,
        base_context=base, enhanced_context=enhanced, realtime_quote=quote, trend_result=trend,
    ))
    if overlay:
        assert len(daily) == len(original) + 1
        assert enhanced["today"]["is_estimated"] is True
        assert enhanced["today"]["is_partial_bar"] is True
        assert enhanced["today"]["date"] == market_time.date().isoformat()
        assert "technical: partial" in pack.data_quality.limitations
    else:
        pd.testing.assert_frame_equal(daily, original)
        assert enhanced["today"] == base["today"]
        assert enhanced["date"] == last_date
        assert "is_estimated" not in enhanced["today"]
        assert pack.blocks["technical"].status == "available"
        assert "technical: partial" not in pack.data_quality.limitations


@pytest.mark.parametrize("marker", [
    {"is_partial_bar": True}, {"is_estimated": True}, {"estimated_fields": ["close"]},
])
def test_weekend_does_not_clear_genuinely_partial_input(marker):
    phase = build_market_phase_context(
        market="us", current_time=datetime.fromisoformat("2026-09-26T12:00:00+08:00"),
    ).to_dict()
    quote = _fetch_quote("AVGO")
    trend = StockTrendAnalyzer().analyze(_daily("2026-09-25"), "AVGO")
    base = {"code": "AVGO", "today": {"close": 210.0, **marker}}
    enhanced = _pipeline()._enhance_context(base, quote, None, trend, market_phase_context=phase)
    pack = AnalysisContextBuilder.build(_artifacts(
        market="us", phase=phase, enhanced_context=enhanced, trend_result=trend,
    ))
    assert pack.blocks["technical"].status == "partial"


def test_requested_intraday_phase_cannot_create_weekend_bar():
    phase = build_market_phase_context(
        market="us", current_time=datetime.fromisoformat("2026-09-26T12:00:00+08:00"),
        analysis_phase="intraday",
    ).to_dict()
    assert phase["phase"] == "intraday"
    assert not _pipeline()._allow_realtime_technical_overlay("us", phase)


@pytest.mark.parametrize("turnover,expected", [
    (None, "暂无数据"), ("数据缺失，无法判断", "暂无数据"), ("N/A", "暂无数据"),
    ("", "暂无数据"), (float("nan"), "暂无数据"), (0, "0%"), (1.25, "1.25%"), ("1.25%", "1.25%"),
])
@pytest.mark.parametrize("use_template", [False, True])
def test_turnover_rendering_in_both_report_paths(turnover, expected, use_template):
    result = _make_result(code="AVGO", name="Broadcom", dashboard={
        "data_perspective": {"volume_analysis": {"volume_ratio": 0.77, "turnover_rate": turnover}},
    })
    result.market_snapshot = {"price": 210, "volume_ratio": 0.77, "turnover_rate": turnover}
    config = Config(stock_list=[], report_renderer_enabled=use_template)
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        output = (render("markdown", [result]) if use_template else
                  NotificationService().generate_dashboard_report([result]))
    assert output is not None
    assert f"换手率：{expected}" in output
    assert f"| 210 | 0.77 | {expected} |" in output
    assert "无法判断%" not in output
    assert "暂无数据%" not in output


@pytest.mark.parametrize("turnover,expected", [(None, "暂无数据"), (1.25, "1.25%")])
def test_turnover_prompt_and_snapshot(turnover, expected):
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    context = {"code": "AVGO", "today": {"close": 210}, "realtime": {"turnover_rate": turnover}}
    with patch.object(analyzer, "_get_skill_prompt_sections", return_value=("", "", True)):
        prompt = analyzer._format_prompt(context, "Broadcom")
    assert f"| **换手率** | **{expected}**" in prompt
    assert analyzer._build_market_snapshot(context)["turnover_rate"] == expected


def test_unconfirmed_scalar_cannot_be_described_as_current_intraday_contraction():
    quote = _fetch_quote("AVGO")
    quote.volume_ratio = 0.77
    context = _pipeline()._enhance_context({"code": "AVGO"}, quote, None, None)
    assert context["realtime"]["volume_ratio"] == 0.77
    assert "统计周期未确认" in context["realtime"]["volume_ratio_desc"]
    for phase in ("non_trading", "intraday", "postmarket"):
        prompt = format_market_phase_prompt_section({"market": "us", "phase": phase})
        assert "来源、分子和分母的日期与交易时段" in prompt
        assert "不得仅凭该数值推导" in prompt
