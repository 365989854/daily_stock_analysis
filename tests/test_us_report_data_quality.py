"""US provider -> routing -> report quality regressions (no live requests)."""

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_provider.base import DataFetcherManager
from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.analysis_context_pack_prompt import format_analysis_context_pack_prompt_section
from src.services.analysis_context_builder import AnalysisContextBuilder
from tests.test_analysis_context_builder import _artifacts


def _fetch_quote(code, *, history=False, missing_volume=False):
    ticker = Mock()
    ticker.info = {"currency": "USD"}
    ticker.fast_info = None if history else SimpleNamespace(
        last_price=210.0, previous_close=200.0, open=202.0,
        day_high=212.0, day_low=198.0,
        last_volume=None if missing_volume else 1000000, market_cap=None,
    )
    ticker.history.return_value = pd.DataFrame({
        "Close": [200.0, 210.0], "Open": [198.0, 202.0],
        "High": [202.0, 212.0], "Low": [195.0, 198.0],
        "Volume": [900000, 1000000],
    })
    with patch("yfinance.Ticker", return_value=ticker):
        return YfinanceFetcher().get_realtime_quote(code)


def _pack(code, quote, today=None):
    return AnalysisContextBuilder.build(_artifacts(
        code=code, market="us", stock_name=code, chip_data=None,
        phase={"market": "us", "phase": "postmarket"},
        realtime_quote=quote,
        enhanced_context={"today": today or {
            "close": 210.0, "data_source": "realtime:yfinance",
            "is_partial_bar": False, "is_estimated": False,
        }},
    ))


@pytest.mark.parametrize("code", ["AAPL", "SPX"])
@pytest.mark.parametrize("history", [False, True])
def test_yfinance_success_and_optional_gaps_do_not_degrade_report(code, history):
    quote = _fetch_quote(code, history=history)
    assert quote.source == RealtimeSource.YFINANCE
    assert quote.fallback_from is None
    assert quote.data_quality == "ok"
    assert {"amount", "pe_ratio", "pb_ratio"} <= set(quote.missing_fields)
    assert quote.volume_ratio is None
    assert quote.turnover_rate is None
    pack = _pack(code, quote)
    for key in ("quote", "daily_bars", "technical"):
        assert pack.blocks[key].status == "available"
    assert pack.blocks["chip"].status == "not_supported"
    assert pack.blocks["chip"].items["chip_distribution"].missing_reason == "chip_not_applicable"
    assert pack.data_quality.overall_score == 100
    assert pack.data_quality.limitations == []
    assert "不适用" in format_analysis_context_pack_prompt_section(pack)
    assert "do not lower confidence" in format_analysis_context_pack_prompt_section(pack, report_language="en")


@pytest.mark.parametrize("code", ["AAPL", "SPX"])
def test_missing_core_volume_remains_partial_at_provider(code):
    assert _fetch_quote(code, missing_volume=True).data_quality == "partial"


@pytest.mark.parametrize("code", ["AAPL", "SPX"])
@pytest.mark.parametrize("marker", [
    {"is_partial_bar": True}, {"is_estimated": True},
    {"estimated_fields": ["close"]},
])
def test_unfinished_or_estimated_us_bar_still_degrades_technical(code, marker):
    pack = _pack(code, _fetch_quote(code), {"close": 210.0, **marker})
    assert pack.blocks["technical"].status == "partial"
    assert pack.blocks["technical"].items["intraday_overlay"].status == "estimated"
    assert "technical: partial" in pack.data_quality.limitations
    assert "confidence_level" in format_analysis_context_pack_prompt_section(pack)


@pytest.mark.parametrize("code,prefer_lb,primary_failed", [
    ("AAPL", False, False), ("AAPL", True, False),
    ("AAPL", True, True), ("AAPL", False, True),
    ("SPX", True, False), ("SPX", True, True),
])
def test_manager_only_marks_actual_primary_failure_as_fallback(code, prefer_lb, primary_failed):
    yf_quote = _fetch_quote(code)
    lb_quote = UnifiedRealtimeQuote(
        code=code, name=code, price=210.0, source=RealtimeSource.LONGBRIDGE,
        pe_ratio=20.0,
    )
    primary = "LongbridgeFetcher" if prefer_lb and code == "AAPL" else "YfinanceFetcher"
    calls = []

    def fetch(stock_code, source, **kwargs):
        calls.append(source)
        if source == primary and primary_failed:
            return None
        return {"YfinanceFetcher": yf_quote, "LongbridgeFetcher": lb_quote}.get(source)

    manager = DataFetcherManager.__new__(DataFetcherManager)
    config = SimpleNamespace(enable_realtime_quote=True, realtime_cache_ttl=600)
    with patch("src.config.get_config", return_value=config), \
         patch.object(manager, "_longbridge_preferred", return_value=prefer_lb), \
         patch.object(manager, "_try_fetcher_quote", side_effect=fetch):
        quote = manager.get_realtime_quote(code)
    if code == "SPX":
        assert calls == ["YfinanceFetcher"]
        if primary_failed:
            assert quote is None
            assert _pack(code, quote).blocks["quote"].status == "missing"
            return
    expected_fallback = ("longbridge" if prefer_lb else "yfinance") if primary_failed else None
    assert quote.fallback_from == expected_fallback
    pack = _pack(code, quote)
    assert pack.blocks["quote"].status == ("fallback" if primary_failed else "available")
    assert ("quote: fallback" in pack.data_quality.limitations) == primary_failed


def test_yfinance_failure_then_stooq_success_is_real_fallback():
    responses = [
        BytesIO(b"AAPL.US,20260925,220000,202,212,198,210,1000000\n"),
        BytesIO(b"Date,Close\n2026-09-24,200\n2026-09-25,210\n"),
    ]
    with patch("yfinance.Ticker", side_effect=RuntimeError("provider unavailable")), \
         patch("data_provider.yfinance_fetcher.urlopen", side_effect=responses):
        quote = YfinanceFetcher().get_realtime_quote("AAPL")
    assert quote.source == RealtimeSource.STOOQ
    assert quote.fallback_from == "yfinance"
    assert _pack("AAPL", quote).blocks["quote"].status == "fallback"


def test_non_us_quality_semantics_are_unchanged():
    artifacts = _artifacts(chip_data=None, enhanced_context={
        "today": {"close": 210.0, "data_source": "realtime:akshare_em"},
    })
    for market in ("cn", "hk"):
        pack = AnalysisContextBuilder.build(replace(artifacts, market=market))
        assert pack.blocks["technical"].status == "partial"
        assert pack.blocks["chip"].status == "missing"
        assert pack.data_quality.block_scores["chip"] == 35
