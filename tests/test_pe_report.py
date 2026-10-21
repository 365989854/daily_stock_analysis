"""Verified PE source -> fundamental context -> model input -> Markdown (offline)."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from data_provider.base import DataFetcherManager
from src.config import Config
from src.notification import NotificationService
from src.services.report_validation import report_display_result
from tests.test_default_us_checklist import _analyze, PROFILE
from tests.test_yfinance_fundamental_adapter import _build_mock_ticker


@pytest.mark.parametrize("code,ttm,forward", [("MU", 12.06, 5.08), ("AVGO", 45.58, 28.4)])
@pytest.mark.parametrize("availability", ["both", "ttm", "forward", "none", "invalid"])
@pytest.mark.parametrize("template", [True, False])
def test_pe_source_to_report(code, ttm, forward, availability, template):
    # These are mock values, not assertions about live market prices.
    info = {"trailingPE": ttm, "forwardPE": forward, "peRatio": 999}
    if availability not in ("both", "ttm"):
        info.pop("trailingPE")
    if availability not in ("both", "forward"):
        info.pop("forwardPE")
    if availability == "invalid":
        info.update(trailingPE=float("nan"), forwardPE="not-a-number")
    ticker = _build_mock_ticker(info)
    config = Config(stock_list=[], fundamental_cache_ttl_seconds=0,
                    fundamental_stage_timeout_seconds=10, fundamental_fetch_timeout_seconds=5,
                    report_renderer_enabled=template)
    manager = DataFetcherManager(fetchers=[])
    with patch("yfinance.Ticker", return_value=ticker), \
         patch("src.config.get_config", return_value=config), \
         patch.object(manager, "get_realtime_quote", return_value=SimpleNamespace(pe_ratio=777)):
        fundamental = manager.get_fundamental_context(code)
    values = fundamental["valuation"]["data"]
    assert values["pe_ratio"] == 777  # Existing consumer contract remains intact.
    assert values["pe_ttm"] == (ttm if availability in ("both", "ttm") else None)
    assert values["pe_forward"] == (forward if availability in ("both", "forward") else None)
    ticker.get_info.assert_called_once()
    result, original, analyzer, context = _analyze(
        profile=PROFILE, code=code, source={"fundamental_context": fundamental})
    before = deepcopy(result.to_dict())
    prompt = analyzer._format_prompt(context, code)
    assert "TTM PE:" in prompt and "Forward PE:" in prompt
    assert "动态" not in prompt
    assert result.sentiment_score == original["sentiment_score"]
    assert result.decision_type == original["decision_type"]
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        markdown = NotificationService().generate_dashboard_report([result])
    assert "动态PE" not in markdown and "动态 PE" not in markdown
    assert "PE估值参考" in markdown and "PE估值合理" not in markdown
    assert (f"TTM PE：{ttm:.2f}倍" in markdown) == (availability in ("both", "ttm"))
    assert (f"Forward PE：{forward:.2f}倍" in markdown) == (availability in ("both", "forward"))
    assert ("PE数据不足" in markdown) == (availability in ("none", "invalid"))
    assert result.to_dict() == before  # Rendering never updates the saved result.
    assert report_display_result(report_display_result(result)).to_dict() == report_display_result(result).to_dict()


def test_prompt_rules_cover_single_and_multi_agent():
    from src.agent.executor import AgentExecutor
    from src.agent.agents.intel_agent import IntelAgent
    from src.agent.protocols import AgentContext
    from src.market_context import PE_GUIDELINES
    executor = AgentExecutor(MagicMock(), MagicMock(), max_steps=1)
    with patch.object(executor, "_run_loop") as loop:
        executor.run("Analyze MU", context={"stock_code": "MU"})
    assert PE_GUIDELINES in loop.call_args.args[0][0]["content"]
    agent = IntelAgent(MagicMock(), MagicMock())
    with patch.object(agent, "_build_memory_context", return_value=""):
        messages = agent._build_messages(AgentContext(stock_code="MU"))
    assert PE_GUIDELINES in messages[0]["content"]


def test_historical_report_without_pe_snapshot_is_unchanged():
    result, _, _, _ = _analyze(profile=PROFILE)
    assert "pe_valuation" not in result.market_snapshot
    before = deepcopy(result.to_dict())
    assert report_display_result(result).dashboard == report_display_result(result).dashboard
    assert result.to_dict() == before


@pytest.mark.parametrize("code", ["MU", "AVGO"])
@pytest.mark.parametrize("legacy", [True, False])
def test_agent_pipeline_binds_actual_pe_context(code, legacy):
    import json
    from src.agent.executor import AgentResult
    from src.enums import ReportType
    from tests.test_pipeline_market_phase_context import _make_pipeline
    from tests.test_default_us_checklist import RAW
    pipeline = _make_pipeline(agent_mode=True)
    pipeline._ensure_agent_history = MagicMock()
    executor = MagicMock()
    executor.use_legacy_default_prompt = legacy
    payload = deepcopy(RAW)
    executor.run.return_value = AgentResult(success=True, content=json.dumps(payload),
                                           dashboard=payload, provider="test")
    fundamental = {"market": "us", "valuation": {"data": {"pe_ttm": 19.2, "pe_forward": 11.3}}}
    with patch("src.agent.factory.build_agent_executor", return_value=executor):
        result = pipeline._analyze_with_agent(code, ReportType.SIMPLE, "pe-test", code,
                                             None, None, fundamental, None, PROFILE)
    assert result is not None
    saved = pipeline.db.save_analysis_history.call_args.kwargs["result"]
    assert saved.market_snapshot["pe_valuation"] == fundamental["valuation"]["data"]
    assert "TTM PE：19.20倍；Forward PE：11.30倍" in saved.dashboard["battle_plan"]["action_checklist"][-1]
    assert saved.fundamental_context == fundamental


@pytest.mark.parametrize("ttm,forward,expected", [
    (float("inf"), True, "PE数据不足"),
    (0, -3, "TTM PE：0.00倍；Forward PE：-3.00倍"),
])
def test_pe_edge_values_do_not_create_valuation_judgments(ttm, forward, expected):
    result, _, _, _ = _analyze(profile=PROFILE, source={
        "fundamental_context": {"valuation": {"data": {"pe_ttm": ttm, "pe_forward": forward}}}})
    assert expected in result.dashboard["battle_plan"]["action_checklist"][-1]
