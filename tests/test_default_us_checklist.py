"""Default US reports have six dimensions; explicit skills retain their contract."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.notification import NotificationService
from src.services.report_validation import reconcile_report, report_display_result
from src.services.us_report_checklist import NO_PROFILE, TITLES


RAW = json.loads((Path(__file__).parent / "fixtures/avgo_checklist_model_response.json").read_text(encoding="utf-8"))
PROFILE = RAW["dashboard"]["data_perspective"]["volume_profile"]
CORRECT_FIFTH = "✅ 检查项5：成交量价格结构（Volume Profile）：价格位于价值区内、POC下方，POC $364.86 为历史最大成交密集价"


def _analyze(*, profile, legacy=True, correct=False, code="AVGO"):
    payload = deepcopy(RAW)
    if correct:
        payload["dashboard"]["battle_plan"]["action_checklist"][4] = CORRECT_FIFTH
    if not legacy:
        payload["dashboard"]["battle_plan"]["action_checklist"][4] = "✅ 检查项5：仓位与止损计划明确"
    context = {
        "code": code, "date": "2026-09-25", "stock_name": "Broadcom",
        "today": {"close": 352.81}, "realtime": {"price": 352.81, "volume_ratio": None, "turnover_rate": None},
        "volume_change_ratio": 0.77, "volume_profile": profile,
        "analysis_context_pack": {"blocks": {"chip": {"status": "not_supported"}}},
    }
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._config_override = Config(stock_list=[], gemini_request_delay=0, report_integrity_enabled=False)
    from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_ZH
    analyzer._skill_instructions_override = ""
    analyzer._default_skill_policy_override = CORE_TRADING_SKILL_POLICY_ZH if legacy else ""
    analyzer._use_legacy_default_prompt_override = legacy
    raw = json.dumps(payload, ensure_ascii=False)
    with patch.object(analyzer, "get_generation_backend_config_error", return_value=None), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_call_litellm", return_value=(raw, "test-model", {})):
        result = analyzer.analyze(context)
    assert result.success, result.error_message
    assert result.raw_response == raw
    return result, payload, analyzer, context


@pytest.mark.parametrize("profile", [PROFILE, None])
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("template", [True, False])
def test_default_us_six_dimensions_end_to_end(profile, correct, template):
    result, original, analyzer, context = _analyze(profile=profile, correct=correct)
    checklist = result.dashboard["battle_plan"]["action_checklist"]
    assert len(checklist) == 6
    for number, title in enumerate(TITLES, 1):
        assert f"检查项{number}：{title}" in checklist[number - 1]
    original_checks = original["dashboard"]["battle_plan"]["action_checklist"]
    assert checklist[:4] + checklist[5:] == original_checks[:4] + original_checks[5:]
    if profile:
        assert "价格位于价值区内、POC下方" in checklist[4]
        if correct:
            assert checklist[4] == CORRECT_FIFTH
        else:
            for value in ("$364.86", "$371.56", "$348.60", "$352.81", "$352.42", "$369.65"):
                assert value in checklist[4]
    else:
        assert NO_PROFILE in checklist[4]
        assert "volume_profile" not in result.dashboard["data_perspective"]  # Reject model-invented VP.
    for word in ("筹码", "套牢盘", "主力成本", "持仓成本", "获利比例"):
        assert word not in checklist[4]

    system = analyzer._get_analysis_system_prompt("zh", "AVGO")
    user = analyzer._format_prompt(context, "Broadcom")
    assert "检查项5：筹码健康" not in system
    assert "- ✅ 筹码集中健康" not in system
    assert "### 3. 效率优先（筹码结构）" not in system
    assert "筹码结构是否健康？" not in user
    assert '"profit_ratio":' not in system
    for number, title in enumerate(TITLES, 1):
        assert f"检查项{number}：{title}" in system
    assert NO_PROFILE in user
    before = deepcopy(result.to_dict())
    config = Config(stock_list=[], report_renderer_enabled=template)
    with patch("src.notification.get_config", return_value=config), \
         patch("src.services.report_renderer.get_config", return_value=config):
        output = NotificationService().generate_dashboard_report([result])
    assert output.count("量比 暂无数据") == 1
    assert "换手率：暂无数据" in output
    assert "较前一交易日成交量为 0.77 倍" in output
    assert "$351.06" in output
    assert "检查项5：成交量价格结构（Volume Profile）" in output
    if profile:
        assert "高成交量密集区" in output and "| $364.86 | $371.56 | $348.60 |" in output
    assert result.to_dict() == before
    assert report_display_result(report_display_result(result)).dashboard == report_display_result(result).dashboard


def test_explicit_skill_fifth_and_cn_prompt_are_unchanged():
    result, payload, analyzer, context = _analyze(profile=PROFILE, legacy=False)
    assert result.dashboard["battle_plan"]["action_checklist"] == payload["dashboard"]["battle_plan"]["action_checklist"]
    assert "checklist_context" not in result.market_snapshot
    assert "检查项5：仓位与止损计划明确" in analyzer._get_analysis_system_prompt("zh", "AVGO")
    assert "默认美股趋势报告检查清单" not in analyzer._format_prompt(context, "Broadcom")
    analyzer._use_legacy_default_prompt_override = True
    cn_prompt = analyzer._get_analysis_system_prompt("zh", "600519")
    assert "检查项5：筹码健康" in cn_prompt
    assert "- ✅ 筹码集中健康" in cn_prompt
    assert "### 3. 效率优先（筹码结构）" in cn_prompt
    assert "默认美股趋势报告检查清单" not in cn_prompt


@pytest.mark.parametrize("wrong", [None, "筹码健康", "筹码缺失", "套牢盘", "主力成本", "真实持仓成本", "获利比例"])
def test_missing_or_wrong_fifth_restored_from_persisted_source_facts(wrong):
    result, _, _, _ = _analyze(profile=PROFILE, correct=True)
    checks = result.dashboard["battle_plan"]["action_checklist"]
    if wrong is None:
        del checks[4]
    else:
        checks[4] = f"✅ 检查项5：成交量价格结构（Volume Profile）：{wrong}，任意后半句）"
    # An incorrect model/dashboard field cannot become the fallback's data source.
    result.dashboard["data_perspective"]["volume_profile"]["poc"] = 999
    reconcile_report(result)
    checks = result.dashboard["battle_plan"]["action_checklist"]
    assert len(checks) == 6 and "$364.86" in checks[4]
    assert "999" not in checks[4] and "任意后半句" not in checks[4]


def test_history_roundtrip_and_incomplete_profile_do_not_invent_an_evaluation():
    result, _, _, _ = _analyze(profile={"poc": 364.86}, correct=True)
    result.market_snapshot = json.loads(json.dumps(result.market_snapshot))
    result.dashboard = json.loads(json.dumps(result.dashboard))
    reconcile_report(result)
    assert len(result.dashboard["battle_plan"]["action_checklist"]) == 6
    assert NO_PROFILE in result.dashboard["battle_plan"]["action_checklist"][4]
    assert "✅" not in result.dashboard["battle_plan"]["action_checklist"][4]


@pytest.mark.parametrize("legacy", [True, False])
def test_agent_prompt_uses_same_default_contract_and_prefetched_profile(legacy):
    from src.agent.executor import AgentExecutor
    from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_ZH
    executor = AgentExecutor(MagicMock(), MagicMock(), default_skill_policy=CORE_TRADING_SKILL_POLICY_ZH,
                             use_legacy_default_prompt=legacy, max_steps=1)
    with patch.object(executor, "_run_loop") as loop:
        executor.run("Analyze AVGO", context={"stock_code": "AVGO", "volume_profile": PROFILE})
    messages = loop.call_args.args[0]
    if legacy:
        assert "检查项5：成交量价格结构（Volume Profile）" in messages[0]["content"]
        assert "筹码集中健康" not in messages[0]["content"]
        assert "364.863" in messages[1]["content"]
    else:
        assert "默认美股趋势报告检查清单" not in messages[0]["content"]


@pytest.mark.parametrize("profile", [PROFILE, None])
@pytest.mark.parametrize("legacy", [True, False])
def test_agent_pipeline_persists_mode_and_six_dimensions(profile, legacy):
    from tests.test_pipeline_market_phase_context import _make_pipeline
    from src.agent.executor import AgentResult
    from src.enums import ReportType
    pipeline = _make_pipeline(agent_mode=True)
    pipeline._ensure_agent_history = MagicMock()
    executor = MagicMock()
    executor.use_legacy_default_prompt = legacy
    payload = deepcopy(RAW)
    if not legacy:
        payload["dashboard"]["battle_plan"]["action_checklist"][4] = "✅ 检查项5：仓位与止损计划明确"
    executor.run.return_value = AgentResult(success=True, content=json.dumps(payload), dashboard=payload, provider="test")
    with patch("src.agent.factory.build_agent_executor", return_value=executor):
        result = pipeline._analyze_with_agent("AVGO", ReportType.SIMPLE, "us-checklist", "Broadcom", None, None,
                                              {"market": "us"}, None, profile)
    assert result is not None
    saved = pipeline.db.save_analysis_history.call_args.kwargs["result"]
    checks = saved.dashboard["battle_plan"]["action_checklist"]
    assert len(checks) == 6
    if legacy:
        assert "成交量价格结构（Volume Profile）" in checks[4]
        assert saved.market_snapshot["checklist_context"]["volume_profile"] == profile
        assert (NO_PROFILE in checks[4]) == (profile is None)
    else:
        assert "仓位与止损计划明确" in checks[4]
        assert "checklist_context" not in saved.market_snapshot
