"""Actions #63 raw Gemini response: six names without supporting reasons.

Source: https://github.com/365989854/daily_stock_analysis/actions/runs/36238279804
Artifact analysis-reports-63, debug log, commit 08698618. Fixture is unmodified JSON.
"""
from copy import deepcopy
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config
from src.notification import NotificationService
from src.services.report_validation import report_display_result
from src.services.us_report_checklist import TITLES, NO_PROFILE
from tests.test_default_us_checklist import _analyze, PROFILE

RAW = json.loads((Path(__file__).parent / "fixtures/avgo_checklist_names_only_response.json").read_text(encoding="utf-8"))
# Original #63 input facts; model-produced dashboard fields are not evidence.
SOURCE = {
    "today": {"close": 352.81, "ma5": 357.07, "ma10": 351.06, "ma20": 357.54},
    "trend_analysis": {"bias_ma5": -1.19},
    "realtime": {"price": 352.81, "volume_ratio": None, "turnover_rate": None, "pe_ratio": 44.94395},
    "volume_change_ratio": 0.77,
}


@pytest.mark.parametrize("template", [True, False])
def test_actions63_names_only_complete_reasons_and_rendering(template):
    assert RAW["dashboard"]["battle_plan"]["action_checklist"] == [
        f"{status} 检查项{i}：{title}" for i, (status, title) in enumerate(
            zip(("❌", "✅", "⚠️", "⚠️", "✅", "⚠️"), TITLES), 1)]
    result, _, analyzer, _ = _analyze(profile=PROFILE, payload=RAW, source=SOURCE)
    checks = result.dashboard["battle_plan"]["action_checklist"]
    assert len(checks) == 6
    for i, (title, check) in enumerate(zip(TITLES, checks), 1):
        assert f"检查项{i}：{title}（" in check
        assert check.startswith("⚠️")
    for text, check in zip(("MA5 $357.07", "-1.19%", "较前一交易日成交量为 0.77 倍",
                            "依据不足，未作判断", "价格位于价值区内、POC下方", "PE 44.94倍"), checks):
        assert text in check
    assert "未判断估值高低" in checks[5]
    assert "✅" not in checks[4] and "❌" not in checks[4]
    system = analyzer._get_analysis_system_prompt("zh", "AVGO")
    examples = json.loads(re.search(r'"action_checklist":\s*(\[[^\]]*\])', system).group(1))
    for title, example in zip(TITLES, examples):
        assert title + "（" in example
        assert example.startswith("✅/⚠️/❌ 检查项")
    assert all("⚪" not in item and "ℹ️" not in item for item in examples)
    assert "未据此作方向性判断" in checks[4]
    assert "不算判断依据" in system
    config = Config(stock_list=[], report_renderer_enabled=template)
    before = deepcopy(result.to_dict())
    with patch("src.notification.get_config", return_value=config), patch("src.services.report_renderer.get_config", return_value=config):
        output = NotificationService().generate_dashboard_report([result])
    assert output.count("量比 暂无数据") == 1
    assert "换手率：暂无数据" in output and "换手率 0%" not in output
    assert "较前一交易日成交量为 0.77 倍" in output
    assert "$351.06" in output and "351.06元" not in output
    for check in checks:
        assert check in output
    assert result.to_dict() == before
    assert report_display_result(report_display_result(result)).dashboard == report_display_result(result).dashboard


def test_existing_complete_reasons_preserved_verbatim():
    payload = deepcopy(RAW)
    reasons = ("MA5 $357.07，MA10 $351.06，MA20 $357.54，未形成多头排列", "MA5乖离率 -1.19%",
               "较前一交易日成交量为 0.77 倍", "依据不足，未作判断",
               "当前价位于价值区内、POC下方", "PE 44.94倍，缺少比较依据")
    checks = [f"{status} 检查项{i}：{title}（{reason}）"
              for i, (status, title, reason) in enumerate(zip(("❌", "✅", "⚠️", "⚠️", "⚠️", "⚠️"), TITLES, reasons), 1)]
    payload["dashboard"]["battle_plan"]["action_checklist"] = checks
    result, _, _, _ = _analyze(profile=PROFILE, payload=payload, source=SOURCE)
    assert result.dashboard["battle_plan"]["action_checklist"] == checks


@pytest.mark.parametrize("suffix", ["", "（）", "（ ）", ": ()"])
def test_empty_reasons_never_use_model_data_or_invent_evidence(suffix):
    payload = deepcopy(RAW)
    payload["dashboard"]["battle_plan"]["action_checklist"] = [
        f"✅ 检查项{i}：{title}{suffix}" for i, title in enumerate(TITLES, 1)]
    result, _, _, _ = _analyze(profile=None, payload=payload, source={"volume_change_ratio": None})
    checks = result.dashboard["battle_plan"]["action_checklist"]
    for i, check in enumerate(checks):
        assert check.startswith("⚠️")
        assert (NO_PROFILE if i == 4 else "依据不足，未作判断") in check
    assert "45" not in "".join(checks) and "减持" not in "".join(checks)


def test_explicit_skill_bare_names_are_not_repaired():
    result, original, _, _ = _analyze(profile=PROFILE, legacy=False, payload=RAW)
    assert result.dashboard["battle_plan"]["action_checklist"] == original["dashboard"]["battle_plan"]["action_checklist"]


@pytest.mark.parametrize("status", ["⚪", "ℹ️", "🔵", ""])
def test_non_contract_status_preserves_valid_reason(status):
    payload = deepcopy(RAW)
    checks = [f"{status} 检查项{i}：{title}（依据不足，未作判断）"
              for i, title in enumerate(TITLES, 1)]
    checks[4] = f"{status} 检查项5：{TITLES[4]}（当前价位于价值区内、POC下方，未据此作方向性判断）"
    payload["dashboard"]["battle_plan"]["action_checklist"] = checks
    result, _, _, _ = _analyze(profile=PROFILE, payload=payload, source=SOURCE)
    expected = ["⚠️ " + item[item.index("检查项"):] for item in checks]
    assert result.dashboard["battle_plan"]["action_checklist"] == expected
    assert report_display_result(result).dashboard["battle_plan"]["action_checklist"] == expected


@pytest.mark.parametrize("template", [True, False])
def test_insufficient_evidence_in_wechat_attention_items(template):
    result, _, _, _ = _analyze(profile=None, payload=RAW, source={"volume_change_ratio": None})
    config = Config(stock_list=[], report_renderer_enabled=template)
    with patch("src.notification.get_config", return_value=config), patch("src.services.report_renderer.get_config", return_value=config):
        output = NotificationService().generate_wechat_dashboard([result])
    # Both existing renderers show the first three warning/failure items.
    for i, title in enumerate(TITLES[:3], 1):
        assert f"⚠️ 检查项{i}：{title}（依据不足，未作判断）" in output
    assert all("⚪" not in line and "ℹ️" not in line
               for line in output.splitlines() if "检查项" in line)
