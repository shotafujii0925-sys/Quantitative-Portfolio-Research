"""実際のQlib APIを使った合成データ統合テスト。"""

import numpy as np

from quant_ai.qlib_workflow import build_sample_provider, run_qlib_demo


def test_qlib_end_to_end(tmp_path):
    """初期化→式評価→LGBModel→公式売買バックテスト→risk_analysis。"""
    result = run_qlib_demo(build_sample_provider(tmp_path / "provider", rows=180))
    assert len(result["predictions"]) > 0
    assert np.isfinite(result["predictions"]).all()
    assert len(result["report"]) > 10
    assert result["report"]["cost"].sum() > 0
    assert np.isfinite(result["risk"].to_numpy()).all()
    assert result["positions_count"] > 0
