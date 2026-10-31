"""保存された日次台帳と集計値・時点・費用の整合性を独立に監査する。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.risk_research import load_risk_research

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """既存価格・純損益分解・資金制約・約定日・費用前後の関係をassertで確認。"""
    bundle = load_risk_research(ROOT)
    table = bundle.summary
    result = {
        "status": "passed",
        "checked_daily_paths": len(bundle.daily),
        "scenario_rows": len(table),
        "checks": [],
    }
    path = ROOT / "data/processed/spy_tlt_gld_adjusted.parquet"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle.provenance["sha256"]
    for case_id, obj in bundle.daily.items():
        d = obj.daily
        row = table.loc[case_id]
        np.testing.assert_allclose((1 + d.net_return).cumprod(), d.wealth, atol=1e-12)
        np.testing.assert_allclose(d.gross_return - d.net_return, d.trade_cost_return, atol=1e-12)
        np.testing.assert_allclose(
            d.asset_return + d.cash_income_return - d.funding_cost_return - d.trade_cost_return,
            d.net_return,
            atol=1e-12,
        )
        np.testing.assert_allclose(d.trade_amount * row.cost_bps / 10000, d.trade_cost, atol=1e-12)
        np.testing.assert_allclose(
            d[["post_SPY", "post_TLT", "post_GLD"]].sum(axis=1) + d.post_cash_weight, 1, atol=1e-12
        )
        np.testing.assert_allclose(
            d.borrowed, np.maximum(-d.post_cash_weight * d.wealth, 0), atol=1e-12
        )
        assert d.accounting_error.abs().max() < 1e-10
        assert np.isfinite(d.to_numpy(dtype=float)).all()
        assert abs(d.trade_cost.sum() - row.total_trade_cost_initial_capital) < 1e-12
        audit = bundle.audits[case_id]
        assert (pd.to_datetime(audit.history_end) <= audit.index).all()
        assert (pd.to_datetime(audit.execution) > audit.index).all()
        execution_dates = pd.DatetimeIndex(pd.to_datetime(audit.execution))
        assert d.index[d.rebalance].equals(execution_dates)
        if case_id.startswith("cap15"):
            assert audit.target_exposure.max() <= 1.5 + 1e-12
        if case_id.startswith("unbounded"):
            np.testing.assert_allclose(audit.predicted_volatility, 0.10, atol=1e-12)
    assert (table[table.cost_bps == 0].total_trade_cost_initial_capital == 0).all()
    assert table.accounting_max_error.max() < 1e-10
    assert np.isfinite(
        table[
            [
                "annual_return",
                "volatility",
                "sharpe",
                "sortino",
                "calmar",
                "daily_es95_loss",
                "annual_turnover",
            ]
        ].to_numpy()
    ).all()
    np.testing.assert_allclose(
        table.gross_path_annual_return - table.annual_return, table.cost_cagr_drag, atol=1e-12
    )
    assert table[table.family == "lookback_sensitivity"].start.nunique() == 1
    result["checks"] = [
        "price hash",
        "compounded NAV",
        "gross-net cost identity",
        "asset/cash/funding/trading P&L identity",
        "actual traded amounts",
        "signed cash and borrowing",
        "past-only history and next-close execution",
        "target constraint and ex-ante forecast",
        "zero costs",
        "common sensitivity start",
    ]
    output = ROOT / "outputs/reports/リスク統一比較_数値監査.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
