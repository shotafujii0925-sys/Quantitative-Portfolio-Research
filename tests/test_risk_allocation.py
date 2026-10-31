"""月次リスク配分の既知解と時点整合性を検証する。外部接続不要。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.backtest import backtest
from quant_ai.data_loader import synthetic_prices
from quant_ai.risk_allocation import monthly_targets, risk_weights, run_allocation_study


def test_risk_weight_formulas() -> None:
    """独立・σ比1:2なら逆σは2/3:1/3、最小分散は.8:.2。"""
    history = pd.DataFrame(
        {"a": np.tile([0.01, -0.01, 0.01, -0.01], 10), "b": np.tile([0.02, 0.02, -0.02, -0.02], 10)}
    )
    np.testing.assert_allclose(risk_weights(history, "equal_weight"), [0.5, 0.5])
    np.testing.assert_allclose(risk_weights(history, "inverse_vol"), [2 / 3, 1 / 3])
    np.testing.assert_allclose(risk_weights(history, "min_variance"), [0.8, 0.2], atol=1e-5)
    shrunk = risk_weights(history, "shrink_min_variance")
    assert shrunk.sum() == pytest.approx(1)
    assert (shrunk >= 0).all()


def test_monthly_clock_and_future_invariance() -> None:
    """過去窓の終端<=判断<約定<新保有の損益。未来価格変更で過去目標は不変。"""
    prices = synthetic_prices(400, 3)
    targets, audit = monthly_targets(prices, "min_variance", 63, "2020-01-01")
    assert (audit.observations == 63).all()
    assert (audit.history_end <= audit.index).all()
    assert (audit.execution > audit.index).all()
    known = audit.first_new_weight_pnl.notna()
    assert (audit.loc[known, "first_new_weight_pnl"] > audit.loc[known, "execution"]).all()
    changed = prices.copy()
    changed.iloc[250:] *= [2.0, 0.5, 1.3]
    later_targets, _ = monthly_targets(changed, "min_variance", 63, "2020-01-01")
    pd.testing.assert_frame_equal(
        targets.loc[: prices.index[249]], later_targets.loc[: prices.index[249]]
    )
    study = run_allocation_study(prices, lookback=63, evaluation_start="2020-01-01")
    future = run_allocation_study(changed, lookback=63, evaluation_start="2020-01-01")
    for method in study.reports:
        pd.testing.assert_frame_equal(
            study.reports[method].loc[: prices.index[249]],
            future.reports[method].loc[: prices.index[249]],
        )


def test_explicit_decisions_prevent_daily_rebalance() -> None:
    """月末指定だけで売買し、日々の比率ドリフトを許す。"""
    dates = pd.bdate_range("2020-01-30", periods=5)
    prices = pd.DataFrame({"a": [100.0, 200.0, 200.0, 400.0, 400.0], "b": [100.0] * 5}, index=dates)
    signals = pd.DataFrame(0.5, index=dates, columns=prices.columns)
    result = backtest(prices, signals, cost_bps=0, decision_dates=dates[[0, 3]])
    np.testing.assert_allclose(result.net_return, [0.0, 0.0, 0.5, 0.0])
    assert result.turnover.iloc[2] == 0
    assert result.turnover.iloc[-1] == pytest.approx(1 / 3)


def test_reject_unknown_decision_dates() -> None:
    """存在しない判断日は補完して実行しない。"""
    prices = synthetic_prices(20, 2)
    signals = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    with pytest.raises(ValueError):
        backtest(prices, signals, decision_dates=pd.DatetimeIndex(["1999-01-01"]))
