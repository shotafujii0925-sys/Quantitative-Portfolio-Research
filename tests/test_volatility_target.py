"""追加エンジンの閉形式・旧結果一致・時点・資金調達・未来情報不変性。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.backtest import backtest
from quant_ai.data_loader import synthetic_prices
from quant_ai.downside_metrics import drawdown_recovery, expected_shortfall, sortino_ratio
from quant_ai.leveraged_backtest import leveraged_backtest
from quant_ai.volatility_target import volatility_schedule


def test_exact_legacy_parity() -> None:
    """金利0・非レバレッジなら旧エンジンの全日損益・turnoverと一致する。"""
    prices = synthetic_prices(120, 3)
    decisions = prices.index[::21]
    targets = pd.DataFrame(1 / 3, index=decisions, columns=prices.columns)
    old = backtest(prices, targets.reindex(prices.index).ffill(), decision_dates=decisions)
    new = leveraged_backtest(prices, targets).daily
    np.testing.assert_allclose(new.net_return, old.net_return, atol=1e-12)
    np.testing.assert_allclose(new.turnover, old.turnover, atol=1e-12)
    assert new.accounting_error.abs().max() < 1e-12


def test_next_close_and_actual_trade_cost() -> None:
    """tの判断でt+1の倍増を取得しない。2倍投資の初回コストは2z*c。"""
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"a": [100.0, 200.0, 220.0, 242.0]}, index=dates)
    targets = pd.DataFrame({"a": [2.0]}, index=dates[:1])
    daily = leveraged_backtest(prices, targets, cost_bps=10).daily
    z = 1 / (1 + 2 * 0.001)
    assert daily.net_return.iloc[0] == pytest.approx(z - 1)
    assert daily.trade_cost.iloc[0] == pytest.approx(2 * z * 0.001)
    assert daily.trade_amount.iloc[0] == pytest.approx(2 * z)
    assert daily.net_return.iloc[1] == pytest.approx(0.2)
    assert daily.held_a.iloc[0] == 0
    assert daily.pre_a.iloc[0] == 0
    assert daily.post_a.iloc[0] == pytest.approx(2)
    # 2倍に日々戻す戦略ではないので、3日目のCAGR/損益は単純2倍にならない。
    assert daily.wealth.iloc[-1] == pytest.approx(z * 1.42)


def test_borrow_interest_and_weekend_accrual() -> None:
    """借入利息は区間開始の負の現金にACT/365。金曜日から月曜日は3日分。"""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"a": [100.0, 100.0, 100.0]}, index=dates)
    targets = pd.DataFrame({"a": [2.0]}, index=dates[:1])
    daily = leveraged_backtest(prices, targets, cost_bps=0, borrow_rate=0.10).daily
    cost = 1.10 ** (3 / 365) - 1
    assert daily.funding_cost.iloc[0] == 0
    assert daily.funding_cost.iloc[1] == pytest.approx(cost)
    assert daily.wealth.iloc[-1] == pytest.approx(1 - cost)
    assert daily.pre_leverage.iloc[-1] > 2  # ドリフトで目標上限を超え得る


def test_cash_interest_and_accounting() -> None:
    """現金だけの運用は期間の年実効金利と一致。借入費用は生じない。"""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"a": [100.0, 110.0, 120.0]}, index=dates)
    targets = pd.DataFrame({"a": [0.0]}, index=dates[:1])
    daily = leveraged_backtest(prices, targets, cost_bps=0, cash_rate=0.04, borrow_rate=0.06).daily
    assert daily.wealth.iloc[-1] == pytest.approx(1.04 ** (4 / 365))
    assert daily.funding_cost.sum() == 0
    assert daily.accounting_error.abs().max() < 1e-12


def test_ex_ante_forecast_cap_and_future_invariance() -> None:
    """予測ボラは過去窓だけ。未来価格変更が過去目標・過去NAVを変えない。"""
    prices = synthetic_prices(400, 3)
    targets, audit = volatility_schedule(
        prices, "min_variance", lookback=63, evaluation_start="2020", target_volatility=0.10
    )
    np.testing.assert_allclose(audit.predicted_volatility, 0.10)
    assert (audit.history_end <= audit.index).all()
    assert (audit.execution > audit.index).all()
    capped, capped_audit = volatility_schedule(
        prices,
        "min_variance",
        lookback=63,
        evaluation_start="2020",
        target_volatility=0.30,
        leverage_cap=1.5,
    )
    assert (capped.sum(axis=1) <= 1.5 + 1e-12).all()
    assert capped_audit.cap_binding.any()
    changed = prices.copy()
    changed.iloc[250:] *= [2.0, 0.5, 1.2]
    future, future_audit = volatility_schedule(
        changed, "min_variance", lookback=63, evaluation_start="2020", target_volatility=0.10
    )
    cutoff = prices.index[249]
    pd.testing.assert_frame_equal(targets.loc[:cutoff], future.loc[:cutoff])
    pd.testing.assert_frame_equal(audit.loc[:cutoff], future_audit.loc[:cutoff])
    old = leveraged_backtest(prices.loc[targets.index[0] :], targets).daily
    new = leveraged_backtest(changed.loc[future.index[0] :], future).daily
    pd.testing.assert_frame_equal(old.loc[:cutoff], new.loc[:cutoff])


def test_downside_definitions_and_recovery_censoring() -> None:
    """ESの符号、Sortinoの全観測数分母、回復/未回復を既知系列で確認。"""
    dates = pd.bdate_range("2020-01-01", periods=20)
    returns = pd.Series([-0.2] + [0.01] * 19, index=dates)
    assert expected_shortfall(returns) == pytest.approx(0.2)
    expected = np.sqrt(252) * returns.mean() / np.sqrt(0.2**2 / 20)
    assert sortino_ratio(returns) == pytest.approx(expected)
    censored = drawdown_recovery(returns, pd.Timestamp("2019-12-31"))
    assert censored["recovery_censored"] is True
    assert censored["trough_to_recovery_days"] is None
    recovered = drawdown_recovery(
        pd.Series([-0.2, 0.25], index=dates[:2]), pd.Timestamp("2019-12-31")
    )
    assert recovered["recovery_censored"] is False
    assert recovered["trough_to_recovery_days"] == 1


def test_bankruptcy_and_undefined_risk_are_not_hidden() -> None:
    """破綻・ゼロ分散を都合よく補完して運用継続しない。"""
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"a": [100.0, 100.0, 1.0]}, index=dates)
    targets = pd.DataFrame({"a": [2.0]}, index=dates[:1])
    with pytest.raises(RuntimeError, match="Insolvent"):
        leveraged_backtest(prices, targets, cost_bps=0)
    constant = pd.DataFrame(100.0, index=pd.bdate_range("2020-01-01", periods=100), columns=["a"])
    with pytest.raises(ValueError, match="zero volatility"):
        volatility_schedule(constant, "equal_weight", lookback=20, evaluation_start="2020")
