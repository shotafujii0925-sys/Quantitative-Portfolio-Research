"""分解の基準・交互作用・複利・実支払額を合成市場の閉形式で検証。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.data_loader import synthetic_prices
from quant_ai.leveraged_backtest import leveraged_backtest
from quant_ai.metrics import performance
from quant_ai.return_attribution import (
    COMPONENTS,
    actual_payments,
    annual_contributions,
    cagr_bridge,
    daily_attribution,
)


def test_allocation_exposure_and_interaction_definition() -> None:
    """配分差4.5%、倍率効果7%=基準への倍率2.5%+交互作用4.5%。"""
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"a": [100.0, 100.0, 110.0], "b": [100.0, 100.0, 95.0]}, index=dates)
    equal = pd.DataFrame([[0.5, 0.5]], index=dates[:1], columns=prices.columns)
    raw = pd.DataFrame([[0.8, 0.2]], index=dates[:1], columns=prices.columns)
    benchmark = leveraged_backtest(prices, equal, cost_bps=0)
    unlevered = leveraged_backtest(prices, raw, cost_bps=0)
    vt = leveraged_backtest(prices, raw * 2, cost_bps=0)
    pieces = daily_attribution(prices, vt, benchmark, unlevered)
    assert pieces.allocation.iloc[-1] == pytest.approx(0.045)
    assert pieces.exposure.iloc[-1] == pytest.approx(0.07)
    assert pieces.exposure_on_benchmark.iloc[-1] == pytest.approx(0.025)
    assert pieces.interaction_in_exposure.iloc[-1] == pytest.approx(0.045)
    assert pieces.active_return.iloc[-1] == pytest.approx(0.115)
    assert pieces.active_return.iloc[0] == 0  # 判断翌日の値動きを先取りしない


def test_flat_market_funding_and_incremental_trade_cost() -> None:
    """相場損益0なら、実支払利息と追加売買費用だけで差が説明される。"""
    dates = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame(100.0, index=dates, columns=["a"])
    targets = pd.DataFrame(1.0, index=dates[:1], columns=prices.columns)
    benchmark = leveraged_backtest(prices, targets, cost_bps=10)
    portfolio = leveraged_backtest(prices, targets * 2, cost_bps=10, borrow_rate=0.03)
    pieces = daily_attribution(prices, portfolio, benchmark, benchmark)
    np.testing.assert_allclose(pieces.allocation, 0)
    np.testing.assert_allclose(pieces.exposure, 0)
    assert pieces.funding_cost.sum() < 0
    assert pieces.extra_trading_cost.iloc[0] < 0
    np.testing.assert_allclose(
        pieces[list(COMPONENTS)].sum(axis=1), pieces.active_return, atol=1e-12
    )


def test_log_linking_and_actual_dollars_reconcile() -> None:
    """日次算術差・年率対数差・ドル損益が、それぞれの単位で厳密一致する。"""
    prices = synthetic_prices(100, 2)
    dates = prices.index[::21]
    equal = pd.DataFrame(0.5, index=dates, columns=prices.columns)
    raw = pd.DataFrame(np.tile([0.8, 0.2], (len(dates), 1)), index=dates, columns=prices.columns)
    benchmark = leveraged_backtest(prices, equal)
    unlevered = leveraged_backtest(prices, raw)
    portfolio = leveraged_backtest(prices, raw * 1.3, cash_rate=0.02, borrow_rate=0.03)
    pieces = daily_attribution(prices, portfolio, benchmark, unlevered)
    annual = annual_contributions(pieces)
    assert annual.loc[list(COMPONENTS), "arithmetic_annual_contribution"].sum() == pytest.approx(
        pieces.active_return.mean() * 252
    )
    expected = (
        np.log1p(portfolio.daily.net_return).mean() - np.log1p(benchmark.daily.net_return).mean()
    ) * 252
    assert annual.loc[list(COMPONENTS), "annual_log_contribution"].sum() == pytest.approx(expected)
    actual = actual_payments(portfolio, prices)
    assert actual.total_pnl == pytest.approx(portfolio.daily.wealth.iloc[-1] - 1)
    assert abs(actual.residual) < 1e-12


def test_equal_return_limit_and_zero_components() -> None:
    """rp=rbでも連結係数が0/0にならず、全寄与0。"""
    prices = synthetic_prices(60, 2)
    targets = pd.DataFrame(0.5, index=prices.index[::21], columns=prices.columns)
    path = leveraged_backtest(prices, targets)
    pieces = daily_attribution(prices, path, path, path)
    annual = annual_contributions(pieces)
    assert np.isfinite(annual.to_numpy()).all()
    np.testing.assert_allclose(annual, 0, atol=1e-12)
    bridge = cagr_bridge(path, path, path, path, path, path, path)
    assert bridge["total_cagr_difference"] == 0


def test_counterfactual_cagr_bridge() -> None:
    """費用0経路と借入だけの経路を使い、CAGR差の橋渡しが閉じる。"""
    prices = synthetic_prices(100, 2)
    dates = prices.index[::21]
    equal = pd.DataFrame(0.5, index=dates, columns=prices.columns)
    raw = pd.DataFrame(np.tile([0.8, 0.2], (len(dates), 1)), index=dates, columns=prices.columns)
    bg = leveraged_backtest(prices, equal, cost_bps=0)
    bn = leveraged_backtest(prices, equal)
    rg = leveraged_backtest(prices, raw, cost_bps=0)
    vg = leveraged_backtest(prices, raw * 1.3, cost_bps=0)
    vn = leveraged_backtest(prices, raw * 1.3)
    vb = leveraged_backtest(prices, raw * 1.3, borrow_rate=0.03)
    vf = leveraged_backtest(prices, raw * 1.3, cash_rate=0.02, borrow_rate=0.03)
    bridge = cagr_bridge(bg, bn, rg, vg, vn, vb, vf)
    assert bridge["funding_cost"] < 0
    assert bridge["cash_income"] >= 0
    assert sum(bridge[c] for c in COMPONENTS) == pytest.approx(
        performance(vf.daily.net_return).annual_return
        - performance(bn.daily.net_return).annual_return
    )
    assert abs(bridge["residual"]) < 1e-12


def test_misaligned_paths_rejected() -> None:
    """期間が違う経路を都合よくinner joinして分解しない。"""
    prices = synthetic_prices(60, 2)
    targets = pd.DataFrame(0.5, index=prices.index[::21], columns=prices.columns)
    path = leveraged_backtest(prices, targets)
    shorter = leveraged_backtest(prices.iloc[:-1], targets)
    with pytest.raises(ValueError, match="evaluation dates"):
        daily_attribution(prices, path, shorter, path)
