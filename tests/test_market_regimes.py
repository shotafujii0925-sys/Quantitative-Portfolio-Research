"""局面の時点整合・ペア比較・再標本化を合成データで検証する。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.market_regimes import (
    GROUPS,
    market_states,
    pair_summary,
    paired_regime_bootstrap,
    rolling_comparison,
)


def test_market_states_ignore_today_and_future() -> None:
    """今日以後の価格を書き換えても今日までの局面は変わらない。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2000-01-01", periods=800)
    prices = pd.DataFrame(
        100 * np.exp(rng.normal(0, 0.01, (800, 3)).cumsum(axis=0)),
        index=dates,
        columns=["SPY", "TLT", "GLD"],
    )
    altered = prices.copy()
    altered.iloc[500:] *= 2
    pd.testing.assert_frame_equal(
        market_states(prices).iloc[:501], market_states(altered).iloc[:501]
    )
    states = market_states(prices)
    valid = states["all"]
    assert (states.loc[valid, list(GROUPS[1:5])].sum(axis=1) == 1).all()
    expected = prices.SPY.iloc[499] / prices.SPY.iloc[247] - 1
    assert states.stock_trend.iloc[500] == pytest.approx(expected)


def test_equal_pair_has_zero_difference_and_zero_interval() -> None:
    """同じ運用なら全てのブロック・群の差と区間はゼロになる。"""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2000-01-01", periods=200)
    r = rng.normal(0, 0.01, 200)
    pair = pd.DataFrame({"minimum": r, "equal": r}, index=dates)
    states = pd.DataFrame(True, index=dates, columns=GROUPS)
    summary = pair_summary(pair)
    assert summary["cagr_difference"] == 0
    assert summary["sharpe_difference"] == 0
    result = paired_regime_bootstrap(pair, states, block_days=21, repetitions=200)
    np.testing.assert_allclose(result[["estimate", "lower_95", "upper_95"]], 0, atol=1e-12)
    np.testing.assert_allclose(rolling_comparison(pair, 21), 0, atol=1e-12)


def test_known_compound_gap_and_reproducible_bootstrap() -> None:
    """CAGR差は複利で計算。非整数個ブロックの末尾も標本数を維持。"""
    dates = pd.bdate_range("2000-01-01", periods=101)
    pair = pd.DataFrame(
        {"minimum": np.tile([0.01, -0.005], 51)[:101], "equal": np.tile([0.005, -0.004], 51)[:101]},
        index=dates,
    )
    expected = np.expm1(np.log1p(pair).mean() * 252)
    assert pair_summary(pair)["cagr_difference"] == pytest.approx(expected.minimum - expected.equal)
    states = pd.DataFrame(True, index=dates, columns=GROUPS)
    a = paired_regime_bootstrap(pair, states, block_days=21, repetitions=200)
    b = paired_regime_bootstrap(pair, states, block_days=21, repetitions=200)
    pd.testing.assert_frame_equal(a, b)
    assert a.estimate.iloc[1] == pytest.approx(expected.minimum - expected.equal)


def test_missing_returns_and_misaligned_states_rejected() -> None:
    """欠損や局面の日付ずれを落として比較しない。"""
    dates = pd.bdate_range("2000-01-01", periods=101)
    pair = pd.DataFrame({"minimum": 0.001, "equal": 0.002}, index=dates)
    missing = pair.copy()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        pair_summary(missing)
    states = pd.DataFrame(True, index=dates, columns=GROUPS).iloc[:-1]
    with pytest.raises(ValueError, match="paired date"):
        paired_regime_bootstrap(pair, states, repetitions=200)


def test_block_sufficient_statistics_match_direct_resampling() -> None:
    """集約実装を実際の101日再標本の直接計算で照合し、末尾17日も検証する。"""
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2000-01-01", periods=101)
    pair = pd.DataFrame(
        rng.normal(0.001, 0.01, (101, 2)), index=dates, columns=["minimum", "equal"]
    )
    states = pd.DataFrame(True, index=dates, columns=GROUPS)
    result = paired_regime_bootstrap(pair, states, block_days=21, repetitions=128, seed=7)
    sampler = np.random.default_rng(7)
    direct = []
    for _ in range(2):
        starts = sampler.integers(101, size=(64, 4))
        tails = sampler.integers(101, size=64)
        for block_starts, tail in zip(starts, tails, strict=True):
            indexes = np.concatenate(
                [
                    ((block_starts[:, None] + np.arange(21)) % 101).ravel(),
                    (tail + np.arange(17)) % 101,
                ]
            )
            sample = pair.iloc[indexes].to_numpy()
            cagr = np.expm1(np.log1p(sample).mean(axis=0) * 252)
            direct.append(cagr[0] - cagr[1])
    expected = np.quantile(direct, [0.025, 0.975])
    actual = result[(result.group == "all") & (result.metric == "cagr_difference")].iloc[0]
    np.testing.assert_allclose([actual.lower_95, actual.upper_95], expected, atol=1e-12)
