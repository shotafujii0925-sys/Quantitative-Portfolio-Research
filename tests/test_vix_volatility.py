"""VIXボラ目標の時点・校正・換算・統計処理の検証。合成データのみ。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.leveraged_backtest import leveraged_backtest
from quant_ai.returns import simple_returns
from quant_ai.vix_research import bootstrap_mean_difference, holm, paired_bootstrap
from quant_ai.vix_volatility import (
    align_vix,
    calibration_level,
    covariance_with_spy_vol,
    decision_dates,
    forecast_table,
    forward_realized_vol,
    qlike,
    rescaled_cagr,
    vix_schedule,
)
from quant_ai.volatility_target import volatility_schedule

START = "2021-02-01"


def market(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """SPY/TLT/GLDとVIXの合成系列。VIXの水準はSPYのボラと連動させる。"""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    vol = np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.2) * 0.01
    spy = rng.normal(0.0004, 1.0, n) * vol
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.cumprod(1 + spy),
            "TLT": 100 * np.cumprod(1 + rng.normal(0.0002, 0.006, n)),
            "GLD": 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n)),
        },
        index=dates,
    )
    vix = pd.Series(vol * np.sqrt(252) * 120, index=dates, name="VIX")
    return prices, vix


@pytest.mark.parametrize("method", ["equal_weight", "min_variance"])
def test_hist252_matches_existing_volatility_target(method: str) -> None:
    """hist252は既存のボラ目標と完全に一致する（比較の土台が同じ）。"""
    prices, vix = market()
    new, _ = vix_schedule(prices, vix, method, "hist252", evaluation_start=START)
    old, _ = volatility_schedule(
        prices, method, evaluation_start=START, target_volatility=0.10, leverage_cap=1.5
    )
    pd.testing.assert_frame_equal(new, old, check_freq=False)


def test_none_keeps_base_weights_at_unit_exposure() -> None:
    prices, vix = market()
    targets, audit = vix_schedule(prices, vix, "equal_weight", "none", evaluation_start=START)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (audit.target_exposure == 1.0).all()


def test_spy_only_vix_raw_exposure_is_target_over_vix() -> None:
    """SPYのみ・校正なしなら L=min(目標/(VIX/100), 上限)、他の資産は0。"""
    prices, vix = market()
    targets, audit = vix_schedule(prices, vix, "spy_only", "vix_raw", evaluation_start=START)
    expected = np.minimum(0.10 / (vix.loc[targets.index] / 100), 1.5)
    assert np.allclose(targets.SPY, expected)
    assert (targets[["TLT", "GLD"]] == 0).all().all()
    assert audit.target_exposure.max() <= 1.5


def test_rv21_uses_only_the_last_21_returns() -> None:
    prices, vix = market()
    targets, audit = vix_schedule(
        prices, vix, "spy_only", "rv21", evaluation_start=START, leverage_cap=None
    )
    date = targets.index[3]
    returns = simple_returns(prices)["SPY"].loc[:date].iloc[-21:]
    assert audit.loc[date, "spy_vol_forecast"] == pytest.approx(returns.std() * np.sqrt(252))
    assert targets.loc[date, "SPY"] == pytest.approx(0.10 / (returns.std() * np.sqrt(252)))


@pytest.mark.parametrize("forecast", ["hist252", "rv21", "vix_cal", "vix_raw"])
@pytest.mark.parametrize("base", ["spy_only", "equal_weight", "min_variance"])
def test_future_prices_and_vix_do_not_change_past_targets(base: str, forecast: str) -> None:
    """判断日より後の価格とVIXを書き換えても、その日までの配分は変わらない。"""
    prices, vix = market()
    a, _ = vix_schedule(prices, vix, base, forecast, evaluation_start=START)
    date = a.index[len(a) // 2]
    later = prices.index > date
    prices2, vix2 = prices.copy(), vix.copy()
    prices2.loc[later] = prices2.loc[later] * np.linspace(1, 3, later.sum())[:, None]
    vix2.loc[later] = vix2.loc[later] * 5
    b, _ = vix_schedule(prices2, vix2, base, forecast, evaluation_start=START)
    pd.testing.assert_frame_equal(a.loc[:date], b.loc[:date], check_freq=False)


def test_forward_realized_vol_window_and_timing() -> None:
    """日iの値はr[i+2]〜r[i+22]の21本。"""
    prices, _ = market(100)
    r = simple_returns(prices)["SPY"].reindex(prices.index)
    value = forward_realized_vol(r).iloc[10]
    assert value == pytest.approx(r.iloc[12:33].std(ddof=1) * np.sqrt(252))
    assert np.isnan(forward_realized_vol(r).iloc[-22])


def test_calibration_matches_brute_force_and_uses_completed_windows() -> None:
    """k_tはi+22<=tの確定分の平均に一致し、それより新しい結果に依存しない。"""
    prices, vix = market(200)
    r = simple_returns(prices)["SPY"].reindex(prices.index)
    k = calibration_level(r, vix)
    t = 150
    values = [
        np.log(r.iloc[i + 2 : i + 23].std(ddof=1) * np.sqrt(252)) - np.log(vix.iloc[i] / 100)
        for i in range(0, t - 22 + 1)
    ]
    assert k.iloc[t] == pytest.approx(np.mean(values))
    r2 = r.copy()
    r2.iloc[t + 1 :] = r2.iloc[t + 1 :] * 5
    assert calibration_level(r2, vix).iloc[t] == pytest.approx(k.iloc[t])


def test_calibration_recovers_constant_ratio() -> None:
    """実現ボラ=VIXの定数倍なら、kは定数の対数に近い。

    21本の標本標準偏差の対数は E[log s] ≈ log σ - 1/(2*20) だけ下に偏る。
    """
    rng = np.random.default_rng(1)
    n = 3000
    dates = pd.bdate_range("2010-01-01", periods=n)
    r = pd.Series(rng.normal(0, 0.01, n), index=dates)
    vix = pd.Series(0.01 * np.sqrt(252) * 100 / 0.8, index=dates)
    k = calibration_level(r, vix).iloc[-1]
    assert k == pytest.approx(np.log(0.8) - 1 / 40, abs=0.01)


def test_covariance_replaces_only_spy_vol() -> None:
    prices, _ = market(300)
    history = simple_returns(prices).iloc[-252:]
    base = history.cov() * 252
    new = covariance_with_spy_vol(history, 0.30)
    assert np.sqrt(new.loc["SPY", "SPY"]) == pytest.approx(0.30)
    assert new.loc["TLT", "GLD"] == pytest.approx(base.loc["TLT", "GLD"])
    assert new.loc["TLT", "TLT"] == pytest.approx(base.loc["TLT", "TLT"])
    corr = new.loc["SPY", "TLT"] / np.sqrt(new.loc["SPY", "SPY"] * new.loc["TLT", "TLT"])
    old = base.loc["SPY", "TLT"] / np.sqrt(base.loc["SPY", "SPY"] * base.loc["TLT", "TLT"])
    assert corr == pytest.approx(old)
    assert np.linalg.eigvalsh(new.to_numpy()).min() > 0
    pd.testing.assert_frame_equal(covariance_with_spy_vol(history, None), base)


def test_missing_vix_on_decision_stops_without_filling() -> None:
    prices, vix = market()
    targets, _ = vix_schedule(prices, vix, "spy_only", "vix_raw", evaluation_start=START)
    broken = vix.copy()
    broken.loc[targets.index[2]] = np.nan
    with pytest.raises(ValueError, match="missing"):
        vix_schedule(prices, broken, "spy_only", "vix_raw", evaluation_start=START)


def test_align_vix_does_not_fill() -> None:
    prices, vix = market(50)
    aligned = align_vix(vix.drop(vix.index[10]), prices)
    assert np.isnan(aligned.iloc[10])
    assert aligned.index.equals(prices.index)


def test_weekly_decisions_hold_last_month_end_base_weights() -> None:
    """週次判断でも基本配分は直近の月末のまま、倍率だけが動く。"""
    prices, vix = market()
    monthly, _ = vix_schedule(
        prices, vix, "equal_weight", "none", evaluation_start=START, frequency="M"
    )
    weekly, audit = vix_schedule(
        prices, vix, "equal_weight", "rv21", evaluation_start=START, frequency="W"
    )
    assert len(weekly) > 3 * len(monthly)
    shares = weekly.div(weekly.sum(axis=1), axis=0)
    assert np.allclose(shares, 1 / 3)
    assert (audit.execution > audit.index).all()
    assert decision_dates(prices, "W").isin(prices.index).all()


def test_last_observation_is_not_a_decision() -> None:
    prices, _ = market()
    for frequency in ("M", "W"):
        assert prices.index[-1] not in decision_dates(prices, frequency)


def test_reallocation_reduces_spy_when_vix_is_high() -> None:
    """VIXを高くすると、配分も変更する最小分散はSPYの比率を下げる。"""
    prices, vix = market()
    low, _ = vix_schedule(
        prices, vix * 0.5, "min_variance", "vix_raw", evaluation_start=START, reallocate=True
    )
    high, _ = vix_schedule(
        prices, vix * 3, "min_variance", "vix_raw", evaluation_start=START, reallocate=True
    )
    low_share = low.SPY / low.sum(axis=1)
    high_share = high.SPY / high.sum(axis=1)
    assert high_share.mean() < low_share.mean()
    with pytest.raises(ValueError):
        vix_schedule(prices, vix, "spy_only", "vix_cal", evaluation_start=START, reallocate=True)


def test_vix_targets_work_with_the_backtest_engine() -> None:
    """t判断の倍率はt+1→t+2から損益に反映され、会計恒等式が保たれる。"""
    prices, vix = market()
    targets, _ = vix_schedule(prices, vix, "spy_only", "vix_cal", evaluation_start=START)
    result = leveraged_backtest(prices.loc[targets.index[0] :], targets).daily
    assert result.accounting_error.abs().max() < 1e-10
    assert result.held_SPY.iloc[0] == 0
    assert result.post_SPY.iloc[0] == pytest.approx(targets.SPY.iloc[0])


def test_forecast_table_excludes_unresolved_months() -> None:
    prices, vix = market()
    table = forecast_table(prices, vix)
    last = prices.index.get_loc(table.index[-1])
    assert last + 22 <= len(prices) - 1
    assert table.notna().all().all()
    assert (table.drop(columns="realized") > 0).all().all()


def test_qlike_known_values() -> None:
    assert qlike(np.array([0.2]), np.array([0.2]))[0] == pytest.approx(0.0)
    x = (0.3 / 0.2) ** 2
    assert qlike(np.array([0.2]), np.array([0.3]))[0] == pytest.approx(x - np.log(x) - 1)
    assert qlike(np.array([0.2]), np.array([0.3]))[0] > 0


def test_rescaled_cagr_reaches_target_volatility() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.02, 2000))
    expected = np.expm1(np.log1p(r * 0.10 / (r.std() * np.sqrt(252))).mean() * 252)
    assert rescaled_cagr(r) == pytest.approx(expected)
    # すでに目標ボラなら元のCAGRと一致する
    scaled = r * 0.10 / (r.std() * np.sqrt(252))
    assert rescaled_cagr(scaled) == pytest.approx(np.expm1(np.log1p(scaled).mean() * 252))


def test_paired_bootstrap_identical_columns_have_no_difference() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(0, 0.01, 500)})
    frame["b"] = frame.a
    table = paired_bootstrap(frame, [("a", "b")], block=21, repetitions=200)
    assert (table.lower_95 == 0).all() and (table.upper_95 == 0).all()
    assert (table.p_two_sided == 1).all()


def test_paired_bootstrap_detects_clear_difference() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.0002, 0.01, 2000)
    frame = pd.DataFrame({"a": base + 0.002, "b": base})
    table = paired_bootstrap(frame, [("a", "b")], block=21, repetitions=500)
    row = table[table.metric == "sharpe"].iloc[0]
    assert row.lower_95 > 0 and row.p_two_sided < 0.01


def test_holm_known_values_and_monotonic() -> None:
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adjusted = holm(p)
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert holm(pd.Series({"a": 0.9, "b": 0.8})).max() == 1.0


def test_bootstrap_mean_difference_centers_on_mean() -> None:
    rng = np.random.default_rng(2)
    d = rng.normal(0.5, 0.1, 240)
    result = bootstrap_mean_difference(d, repetitions=500)
    assert result["mean"] == pytest.approx(d.mean())
    assert result["lower_95"] < result["mean"] < result["upper_95"]
    assert result["p_two_sided"] < 0.01
