"""閉形式、境界、予測・執行時点を基準に検証。ネットワーク不要。"""

import numpy as np
import pandas as pd
import pytest

from quant_ai.backtest import backtest
from quant_ai.data_loader import close_prices, download_prices, synthetic_prices
from quant_ai.factors import capm_beta
from quant_ai.metrics import max_drawdown, sharpe_ratio
from quant_ai.ml import chronological_split, compare_models, make_features
from quant_ai.portfolio import cash_allocation, efficient_frontier, optimize_portfolio
from quant_ai.returns import annualized_return, cumulative_returns, log_returns, simple_returns


def test_returns_closed_form():
    """価格100→110→99なら単純10%,-10%、累積-1%。"""
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = simple_returns(prices)
    np.testing.assert_allclose(returns, [0.1, -0.1])
    assert cumulative_returns(returns).iloc[-1] == pytest.approx(-0.01)
    assert log_returns(prices).sum() == pytest.approx(np.log(0.99))
    assert annualized_return(returns, periods=2) == pytest.approx(-0.01)


@pytest.mark.parametrize("prices", [[1.0, np.nan], [1.0, 0.0], [1.0, -1.0]])
def test_returns_reject_bad_prices(prices):
    """欠損・非正価格を暗黙に補完しない。"""
    with pytest.raises(ValueError):
        simple_returns(pd.Series(prices))


def test_drawdown_includes_initial_capital():
    """初日-20%、次の日+10%でもMDDは-20%。"""
    assert max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)
    assert np.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))


def test_capm_known_beta():
    """ノイズなしの既知β=1.7、α=.0004を復元。"""
    market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    model = capm_beta(0.0004 + 1.7 * market, market)
    assert model.params["market_excess"] == pytest.approx(1.7)
    assert model.params["const"] == pytest.approx(0.0004)


def test_min_variance_closed_form_and_frontier():
    """独立2資産の最小分散重みは逆分散比(.8,.2)。"""
    mean = pd.Series([0.04, 0.12], index=["a", "b"])
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=mean.index, columns=mean.index)
    weights = optimize_portfolio(mean, cov)
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-5)
    tangent = optimize_portfolio(mean, cov, objective="max_sharpe", risk_free=0.01)
    assert tangent.sum() == pytest.approx(1)
    assert cash_allocation(tangent, 0.7)["CASH"] == pytest.approx(0.3)
    frontier = efficient_frontier(mean, cov, points=8)
    assert frontier["volatility"].is_monotonic_increasing
    with pytest.raises(ValueError):
        optimize_portfolio(mean, cov, target_return=0.2)


def test_execution_lag_cost_and_drift():
    """t=0の判断はt=1終値約定なので、t=1の倍増を取得してはいけない。"""
    prices = pd.DataFrame({"a": [100.0, 200.0, 220.0, 242.0]})
    weights = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0]})
    report = backtest(prices, weights, cost_bps=0)
    np.testing.assert_allclose(report.net_return, [0, 0.1, 0.1])
    charged = backtest(prices, weights, cost_bps=10)
    assert charged.wealth.iloc[-1] == pytest.approx(1.21 / 1.001)
    assert charged.turnover.iloc[1] == pytest.approx(0, abs=1e-9)
    two = pd.DataFrame({"a": [100.0, 100.0, 200.0, 200.0], "b": [100.0] * 4})
    equal = pd.DataFrame(0.5, index=two.index, columns=two.columns)
    drift = backtest(two, equal, cost_bps=0, rebalance_every=2)
    assert drift.net_return.iloc[1] == pytest.approx(0.5)
    assert drift.turnover.iloc[-1] == pytest.approx(1 / 3)


def test_future_perturbation_does_not_change_features_or_pnl():
    """未来価格だけを書換えても過去の特徴量・バックテスト結果は不変。"""
    prices = synthetic_prices(150, 2)
    changed = prices.copy()
    changed.iloc[100:] *= 2
    x, _ = make_features(prices.iloc[:, 0])
    z, _ = make_features(changed.iloc[:, 0])
    pd.testing.assert_frame_equal(x.iloc[:75], z.iloc[:75])
    signals = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    pd.testing.assert_frame_equal(
        backtest(prices, signals).iloc[:99], backtest(changed, signals).iloc[:99]
    )


def test_purge_boundary_and_models():
    """訓練ラベル終点 < 検証先頭。全モデルが有限の予測を返す。"""
    prices = synthetic_prices(250, 1).iloc[:, 0]
    x, y = make_features(prices)
    splits = chronological_split(x, y, gap=2)
    dates = prices.index
    train_end = dates.get_loc(splits["train"][0].index[-1]) + 2
    assert train_end < dates.get_loc(splits["valid"][0].index[0])
    scores, predictions = compare_models(splits)
    assert np.isfinite(scores.to_numpy()).all()
    assert "lightgbm" in predictions


def test_yahoo_cache_metadata_and_explicit_missing(tmp_path, monkeypatch):
    """外部通信をmockしキャッシュ・出典・欠損方針を確認。実取得の証明ではない。"""
    columns = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
    frame = pd.DataFrame(
        [[100.0, 50.0], [101.0, np.nan], [102.0, 52.0]],
        index=pd.date_range("2020-01-01", periods=3),
        columns=columns,
    )
    calls = []

    def fake_download(*args, **kwargs):
        """取得パラメータを記録する。"""
        calls.append(kwargs)
        return frame.copy()

    monkeypatch.setattr("quant_ai.data_loader.yf.download", fake_download)
    kwargs = {"cache_dir": tmp_path}
    first = download_prices(["AAA", "BBB"], "2020-01-01", "2020-02-01", **kwargs)
    cached = download_prices(["BBB", "AAA"], "2020-01-01", "2020-02-01", **kwargs)
    assert len(calls) == 1 and calls[0]["auto_adjust"] is True
    assert cached.attrs["provenance"]["retrieved_at_utc"]
    with pytest.raises(ValueError):
        close_prices(first)
    assert len(close_prices(first, "drop")) == 2
    assert close_prices(first, "ffill").iloc[1, 1] == 50
