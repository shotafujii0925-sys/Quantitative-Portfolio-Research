"""終値判断→翌終値約定。現金あり・空売りなし、ドリフトと比例売買コスト。"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from quant_ai.returns import simple_returns, validate_frame


def backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    rebalance_every: int = 1,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
    decision_dates: pd.Index | None = None,
) -> pd.DataFrame:
    """signals[t]をt+1終値で執行。t+1の価格変化には旧保有を適用する。

    signalsは目標NAV比率、Σw<=1。毎k営業日の判断を翌終値で実行する。
    既存保有は価格変化でドリフト。コスト率c=bps/10000に対し
    z=1-c*Σ|z*w_target-w_pre|を解き、自己資金制約を満たす。
    r_net=(1+r_gross)*z-1。現金に年実効risk_freeを付利。
    約定可能性、スプレッド、税、マーケットインパクトは扱わない。
    decision_dates指定時は、その判断日の翌観測日終値だけでリバランスする。
    指定がなければ従来のrebalance_everyを使う。
    """
    validate_frame(prices, positive=True)
    validate_frame(signals)
    if not prices.index.equals(signals.index) or not prices.columns.equals(signals.columns):
        raise ValueError("Signals must exactly match price dates and assets")
    if (signals < 0).any().any() or (signals.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Long-only weights with sum <= 1 required")
    if not 0 <= cost_bps < 10000 or rebalance_every < 1 or risk_free <= -1:
        raise ValueError("Invalid execution parameters")
    if decision_dates is not None:
        if rebalance_every != 1:
            raise ValueError("Use either decision_dates or rebalance_every")
        if not decision_dates.is_unique or not decision_dates.isin(prices.index).all():
            raise ValueError("Decision dates must be unique observed price dates")
    returns = simple_returns(prices)
    held = np.zeros(prices.shape[1])
    rf = (1 + risk_free) ** (1 / 252) - 1
    rows = []
    for i, (_, row) in enumerate(returns.iterrows(), start=1):
        gross = float(held @ row.to_numpy() + (1 - held.sum()) * rf)
        pre = held * (1 + row.to_numpy()) / (1 + gross)
        trade_date = (
            (i - 1) % rebalance_every == 0
            if decision_dates is None
            else prices.index[i - 1] in decision_dates
        )
        if trade_date:
            target = signals.iloc[i - 1].to_numpy()
            c = cost_bps / 10000
            z = brentq(
                lambda z, c=c, target=target, pre=pre: z + c * np.abs(z * target - pre).sum() - 1,
                0,
                1,
            )
            turnover = float(np.abs(z * target - pre).sum())
            held = target.copy()
        else:
            z, turnover, held = 1.0, 0.0, pre
        rows.append(
            {
                "gross_return": gross,
                "cost_fraction": 1 - z,
                "net_return": (1 + gross) * z - 1,
                "turnover": turnover,
                "risky_weight": held.sum(),
            }
        )
    result = pd.DataFrame(rows, index=returns.index)
    result["wealth"] = (1 + result["net_return"]).cumprod()
    if benchmark is not None:
        validate_frame(benchmark, positive=True)
        if not benchmark.index.equals(prices.index):
            raise ValueError("Benchmark prices must have identical dates")
        result["benchmark_return"] = simple_returns(benchmark)
        result["benchmark_wealth"] = (1 + result["benchmark_return"]).cumprod()
        result["active_return"] = result["net_return"] - result["benchmark_return"]
    return result
