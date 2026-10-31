"""下方リスク・回復期間。頻度、分母、未回復の打切りを明記して扱う。"""

import numpy as np
import pandas as pd

from quant_ai.leveraged_backtest import LeveragedResult
from quant_ai.metrics import performance
from quant_ai.returns import validate_frame


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """日次ES95損失 = -mean(昇順の下位ceil(alpha*n)個)。年率化しない。"""
    validate_frame(returns)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    count = max(1, int(np.ceil(alpha * len(returns))))
    return -float(np.sort(returns.to_numpy())[:count].mean())


def sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    """MAR=0、Sortino=sqrt(A)*mean(r)/sqrt(mean(min(r,0)^2))。全日数を分母とする。"""
    validate_frame(returns)
    downside = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0) ** 2))
    return float(np.sqrt(periods) * returns.mean() / downside) if downside > 0 else np.nan


def drawdown_recovery(returns: pd.Series, initial_date: pd.Timestamp) -> dict:
    """最大DDエピソードの直前ピーク・底・回復日。未回復は打切りとし期間を偽らない。

    trough_to_recovery_daysは底から回復までの暦日。peak_to_recovery_daysも併記。
    ピークが複数なら底以前の最後の同値ピークを採用。初期NAV=1もピーク候補。
    """
    validate_frame(returns)
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    dates = pd.DatetimeIndex([initial_date, *returns.index])
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1
    trough = int(drawdowns.argmin())
    peak = int(
        np.flatnonzero(np.isclose(wealth[: trough + 1], peaks[trough], rtol=1e-12, atol=1e-14))[-1]
    )
    reached = np.flatnonzero(wealth[trough:] >= peaks[trough] * (1 - 1e-12))
    recovery = trough + int(reached[0]) if reached.size else None
    return {
        "peak_date": str(dates[peak].date()),
        "trough_date": str(dates[trough].date()),
        "recovery_date": str(dates[recovery].date()) if recovery is not None else None,
        "recovery_censored": recovery is None,
        "trough_to_recovery_days": (dates[recovery] - dates[trough]).days
        if recovery is not None
        else None,
        "peak_to_recovery_days": (dates[recovery] - dates[peak]).days
        if recovery is not None
        else None,
        "observed_days_after_trough": (dates[-1] - dates[trough]).days,
    }


def evaluate_leveraged(
    result: LeveragedResult,
    target_volatility: float | None,
    leverage_cap: float | None,
) -> dict:
    """コスト後の全指標。Sharpeは日次算術平均、CAGRとは分子を区別する。"""
    daily = result.daily
    returns = daily.net_return
    stats = performance(returns).to_dict()
    stats["sortino"] = sortino_ratio(returns)
    stats["calmar"] = (
        stats["annual_return"] / abs(stats["max_drawdown"]) if stats["max_drawdown"] < 0 else np.nan
    )
    stats["daily_es95_loss"] = expected_shortfall(returns)
    stats["annual_arithmetic_mean"] = float(returns.mean() * 252)
    excess = returns - daily.cash_opportunity_return
    stats["sharpe_excess_cash"] = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
    years = len(daily) / 252
    stats.update(
        annual_turnover=float(daily.turnover.sum() / years),
        total_trade_cost_initial_capital=float(daily.trade_cost.sum()),
        sum_nav_relative_trade_cost=float(daily.trade_cost_return.sum()),
        total_funding_cost_initial_capital=float(daily.funding_cost.sum()),
        total_cash_income_initial_capital=float(daily.cash_income.sum()),
        mean_held_leverage=float(daily.held_leverage.mean()),
        max_realized_leverage=float(
            daily[["held_leverage", "pre_leverage", "post_leverage"]].max().max()
        ),
        borrowing_day_fraction=float((daily.held_cash_weight < -1e-12).mean()),
        cap_drift_breach_day_fraction=float((daily.pre_leverage > leverage_cap + 1e-10).mean())
        if leverage_cap is not None
        else 0.0,
        realized_minus_target_volatility=stats["volatility"] - target_volatility
        if target_volatility is not None
        else np.nan,
        realized_volatility_rmse_63=float(
            np.sqrt(
                np.mean(
                    (returns.rolling(63).std().dropna() * np.sqrt(252) - target_volatility) ** 2
                )
            )
        )
        if target_volatility is not None
        else np.nan,
        accounting_max_error=float(daily.accounting_error.abs().max()),
        start=str(daily.index[0].date()),
        end=str(daily.index[-1].date()),
        observations=len(daily),
    )
    stats.update(drawdown_recovery(returns, result.initial_date))
    return stats
