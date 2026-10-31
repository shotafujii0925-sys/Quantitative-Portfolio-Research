"""運用指標。無リスク金利は年実効金利、収益率は単純収益率で受け取る。"""

import numpy as np
import pandas as pd

from quant_ai.returns import annualized_return, cumulative_returns, validate_frame, volatility


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Sharpe = sqrt(A)*mean(r-rf_daily)/std(r-rf_daily)。ゼロ分散はNaN。"""
    validate_frame(returns)
    if risk_free <= -1 or periods <= 0:
        raise ValueError("Invalid annual risk-free rate or periods")
    excess = returns - ((1 + risk_free) ** (1 / periods) - 1)
    sigma = excess.std(ddof=1)
    return float(np.sqrt(periods) * excess.mean() / sigma) if sigma > 1e-15 else np.nan


def drawdown(returns: pd.Series) -> pd.Series:
    """DD_t = V_t/max(V_0,...,V_t)-1、V_0=1。初日の損失も含む。"""
    wealth = cumulative_returns(returns) + 1
    return wealth / wealth.cummax().clip(lower=1.0) - 1


def max_drawdown(returns: pd.Series) -> float:
    """最大ドローダウン min_t DD_t <= 0。"""
    return float(drawdown(returns).min())


def performance(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> pd.Series:
    """CAGR、年率変動率、Sharpe、最大DDを同一評価期間で集約する。"""
    return pd.Series(
        {
            "annual_return": annualized_return(returns, periods),
            "volatility": volatility(returns, periods),
            "sharpe": sharpe_ratio(returns, risk_free, periods),
            "max_drawdown": max_drawdown(returns),
        }
    )
