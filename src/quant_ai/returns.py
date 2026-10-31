"""価格と収益率。年率化は等間隔観測と年間252営業日を既定とする。"""

import numpy as np
import pandas as pd


def validate_frame(data: pd.Series | pd.DataFrame, *, positive: bool = False) -> None:
    """有限値・一意な昇順時点を要求。positive=Trueでは P_t > 0 を要求。"""
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Empty data, duplicate or unsorted index")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Missing/infinite values: explicitly clean data first")
    if positive and (data <= 0).to_numpy().any():
        raise ValueError("Prices must be positive")


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """単純収益率 r_t = P_t/P_{t-1}-1。初日は未定義なので除く。補完しない。"""
    validate_frame(prices, positive=True)
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """対数収益率 l_t = log(P_t/P_{t-1})。単純収益率と区別する。"""
    validate_frame(prices, positive=True)
    return np.log(prices).diff().iloc[1:]


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """単純収益率の累積 C_t = product(1+r_s)-1。r >= -1 を要求。"""
    validate_frame(returns)
    if (returns < -1).to_numpy().any():
        raise ValueError("Simple returns cannot be below -1")
    return (1 + returns).cumprod() - 1


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """幾何年率 CAGR = product(1+r)^(periods/n)-1。観測期間ベース。"""
    wealth = cumulative_returns(returns).iloc[-1] + 1
    if periods <= 0:
        raise ValueError("periods must be positive")
    return float(wealth ** (periods / len(returns)) - 1)


def volatility(returns: pd.Series, periods: int = 252) -> float:
    """標本標準偏差による年率 σ = std(r,ddof=1)*sqrt(periods)。"""
    validate_frame(returns)
    if len(returns) < 2 or periods <= 0:
        raise ValueError("At least two observations and positive periods required")
    return float(returns.std(ddof=1) * np.sqrt(periods))


def covariance_correlation(returns: pd.DataFrame, periods: int = 252) -> tuple:
    """年率標本共分散 Σ = periods*cov(r) と相関 Corr(r) を返す。"""
    validate_frame(returns)
    if len(returns) < 2 or periods <= 0:
        raise ValueError("Insufficient observations or invalid periods")
    return returns.cov() * periods, returns.corr()
