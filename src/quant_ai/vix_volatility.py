"""VIXでSPYのボラを予測するボラティリティ目標。期待収益は推定しない。

時点の約束: 判断日tの情報はtの終値まで。約定はt+1終値、新保有の損益はt+1→t+2から。
VIXの確定は指数より遅いが、t+1終値の約定より前なので先読みにならない。
"""

from typing import Literal

import numpy as np
import pandas as pd

from quant_ai.portfolio import optimize_portfolio
from quant_ai.returns import simple_returns, validate_frame
from quant_ai.risk_allocation import monthly_targets

Forecast = Literal["none", "hist252", "rv21", "vix_cal", "vix_raw"]
Base = Literal["spy_only", "equal_weight", "min_variance"]
FORECASTS: tuple[Forecast, ...] = ("none", "hist252", "rv21", "vix_cal", "vix_raw")
BASES: tuple[Base, ...] = ("spy_only", "equal_weight", "min_variance")
HORIZON = 21
SHORT_WINDOW = 21


def align_vix(vix: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """価格の観測日へ揃える。欠損は補完せず残し、判断日に欠損があれば後段で停止する。"""
    if not isinstance(vix.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex is required")
    if not vix.index.is_unique or (vix.dropna() <= 0).any():
        raise ValueError("Unique positive VIX observations required")
    return vix.reindex(prices.index).rename("VIX")


def forward_realized_vol(spy_returns: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """日iの値 = 年率std(r[i+2],...,r[i+1+horizon])。確定はi+1+horizon日の終値。

    判断日iの新保有がt+1→t+2から損益を持つ期間に合わせる。ラベル専用で判断に直接使わない。
    """
    if horizon < 2:
        raise ValueError("Horizon must be at least 2")
    rolling = spy_returns.rolling(horizon).std(ddof=1) * np.sqrt(252)
    return rolling.shift(-(horizon + 1))


def calibration_level(
    spy_returns: pd.Series,
    vix: pd.Series,
    horizon: int = HORIZON,
) -> pd.Series:
    """各日tのk_t = mean(log(RV_i / (VIX_i/100)))。i+1+horizon <= t の確定分だけを使う。

    log(RV)=k+log(VIX) で傾きを1に固定し、水準だけを拡張窓で推定する。確定分がなければNaN。
    """
    log_ratio = np.log(forward_realized_vol(spy_returns, horizon)) - np.log(vix / 100)
    known = log_ratio.shift(horizon + 1)
    level = known.expanding(min_periods=1).mean()
    return level.rename("k")


def spy_volatility(
    forecast: Forecast,
    position: int,
    spy_returns: pd.Series,
    vix: pd.Series,
    level: pd.Series,
    lookback: int = 252,
) -> float | None:
    """判断位置positionでのSPYの年率ボラ予測。hist252はNoneを返し共分散をそのまま使う。"""
    if forecast == "hist252":
        return None
    if forecast == "rv21":
        window = spy_returns.iloc[position - SHORT_WINDOW + 1 : position + 1]
        return float(window.std(ddof=1) * np.sqrt(252))
    if forecast not in ("vix_cal", "vix_raw"):
        raise ValueError(f"Unknown forecast: {forecast}")
    value = float(vix.iloc[position])
    if not np.isfinite(value):
        raise ValueError(f"VIX is missing on decision date {vix.index[position]}")
    if forecast == "vix_raw":
        return value / 100
    k = float(level.iloc[position])
    if not np.isfinite(k):
        raise ValueError(f"No completed calibration data on {vix.index[position]}")
    return value / 100 * float(np.exp(k))


def covariance_with_spy_vol(history: pd.DataFrame, spy_vol: float | None) -> pd.DataFrame:
    """年率共分散 Σ = D R D。RとSPY以外のDは履歴、SPYのDだけをspy_volへ置換する。"""
    covariance = history.cov() * 252
    if spy_vol is None:
        return covariance
    if spy_vol <= 0 or "SPY" not in history.columns:
        raise ValueError("Positive SPY volatility and a SPY column required")
    scale = pd.Series(1.0, index=history.columns)
    scale["SPY"] = spy_vol / np.sqrt(covariance.loc["SPY", "SPY"])
    return covariance.mul(scale, axis=0).mul(scale, axis=1)


def decision_dates(prices: pd.DataFrame, frequency: Literal["M", "W"] = "M") -> pd.DatetimeIndex:
    """各月または各週（金曜区切り）の最後の観測日。次の観測日がない日は除く。"""
    if frequency not in ("M", "W"):
        raise ValueError("Frequency must be M or W")
    period = "M" if frequency == "M" else "W-FRI"
    last = (
        pd.Series(prices.index, index=prices.index).groupby(prices.index.to_period(period)).last()
    )
    return pd.DatetimeIndex(last[last < prices.index[-1]].to_numpy())


def base_weight_schedule(
    prices: pd.DataFrame,
    base: Base,
    lookback: int,
    evaluation_start: str,
) -> pd.DataFrame:
    """月末判断の基本配分。spy_onlyは常にSPY100%、他は過去lookback日の既存の配分規則。"""
    targets, _ = monthly_targets(
        prices,
        "equal_weight" if base == "spy_only" else base,
        lookback,
        evaluation_start,
    )
    if base != "spy_only":
        return targets
    weights = pd.DataFrame(0.0, index=targets.index, columns=prices.columns)
    weights["SPY"] = 1.0
    return weights


def vix_schedule(
    prices: pd.DataFrame,
    vix: pd.Series,
    base: Base,
    forecast: Forecast,
    *,
    lookback: int = 252,
    evaluation_start: str = "2006-01-01",
    target_volatility: float = 0.10,
    leverage_cap: float | None = 1.5,
    frequency: Literal["M", "W"] = "M",
    reallocate: bool = False,
    base_targets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """b_tと予測Σ_tから σhat=sqrt(b'Σ_t b)、L=min(目標/σhat, 上限)、w=L*b を返す。

    週次判断の基本配分は判断日以前で直近の月末の値を維持する。
    reallocate=Trueなら最小分散のウェイトを判断日の予測Σで求め直す（月次のみ）。
    forecast="none" は倍率1で基本配分をそのまま持つ。
    """
    validate_frame(prices, positive=True)
    if "SPY" not in prices.columns or target_volatility <= 0:
        raise ValueError("SPY column and positive target volatility required")
    if leverage_cap is not None and leverage_cap <= 0:
        raise ValueError("Leverage cap must be positive")
    if reallocate and (base != "min_variance" or frequency != "M"):
        raise ValueError("Reallocation is defined only for monthly minimum variance")
    if not vix.index.equals(prices.index):
        raise ValueError("VIX must be aligned with align_vix first")
    returns = simple_returns(prices)
    spy_returns = returns["SPY"].reindex(prices.index)
    level = calibration_level(spy_returns, vix)
    monthly = (
        base_weight_schedule(prices, base, lookback, evaluation_start)
        if base_targets is None
        else base_targets
    )
    decisions = decision_dates(prices, frequency)
    decisions = decisions[decisions >= monthly.index[0]]
    if frequency == "M":
        decisions = decisions.intersection(monthly.index)
    weights = monthly.reindex(decisions, method="ffill")
    rows, audit = [], []
    for date in decisions:
        position = prices.index.get_loc(date)
        history = returns.loc[:date].iloc[-lookback:]
        if len(history) != lookback:
            raise ValueError("Insufficient history")
        spy_vol = (
            None
            if forecast == "none"
            else spy_volatility(forecast, position, spy_returns, vix, level, lookback)
        )
        covariance = covariance_with_spy_vol(history, spy_vol)
        weight = weights.loc[date]
        if reallocate:
            weight = optimize_portfolio(pd.Series(0.0, index=history.columns), covariance)
        vector = weight.to_numpy(dtype=float)
        sigma = float(np.sqrt(vector @ covariance.to_numpy() @ vector))
        if sigma <= 1e-12:
            raise ValueError("Predicted zero volatility: exposure is undefined")
        requested = 1.0 if forecast == "none" else target_volatility / sigma
        exposure = (
            requested
            if leverage_cap is None or forecast == "none"
            else min(requested, leverage_cap)
        )
        rows.append(pd.Series(vector * exposure, index=prices.columns, name=date))
        audit.append(
            {
                "decision": date,
                "execution": prices.index[position + 1],
                "vix": float(vix.iloc[position]),
                "spy_vol_forecast": np.nan if spy_vol is None else spy_vol,
                "unscaled_predicted_volatility": sigma,
                "requested_exposure": requested,
                "target_exposure": exposure,
                "cap_binding": exposure < requested - 1e-12,
            }
        )
    targets = pd.DataFrame(rows)
    targets.index.name = "decision"
    return targets, pd.DataFrame(audit).set_index("decision")


def forecast_table(prices: pd.DataFrame, vix: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """月末判断日ごとのSPYボラ予測と、約定後21日の実現ボラ。結果が未確定の日は除く。"""
    returns = simple_returns(prices)["SPY"].reindex(prices.index)
    level = calibration_level(returns, vix)
    realized = forward_realized_vol(returns)
    rows = []
    for date in decision_dates(prices, "M"):
        position = prices.index.get_loc(date)
        if position < lookback or not np.isfinite(realized.iloc[position]):
            continue
        history = returns.iloc[position - lookback + 1 : position + 1]
        row = {"decision": date, "realized": float(realized.iloc[position])}
        row["hist252"] = float(history.std(ddof=1) * np.sqrt(252))
        for name in ("rv21", "vix_cal", "vix_raw"):
            row[name] = spy_volatility(name, position, returns, vix, level, lookback)
        rows.append(row)
    return pd.DataFrame(rows).set_index("decision")


def qlike(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """QLIKE損失 = x - log(x) - 1、x=実現分散/予測分散。予測が完全なら0、外れるほど大きい。"""
    ratio = (realized / forecast) ** 2
    return ratio - np.log(ratio) - 1


def rescaled_cagr(returns: pd.Series, target: float = 0.10) -> float:
    """純収益をk=目標/実現ボラ倍した系列のCAGR。事後の記述統計で売買可能性を意味しない。"""
    validate_frame(returns)
    volatility = returns.std(ddof=1) * np.sqrt(252)
    if volatility <= 0:
        raise ValueError("Positive volatility required")
    scaled = returns * target / volatility
    if (scaled <= -1).any():
        raise ValueError("Scaled path is insolvent")
    return float(np.expm1(np.log1p(scaled).mean() * 252))
