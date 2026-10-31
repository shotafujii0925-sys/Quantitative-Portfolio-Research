"""制約付き平均分散法。入力μとΣは同じ年率単位、空売り禁止を既定とする。"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_portfolio(
    mean: pd.Series,
    covariance: pd.DataFrame,
    *,
    objective: str = "min_variance",
    target_return: float | None = None,
    risk_free: float = 0.0,
) -> pd.Series:
    """min w'Σw または max (w'μ-rf)/sqrt(w'Σw)、1'w=1, 0<=w<=1。

    max_sharpeは接点ポートフォリオ。rfはμと同じ算術年率単位。
    幾何年率rfを使う場合は252*((1+rf)^(1/252)-1)へ変換する。
    """
    if objective not in {"min_variance", "max_sharpe"}:
        raise ValueError("Unknown objective")
    if not mean.index.is_unique or mean.empty:
        raise ValueError("Unique assets required")
    cov = covariance.loc[mean.index, mean.index].to_numpy(dtype=float)
    mu = mean.to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.isfinite(mu).all():
        raise ValueError("Non-finite estimates")
    if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance must be symmetric positive semidefinite")
    if objective == "max_sharpe" and mu.max() <= risk_free:
        raise ValueError("No positive excess-return asset; use cash or revise assumptions")
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        if not mu.min() - 1e-10 <= target_return <= mu.max() + 1e-10:
            raise ValueError("Target return outside long-only feasible set")
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    def loss(weights: np.ndarray) -> float:
        """最小化用の分散、または符号反転Sharpe。"""
        var = float(weights @ cov @ weights)
        if objective == "min_variance":
            return var
        return -float((weights @ mu - risk_free) / np.sqrt(max(var, 1e-16)))

    result = minimize(
        loss,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    weights = pd.Series(result.x, index=mean.index, name="weight")
    if abs(weights.sum() - 1) > 1e-6:
        raise RuntimeError("Budget constraint failed")
    if target_return is not None and abs(weights @ mean - target_return) > 1e-6:
        raise RuntimeError("Return constraint failed")
    return weights


def efficient_frontier(mean: pd.Series, covariance: pd.DataFrame, points: int = 20) -> pd.DataFrame:
    """大域最小分散の期待収益からmax μまでの上側フロンティアを計算。"""
    if points < 2:
        raise ValueError("points >= 2 required")
    minimum = optimize_portfolio(mean, covariance)
    rows = []
    for target in np.linspace(float(minimum @ mean), float(mean.max()), points):
        weights = optimize_portfolio(mean, covariance, target_return=float(target))
        rows.append(
            {
                "return": float(weights @ mean),
                "volatility": float(np.sqrt(weights @ covariance @ weights)),
                **weights.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def cash_allocation(risky_weights: pd.Series, risky_fraction: float) -> pd.Series:
    """無リスク資産との配分 w_cash=1-y, w_risky=y*w。レバレッジなし。"""
    if not 0 <= risky_fraction <= 1 or not np.isclose(risky_weights.sum(), 1):
        raise ValueError("Invalid allocation")
    return pd.concat([risky_weights * risky_fraction, pd.Series({"CASH": 1 - risky_fraction})])
