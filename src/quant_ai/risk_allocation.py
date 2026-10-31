"""過去リスクだけによる月次資産配分。全期間で重みを最適化しない。"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quant_ai.backtest import backtest
from quant_ai.metrics import performance
from quant_ai.portfolio import optimize_portfolio
from quant_ai.returns import simple_returns, validate_frame

Method = Literal["equal_weight", "inverse_vol", "min_variance", "shrink_min_variance"]
METHODS: tuple[Method, ...] = (
    "equal_weight",
    "inverse_vol",
    "min_variance",
    "shrink_min_variance",
)


@dataclass
class AllocationStudy:
    """同一期間の指標、日次運用結果、月次目標比率、推定・約定時点の監査表。"""

    summary: pd.DataFrame
    reports: dict[str, pd.DataFrame]
    targets: dict[str, pd.DataFrame]
    audits: dict[str, pd.DataFrame]


def risk_weights(history: pd.DataFrame, method: Method) -> pd.Series:
    """等比率1/N、逆σ比率(1/σ)/Σ(1/σ)、またはmin w'Σw, w>=0, Σw=1。

    historyは判断時点までに確定した日次単純リターンのみ。期待収益は推定しない。
    shrink版はLedoit-Wolfの縮小共分散。全条件でレバレッジ・空売りなし。
    """
    validate_frame(history)
    if len(history) < 10 or not history.columns.is_unique:
        raise ValueError("At least 10 observations and unique assets required")
    if method == "equal_weight":
        return pd.Series(1 / history.shape[1], index=history.columns)
    sigma = history.std(ddof=1)
    if (sigma <= 1e-12).any():
        raise ValueError("Zero-risk assets require a separately specified cash policy")
    if method == "inverse_vol":
        inverse = 1 / sigma
        return inverse / inverse.sum()
    if method == "min_variance":
        covariance = history.cov() * 252
    elif method == "shrink_min_variance":
        estimate = LedoitWolf().fit(history.to_numpy()).covariance_ * 252
        covariance = pd.DataFrame(estimate, index=history.columns, columns=history.columns)
    else:
        raise ValueError(f"Unknown allocation method: {method}")
    return optimize_portfolio(pd.Series(0.0, index=history.columns), covariance)


def monthly_targets(
    prices: pd.DataFrame,
    method: Method,
    lookback: int = 252,
    evaluation_start: str = "2006-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """月末tの直近Lリターンだけでw_tを決定。執行は次の観測日終値。

    evaluation_start以降に約定する月末判断を採用する。推定窓は固定長rolling。
    末尾の未完月の最終観測日は、次の観測日がなければ執行されず対象外。
    """
    validate_frame(prices, positive=True)
    if not isinstance(prices.index, pd.DatetimeIndex) or lookback < 10:
        raise ValueError("DatetimeIndex and lookback>=10 required")
    returns = simple_returns(prices)
    month_ends = (
        pd.Series(prices.index, index=prices.index).groupby(prices.index.to_period("M")).last()
    )
    weights, audit = [], []
    for date in month_ends:
        position = prices.index.get_loc(date)
        if position < lookback or position + 1 >= len(prices):
            continue
        execution = prices.index[position + 1]
        if execution < pd.Timestamp(evaluation_start):
            continue
        history = returns.loc[:date].iloc[-lookback:]
        w = risk_weights(history, method)
        w.name = date
        weights.append(w)
        audit.append(
            {
                "decision": date,
                "history_start": history.index[0],
                "history_end": history.index[-1],
                "observations": len(history),
                "execution": execution,
                "first_new_weight_pnl": prices.index[position + 2]
                if position + 2 < len(prices)
                else pd.NaT,
            }
        )
    if not weights:
        raise ValueError("No executable decisions after the required history")
    targets = pd.DataFrame(weights)
    targets.index.name = "decision"
    return targets, pd.DataFrame(audit).set_index("decision")


def run_allocation_study(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    evaluation_start: str = "2006-01-01",
    cost_bps: float = 5.0,
    methods: tuple[Method, ...] = METHODS,
) -> AllocationStudy:
    """同じ履歴長・月次執行・片道コストで4配分を比較するwalk-forward評価。

    Sharpeはrf=0の比較指標。CAGR=(Π(1+r_net))^(252/n)-1。
    年率turnoverはΣ売買金額/NAV/(n/252)。初回の買付も含む。
    """
    reports, all_targets, audits, rows = {}, {}, {}, []
    for method in methods:
        targets, audit = monthly_targets(prices, method, lookback, evaluation_start)
        sample = prices.loc[targets.index[0] :]
        signals = targets.reindex(sample.index).ffill()
        report = backtest(sample, signals, cost_bps=cost_bps, decision_dates=targets.index)
        if reports and not report.index.equals(next(iter(reports.values())).index):
            raise RuntimeError("Evaluation dates must match across methods")
        stats = performance(report.net_return).to_dict()
        stats.update(
            method=method,
            cumulative_return=float(report.wealth.iloc[-1] - 1),
            annual_turnover=float(report.turnover.sum() / (len(report) / 252)),
            average_target_concentration=float((targets**2).sum(axis=1).mean()),
            start=str(report.index[0].date()),
            end=str(report.index[-1].date()),
            observations=len(report),
        )
        rows.append(stats)
        reports[method], all_targets[method], audits[method] = report, targets, audit
    return AllocationStudy(pd.DataFrame(rows).set_index("method"), reports, all_targets, audits)


def period_comparison(study: AllocationStudy) -> pd.DataFrame:
    """事前に定めた固定区間でCAGR・変動率・Sharpe・最大DDを比較する。"""
    intervals = {
        "2006-2012": ("2006", "2012"),
        "2013-2019": ("2013", "2019"),
        "2020-2021": ("2020", "2021"),
        "2022": ("2022", "2022"),
        "2023-latest": ("2023", "2026"),
    }
    rows = []
    for label, (start, end) in intervals.items():
        for method, report in study.reports.items():
            returns = report.loc[start:end, "net_return"]
            if len(returns) >= 2:
                rows.append(
                    {
                        "period": label,
                        "method": method,
                        **performance(returns).to_dict(),
                        "observations": len(returns),
                    }
                )
    return pd.DataFrame(rows).set_index(["period", "method"])


def paired_block_intervals(
    study: AllocationStudy,
    repetitions: int = 2000,
    block_days: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """ペア循環ブロックbootstrapで戦略−等ウェイトの指標差の2.5/97.5%点を返す。

    既に得た戦略リターンの条件付き再標本化で、価格・重みを再推定しない。
    区間は記述的な感度分析であり、非定常レジーム下の正式な優位性検定ではない。
    """
    if repetitions < 100 or block_days < 1:
        raise ValueError("At least 100 repetitions and positive block length required")
    frame = pd.DataFrame({m: r.net_return for m, r in study.reports.items()})
    validate_frame(frame)
    baseline = frame.columns.get_loc("equal_weight")
    values = frame.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, values.shape[1], 3))
    for i in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_days)))
        indexes = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
        resampled = values[indexes]
        vol = resampled.std(axis=0, ddof=1) * np.sqrt(252)
        sharpe = resampled.mean(axis=0) * 252 / vol
        cagr = np.expm1(np.log1p(resampled).mean(axis=0) * 252)
        measures = np.column_stack([cagr, vol, sharpe])
        samples[i] = measures - measures[baseline]
    rows = []
    for j, method in enumerate(frame.columns):
        if method == "equal_weight":
            continue
        for k, metric in enumerate(["annual_return", "volatility", "sharpe"]):
            low, high = np.quantile(samples[:, j, k], [0.025, 0.975])
            rows.append({"method": method, "metric": metric, "lower_95": low, "upper_95": high})
    return pd.DataFrame(rows).set_index(["method", "metric"])
