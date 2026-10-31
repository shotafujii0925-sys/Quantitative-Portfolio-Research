"""リスク統一の追加研究。旧成果を読み取り専用で照合し、新出力先へ保存する。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.downside_metrics import evaluate_leveraged
from quant_ai.leveraged_backtest import LeveragedResult, leveraged_backtest
from quant_ai.metrics import performance
from quant_ai.risk_allocation import monthly_targets
from quant_ai.volatility_target import CORE_METHODS, volatility_schedule


@dataclass
class RiskResearch:
    """全条件の集計、基本ケースの日次/監査、固定区間、条件付き区間、データ出典。"""

    summary: pd.DataFrame
    daily: dict[str, LeveragedResult]
    audits: dict[str, pd.DataFrame]
    periods: pd.DataFrame
    bootstrap: pd.DataFrame
    provenance: dict
    reconciliation: dict


def conditional_intervals(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str]],
    block: int = 21,
    repetitions: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """同じ循環ブロックで全戦略を再標本化する条件付き差の区間。重み再推定なし。

    順にCAGR、年率変動率、Sharpe(rf=0)、最大DD。多重比較補正や非定常性対処はしない。
    """
    if block < 1 or repetitions < 100 or len(frame) < 2:
        raise ValueError("Positive block, at least 100 repetitions and 2 observations required")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Finite solvent returns required")
    n, columns = values.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, columns, 4))
    for i in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        ix = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        sample = values[ix]
        log_wealth = np.log1p(sample).cumsum(axis=0)
        cagr = np.expm1(log_wealth[-1] * 252 / n)
        vol = sample.std(axis=0, ddof=1) * np.sqrt(252)
        sharpe = sample.mean(axis=0) * 252 / vol
        peak = np.maximum.accumulate(np.maximum(log_wealth, 0), axis=0)
        mdd = np.expm1(log_wealth - peak).min(axis=0)
        draws[i] = np.column_stack([cagr, vol, sharpe, mdd])
    rows = []
    for strategy, baseline in pairs:
        differences = (
            draws[:, frame.columns.get_loc(strategy)] - draws[:, frame.columns.get_loc(baseline)]
        )
        for k, metric in enumerate(["annual_return", "volatility", "sharpe", "max_drawdown"]):
            lo, hi = np.quantile(differences[:, k], [0.025, 0.975])
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": baseline,
                    "metric": metric,
                    "block_days": block,
                    "repetitions": repetitions,
                    "lower_95": lo,
                    "upper_95": hi,
                }
            )
    return pd.DataFrame(rows)


def run_risk_research(root: Path, *, repetitions: int = 2000) -> RiskResearch:
    """元データを保持し、目標・上限・金利・コスト・推定窓を固定した全条件を実行。"""
    path = root / "data/processed/spy_tlt_gld_adjusted.parquet"
    prices = pd.read_parquet(path)
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Price snapshot hash differs from the recorded source")
    base_cache, run_cache = {}, {}
    rows, basic_daily, basic_audits = [], {}, {}

    def case(
        case_id: str,
        family: str,
        method: str,
        target: float | None = 0.10,
        cap: float | None = 1.5,
        lookback: int = 252,
        start: str = "2006-01-01",
        cash: float = 0.0,
        borrow: float = 0.0,
        cost: float = 5.0,
    ) -> None:
        """既知条件の計算を再利用し、比較群とパラメータを必ず集計へ記録。"""
        base_key = (method, lookback)
        if base_key not in base_cache:
            base_cache[base_key] = monthly_targets(prices, method, lookback, "2006-01-01")
        key = (method, target, cap, lookback, start, cash, borrow, cost)
        if key not in run_cache:
            base, audit_base = base_cache[base_key]
            targets, audit = volatility_schedule(
                prices,
                method,
                lookback=lookback,
                evaluation_start=start,
                target_volatility=target,
                leverage_cap=cap,
                base_targets=base,
                base_audit=audit_base,
            )
            sample = prices.loc[targets.index[0] :]
            result = leveraged_backtest(
                sample, targets, cost_bps=cost, cash_rate=cash, borrow_rate=borrow
            )
            zero_cost = (
                result
                if cost == 0
                else leveraged_backtest(
                    sample, targets, cost_bps=0, cash_rate=cash, borrow_rate=borrow
                )
            )
            stats = evaluate_leveraged(result, target, cap)
            stats.update(
                gross_path_annual_return=performance(zero_cost.daily.net_return).annual_return,
                gross_path_volatility=performance(zero_cost.daily.net_return).volatility,
                forecast_mean=float(audit.predicted_volatility.mean()),
                cap_binding_month_fraction=float(audit.cap_binding.mean()),
            )
            stats["cost_cagr_drag"] = stats["gross_path_annual_return"] - stats["annual_return"]
            run_cache[key] = (result, audit, stats)
        result, audit, stats = run_cache[key]
        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "method": method,
                "target_volatility": target,
                "leverage_cap": cap,
                "lookback": lookback,
                "cash_rate": cash,
                "borrow_rate": borrow,
                "cost_bps": cost,
                **stats,
            }
        )
        if family in {"original", "primary"}:
            basic_daily[case_id], basic_audits[case_id] = result, audit

    for method in CORE_METHODS:
        case(f"original_{method}", "original", method, None, None)
        for cap, label in [(None, "unbounded"), (1.5, "cap15")]:
            case(f"{label}_{method}", "primary", method, cap=cap)
    print("Primary scenarios complete", flush=True)
    for target in [0.08, 0.10, 0.12]:
        for cap, label in [(None, "unbounded"), (1.5, "cap15")]:
            for method in CORE_METHODS:
                case(f"target{target}_{label}_{method}", "target_sensitivity", method, target, cap)
    for window in [126, 252, 504]:
        for cap, label in [(None, "unbounded"), (1.5, "cap15")]:
            for method in CORE_METHODS:
                case(
                    f"window{window}_{label}_{method}",
                    "lookback_sensitivity",
                    method,
                    cap=cap,
                    lookback=window,
                    start="2007-01-01",
                )
    for cash, borrow in [(0.0, 0.0), (0.02, 0.03), (0.04, 0.06)]:
        for cap, label in [(None, "unbounded"), (1.5, "cap15")]:
            for method in CORE_METHODS:
                case(
                    f"funding{cash}_{borrow}_{label}_{method}",
                    "funding_sensitivity",
                    method,
                    cap=cap,
                    cash=cash,
                    borrow=borrow,
                )
    for cost in [0.0, 5.0, 20.0]:
        for cap, label in [(None, "unbounded"), (1.5, "cap15")]:
            for method in CORE_METHODS:
                case(f"cost{cost}_{label}_{method}", "cost_sensitivity", method, cap=cap, cost=cost)
    print(f"Sensitivity complete: {len(run_cache)} distinct paths", flush=True)
    summary = pd.DataFrame(rows).set_index("case_id")
    reconciliation = {}
    for method in CORE_METHODS:
        old_path = root / f"分析結果/レポート/資産配分比較/daily_{method}.csv"
        old = pd.read_csv(old_path, index_col=0, parse_dates=True)
        new = basic_daily[f"original_{method}"].daily
        if not old.index.equals(new.index):
            raise ValueError("Legacy evaluation dates differ")
        delta = float(np.max(np.abs(old.net_return - new.net_return)))
        reconciliation[method] = {
            "daily_return_max_abs_difference": delta,
            "wealth_max_abs_difference": float(np.max(np.abs(old.wealth - new.wealth))),
        }
        if delta > 1e-10:
            raise ValueError(f"Legacy parity failure for {method}: {delta}")
    intervals = {
        "2006-2012": ("2006", "2012"),
        "2013-2019": ("2013", "2019"),
        "2020-2021": ("2020", "2021"),
        "2022": ("2022", "2022"),
        "2023-latest": ("2023", "2026"),
    }
    periods = []
    for name, result in basic_daily.items():
        for period, (start, end) in intervals.items():
            returns = result.daily.loc[start:end, "net_return"]
            periods.append(
                {
                    "case_id": name,
                    "period": period,
                    **performance(returns).to_dict(),
                    "period_total_return": float((1 + returns).prod() - 1),
                }
            )
    frame = pd.DataFrame(
        {
            name: result.daily.net_return
            for name, result in basic_daily.items()
            if not name.startswith("original")
        }
    )
    pairs = [
        (f"{cap}_{method}", f"{cap}_equal_weight")
        for cap in ["unbounded", "cap15"]
        for method in CORE_METHODS[1:]
    ]
    bootstrap = pd.concat(
        [
            conditional_intervals(frame, pairs, block=block, repetitions=repetitions)
            for block in [21, 63]
        ],
        ignore_index=True,
    )
    provenance.update(
        research_is_post_selection=True,
        bootstrap_reestimates_weights=False,
        distinct_paths=len(run_cache),
        reported_scenarios=len(summary),
    )
    return RiskResearch(
        summary,
        basic_daily,
        basic_audits,
        pd.DataFrame(periods),
        bootstrap,
        provenance,
        reconciliation,
    )


def save_risk_research(bundle: RiskResearch, root: Path) -> None:
    """指定のoutputs/reportsへ出力。既存の分析結果/は上書きしない。"""
    output = root / "outputs/reports/リスク統一比較"
    output.mkdir(parents=True, exist_ok=True)
    bundle.summary.to_csv(output / "all_scenarios.csv")
    bundle.summary[bundle.summary.family.isin(["original", "primary"])].to_csv(
        output / "primary.csv"
    )
    bundle.periods.to_csv(output / "subperiods.csv", index=False)
    bundle.bootstrap.to_csv(output / "conditional_bootstrap.csv", index=False)
    for case_id, result in bundle.daily.items():
        result.daily.to_parquet(output / f"daily_{case_id}.parquet")
        bundle.audits[case_id].to_csv(output / f"audit_{case_id}.csv")
    (output / "provenance.json").write_text(
        json.dumps(bundle.provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "legacy_reconciliation.json").write_text(
        json.dumps(bundle.reconciliation, indent=2), encoding="utf-8"
    )


def load_risk_research(root: Path) -> RiskResearch:
    """保存済みの全結果をNotebookへ読み込む。実装と実行セルは別途Notebookに表示する。"""
    output = root / "outputs/reports/リスク統一比較"
    summary = pd.read_csv(output / "all_scenarios.csv", index_col=0)
    daily, audits = {}, {}
    for case_id in summary.index[summary.family.isin(["original", "primary"])]:
        data = pd.read_parquet(output / f"daily_{case_id}.parquet")
        audit = pd.read_csv(output / f"audit_{case_id}.csv", index_col=0, parse_dates=[0])
        first_decision = audit.index[0]
        daily[case_id] = LeveragedResult(data, first_decision)
        audits[case_id] = audit
    return RiskResearch(
        summary,
        daily,
        audits,
        pd.read_csv(output / "subperiods.csv"),
        pd.read_csv(output / "conditional_bootstrap.csv"),
        json.loads((output / "provenance.json").read_text(encoding="utf-8")),
        json.loads((output / "legacy_reconciliation.json").read_text(encoding="utf-8")),
    )
