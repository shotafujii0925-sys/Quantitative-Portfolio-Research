"""同じ市場データで配分・倍率・現金・借入・売買費用を分解する追加分析。"""

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

COMPONENTS = ("allocation", "exposure", "cash_income", "funding_cost", "extra_trading_cost")
RATES = {"rates0": (0.0, 0.0), "rates23": (0.02, 0.03), "rates46": (0.04, 0.06)}


@dataclass
class AttributionStudy:
    """同一期間の成績、反実仮想CAGR差、日次恒等式、実支払額、監査。"""

    performance: pd.DataFrame
    bridges: pd.DataFrame
    contributions: pd.DataFrame
    asset_contributions: pd.DataFrame
    payments: pd.DataFrame
    daily: dict[str, pd.DataFrame]
    provenance: dict
    audit: dict


def daily_attribution(
    prices: pd.DataFrame,
    portfolio: LeveragedResult,
    benchmark: LeveragedResult,
    unlevered: LeveragedResult,
) -> pd.DataFrame:
    """VT純リターン−非レバEW純リターンを日次で厳密分解する。

    a_s=q_s'r, a_b=q_b'r, L=実際の期首リスク資産/NAV。
    差=(a_s-a_b)+(L-1)*a_s+預金利息/NAV−借入利息/NAV−追加売買費用。
    (L-1)*a_s=(L-1)*a_b+(L-1)*(a_s-a_b)。後者の交互作用を倍率効果に含める。
    qはリスク資産内の実際の構成。約定前後の違いを避け、held列のみを使う。
    """
    actual, base, raw = portfolio.daily, benchmark.daily, unlevered.daily
    if not actual.index.equals(base.index) or not actual.index.equals(raw.index):
        raise ValueError("All paths must share evaluation dates")
    assets = list(prices.columns)
    asset_returns = prices.pct_change(fill_method=None).loc[actual.index]
    held_columns = [f"held_{asset}" for asset in assets]

    def composition(frame: pd.DataFrame) -> np.ndarray:
        """現金だけの初日はq=0。通常はリスク資産の内訳の合計を1にする。"""
        held = frame[held_columns].to_numpy()
        total = held.sum(axis=1, keepdims=True)
        return np.divide(held, total, out=np.zeros_like(held), where=total > 1e-12)

    q_actual, q_raw, q_base = composition(actual), composition(raw), composition(base)
    # 同じ月次配分・約定日なら倍率や費用を変えてもリスク資産内構成は変わらない。
    np.testing.assert_allclose(q_actual, q_raw, atol=1e-11)
    returns = asset_returns.to_numpy()
    allocation_assets = (q_raw - q_base) * returns
    unlevered_assets = q_raw * returns
    baseline_assets = q_base * returns
    exposure_assets = (actual.held_leverage.to_numpy() - 1)[:, None] * unlevered_assets
    result = pd.DataFrame(index=actual.index)
    result["allocation"] = allocation_assets.sum(axis=1)
    result["exposure"] = exposure_assets.sum(axis=1)
    result["exposure_on_benchmark"] = (actual.held_leverage - 1) * baseline_assets.sum(axis=1)
    result["interaction_in_exposure"] = (actual.held_leverage - 1) * result.allocation
    result["cash_income"] = actual.cash_income_return - base.cash_income_return
    result["funding_cost"] = -actual.funding_cost_return + base.funding_cost_return
    result["extra_trading_cost"] = -actual.trade_cost_return + base.trade_cost_return
    result["active_return"] = actual.net_return - base.net_return
    result["portfolio_return"] = actual.net_return
    result["benchmark_return"] = base.net_return
    for j, asset in enumerate(assets):
        result[f"allocation_{asset}"] = allocation_assets[:, j]
        result[f"exposure_{asset}"] = exposure_assets[:, j]
    result["residual"] = result[list(COMPONENTS)].sum(axis=1) - result.active_return
    if result.residual.abs().max() > 1e-11:
        raise RuntimeError("Daily attribution does not reconcile")
    np.testing.assert_allclose(
        result.exposure, result.exposure_on_benchmark + result.interaction_in_exposure, atol=1e-11
    )
    return result


def annual_contributions(daily: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """算術年率差と年率対数差への寄与を返す。後者はCAGRポイントではない。

    k_t=[log(1+rp)-log(1+rb)]/(rp-rb)、同値では1/(1+rb)。
    (A/n)*Σ k_t*c_j,t が年率対数リターン差へ厳密に加算される。
    CAGR自体の差の分解とは区別する。これは評価後の連結処理で売買判断に使わない。
    """
    rp, rb = daily.portfolio_return.to_numpy(), daily.benchmark_return.to_numpy()
    if (rp <= -1).any() or (rb <= -1).any() or not np.isfinite([rp, rb]).all():
        raise ValueError("Finite solvent returns required")
    delta = rp - rb
    coefficients = np.divide(
        np.log1p(delta / (1 + rb)), delta, out=1 / (1 + rb), where=np.abs(delta) > 1e-12
    )
    annualizer = periods / len(daily)
    columns = [*COMPONENTS, "exposure_on_benchmark", "interaction_in_exposure"]
    result = pd.DataFrame(
        {
            "arithmetic_annual_contribution": daily[columns].sum() * annualizer,
            "annual_log_contribution": daily[columns].mul(coefficients, axis=0).sum() * annualizer,
        }
    )
    expected = float(np.log1p(rp).sum() - np.log1p(rb).sum()) * annualizer
    if abs(result.loc[list(COMPONENTS), "annual_log_contribution"].sum() - expected) > 1e-10:
        raise RuntimeError("Annual log attribution does not reconcile")
    return result


def cagr_bridge(
    benchmark_gross: LeveragedResult,
    benchmark_net: LeveragedResult,
    raw_gross: LeveragedResult,
    vt_gross: LeveragedResult,
    vt_zero_rate_net: LeveragedResult,
    vt_borrow_only: LeveragedResult,
    vt_final: LeveragedResult,
) -> dict:
    """反実仮想経路のCAGR差を固定順序で繋ぐ。独立な因果効果とは呼ばない。

    配分=raw_s無費用−EW無費用、倍率=VT無費用−raw_s無費用。
    追加売買費用=(VT費用あり−VT無費用)−(EW費用あり−EW無費用)。
    借入=(預金0/借入b)−金利0、預金=(預金c/借入b)−(預金0/借入b)。
    合計はVT最終CAGR−EW非レバ費用後CAGR。金利→倍率等の順なら寄与は変わり得る。
    """
    paths = [
        benchmark_gross,
        benchmark_net,
        raw_gross,
        vt_gross,
        vt_zero_rate_net,
        vt_borrow_only,
        vt_final,
    ]
    first_index = paths[0].daily.index
    if any(not result.daily.index.equals(first_index) for result in paths):
        raise ValueError("CAGR bridge paths must use identical dates")
    bg, bn, rg, vg, vz, vb, vf = [
        float(performance(p.daily.net_return).annual_return) for p in paths
    ]
    components = {
        "allocation": rg - bg,
        "exposure": vg - rg,
        "extra_trading_cost": (vz - vg) - (bn - bg),
        "funding_cost": vb - vz,
        "cash_income": vf - vb,
    }
    total = sum(components.values())
    if abs(total - (vf - bn)) > 1e-12:
        raise RuntimeError("CAGR path bridge does not reconcile")
    return {
        **components,
        "total_cagr_difference": total,
        "benchmark_net_cagr": bn,
        "portfolio_net_cagr": vf,
        "residual": total - (vf - bn),
    }


def actual_payments(result: LeveragedResult, prices: pd.DataFrame) -> pd.Series:
    """実経路の資産損益・預金利息・借入支払・取引費用を初期資金1当たりで集計。

    借入実支払額と、借入なしの反実仮想との差は複利・NAV経路が違うため一致しない。
    """
    daily = result.daily
    previous_nav = daily.wealth.shift(1, fill_value=1.0)
    returns = prices.pct_change(fill_method=None).loc[daily.index]
    parts = {
        f"asset_pnl_{asset}": float((daily[f"held_{asset}"] * previous_nav * returns[asset]).sum())
        for asset in prices
    }
    parts.update(
        cash_income=float(daily.cash_income.sum()),
        funding_cost=-float(daily.funding_cost.sum()),
        trading_cost=-float(daily.trade_cost.sum()),
    )
    total = float(sum(parts.values()))
    residual = total - float(daily.wealth.iloc[-1] - 1)
    if abs(residual) > 1e-10:
        raise RuntimeError("Actual dollar P&L does not reconcile")
    return pd.Series(
        {
            **parts,
            "terminal_wealth": float(daily.wealth.iloc[-1]),
            "total_pnl": total,
            "residual": residual,
        }
    )


def run_attribution_study(root: Path) -> AttributionStudy:
    """元データと現在のVTを再現し、金利0・2/3・4/6%を同じ期間で分解する。"""
    path = root / "data/processed/spy_tlt_gld_adjusted.parquet"
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Price snapshot hash changed")
    prices = pd.read_parquet(path)
    raw_net, raw_gross, targets_by_case = {}, {}, {}
    results, rows, bridges, contribution_rows, asset_rows, payments_rows, daily = (
        {},
        [],
        [],
        [],
        [],
        [],
        {},
    )
    audit = {
        "legacy_max_daily_difference": 0.0,
        "max_daily_residual": 0.0,
        "max_log_residual": 0.0,
        "max_cagr_bridge_residual": 0.0,
    }
    for method in CORE_METHODS:
        base, base_audit = monthly_targets(prices, method)
        sample = prices.loc[base.index[0] :]
        raw_net[method] = leveraged_backtest(sample, base)
        raw_gross[method] = leveraged_backtest(sample, base, cost_bps=0)
        for cap, label in [(1.5, "cap15"), (None, "unbounded")]:
            targets, _ = volatility_schedule(
                prices, method, leverage_cap=cap, base_targets=base, base_audit=base_audit
            )
            targets_by_case[(label, method)] = targets
        stats = evaluate_leveraged(raw_net[method], None, None)
        rows.append(
            {
                "case_id": f"raw_{method}",
                "family": "unlevered",
                "method": method,
                "mode": "raw",
                "rate_scenario": "rates0",
                "cash_rate": 0.0,
                "borrow_rate": 0.0,
                **stats,
            }
        )
        results[f"raw_{method}"] = raw_net[method]
    benchmark, benchmark_gross = raw_net["equal_weight"], raw_gross["equal_weight"]
    for label, cap in [("cap15", 1.5), ("unbounded", None)]:
        for method in CORE_METHODS:
            targets = targets_by_case[(label, method)]
            sample = prices.loc[targets.index[0] :]
            zero_net = leveraged_backtest(sample, targets)
            zero_gross = leveraged_backtest(sample, targets, cost_bps=0)
            for rate_name, (cash, borrow) in RATES.items():
                print(f"Attributing {label} / {method} / {rate_name}", flush=True)
                case_id = f"{label}_{rate_name}_{method}"
                borrow_only = (
                    zero_net
                    if borrow == 0
                    else leveraged_backtest(sample, targets, borrow_rate=borrow)
                )
                final = (
                    zero_net
                    if cash == 0 and borrow == 0
                    else leveraged_backtest(sample, targets, cash_rate=cash, borrow_rate=borrow)
                )
                stats = evaluate_leveraged(final, 0.10, cap)
                rows.append(
                    {
                        "case_id": case_id,
                        "family": "vol_target",
                        "method": method,
                        "mode": label,
                        "rate_scenario": rate_name,
                        "cash_rate": cash,
                        "borrow_rate": borrow,
                        **stats,
                    }
                )
                results[case_id] = final
                pieces = daily_attribution(prices, final, benchmark, raw_net[method])
                daily[case_id] = pieces
                annual = annual_contributions(pieces)
                for component, values in annual.iterrows():
                    contribution_rows.append(
                        {"case_id": case_id, "component": component, **values.to_dict()}
                    )
                # 同じ連結係数を資産別の配分・倍率効果にも適用する。
                rp, rb = pieces.portfolio_return.to_numpy(), pieces.benchmark_return.to_numpy()
                difference = rp - rb
                coefficients = np.divide(
                    np.log1p(difference / (1 + rb)),
                    difference,
                    out=1 / (1 + rb),
                    where=np.abs(difference) > 1e-12,
                )
                for component in ["allocation", "exposure"]:
                    for asset in prices:
                        values = pieces[f"{component}_{asset}"]
                        asset_rows.append(
                            {
                                "case_id": case_id,
                                "component": component,
                                "asset": asset,
                                "arithmetic_annual_contribution": values.mean() * 252,
                                "annual_log_contribution": float(
                                    (values * coefficients).mean() * 252
                                ),
                            }
                        )
                bridge = cagr_bridge(
                    benchmark_gross,
                    benchmark,
                    raw_gross[method],
                    zero_gross,
                    zero_net,
                    borrow_only,
                    final,
                )
                bridges.append(
                    {
                        "case_id": case_id,
                        "method": method,
                        "mode": label,
                        "rate_scenario": rate_name,
                        **bridge,
                    }
                )
                payments_rows.append(
                    {"case_id": case_id, **actual_payments(final, prices).to_dict()}
                )
                audit["max_daily_residual"] = max(
                    audit["max_daily_residual"], pieces.residual.abs().max()
                )
                audit["max_cagr_bridge_residual"] = max(
                    audit["max_cagr_bridge_residual"], abs(bridge["residual"])
                )
                target_log = np.log1p(rp).mean() * 252 - np.log1p(rb).mean() * 252
                audit["max_log_residual"] = max(
                    audit["max_log_residual"],
                    abs(annual.loc[list(COMPONENTS), "annual_log_contribution"].sum() - target_log),
                )
                if rate_name == "rates0":
                    old = pd.read_parquet(
                        root / f"outputs/reports/リスク統一比較/daily_{label}_{method}.parquet"
                    )
                    if not old.index.equals(final.daily.index):
                        raise RuntimeError("The current volatility targeting dates changed")
                    delta = float(np.max(np.abs(old.net_return - final.daily.net_return)))
                    audit["legacy_max_daily_difference"] = max(
                        audit["legacy_max_daily_difference"], delta
                    )
                    if delta > 1e-11:
                        raise RuntimeError("The current volatility targeting result changed")
    output = root / "outputs/reports/戦略比較と収益分解"
    output.mkdir(parents=True, exist_ok=True)
    for case_id, result in results.items():
        result.daily.to_parquet(output / f"path_{case_id}.parquet")
    bundle = AttributionStudy(
        pd.DataFrame(rows).set_index("case_id"),
        pd.DataFrame(bridges).set_index("case_id"),
        pd.DataFrame(contribution_rows),
        pd.DataFrame(asset_rows),
        pd.DataFrame(payments_rows).set_index("case_id"),
        daily,
        provenance,
        audit,
    )
    for name, frame in [
        ("performance", bundle.performance),
        ("cagr_bridge", bundle.bridges),
        ("annual_contributions", bundle.contributions),
        ("asset_contributions", bundle.asset_contributions),
        ("actual_payments", bundle.payments),
    ]:
        frame.to_csv(
            output / f"{name}.csv",
            index=name not in {"annual_contributions", "asset_contributions"},
        )
    for case_id, frame in daily.items():
        frame.to_parquet(output / f"attribution_{case_id}.parquet")
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle


def load_attribution_study(root: Path) -> AttributionStudy:
    """保存結果をNotebook・描画・報告に読み込む。計算の実装は同じモジュール内。"""
    directory = root / "outputs/reports/戦略比較と収益分解"
    performance_frame = pd.read_csv(directory / "performance.csv", index_col=0)
    daily = {
        case_id: pd.read_parquet(directory / f"attribution_{case_id}.parquet")
        for case_id in performance_frame.index[performance_frame.family == "vol_target"]
    }
    return AttributionStudy(
        performance_frame,
        pd.read_csv(directory / "cagr_bridge.csv", index_col=0),
        pd.read_csv(directory / "annual_contributions.csv"),
        pd.read_csv(directory / "asset_contributions.csv"),
        pd.read_csv(directory / "actual_payments.csv", index_col=0),
        daily,
        json.loads((directory / "provenance.json").read_text(encoding="utf-8")),
        json.loads((directory / "audit.json").read_text(encoding="utf-8")),
    )
