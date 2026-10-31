"""事前に定めた比較研究を実行し、再現用CSV・JSON・図を保存する。"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_ai.metrics import drawdown, performance
from quant_ai.returns import simple_returns
from quant_ai.risk_allocation import (
    METHODS,
    paired_block_intervals,
    period_comparison,
    run_allocation_study,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse volatility",
    "min_variance": "Minimum variance",
    "shrink_min_variance": "Shrinkage min variance",
}
COLORS = {
    "equal_weight": "#3b4a5a",
    "inverse_vol": "#009e73",
    "min_variance": "#d55e00",
    "shrink_min_variance": "#0072b2",
}


def main() -> None:
    """実データを読み、主分析・感度分析・条件付きbootstrapを保存する。"""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt

    path = ROOT / "data/processed/spy_tlt_gld_adjusted.parquet"
    prices = pd.read_parquet(path)
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    destination = ROOT / "分析結果/レポート/資産配分比較"
    figures = ROOT / "分析結果/図/資産配分比較"
    destination.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    study = run_allocation_study(prices)
    study.summary.to_csv(destination / "summary.csv")
    period_comparison(study).to_csv(destination / "subperiods.csv")
    returns = pd.DataFrame({m: r.net_return for m, r in study.reports.items()})
    returns.to_csv(destination / "net_returns.csv")
    for method in METHODS:
        study.reports[method].to_csv(destination / f"daily_{method}.csv")
        study.targets[method].to_csv(destination / f"targets_{method}.csv")
        study.audits[method].to_csv(destination / f"audit_{method}.csv")
    asset_returns = simple_returns(prices)
    asset_returns.corr().to_csv(destination / "asset_correlations.csv")
    pd.DataFrame({asset: performance(asset_returns[asset]) for asset in prices}).T.to_csv(
        destination / "asset_statistics_full_sample.csv"
    )
    # 感度分析は全窓で共通の2007年開始。主分析との開始日の違いを明示する。
    sensitivity = []
    for lookback in [63, 126, 252, 504]:
        result = run_allocation_study(prices, lookback=lookback, evaluation_start="2007-01-01")
        for method, row in result.summary.iterrows():
            sensitivity.append({"lookback": lookback, "method": method, **row.to_dict()})
    pd.DataFrame(sensitivity).to_csv(destination / "lookback_sensitivity.csv", index=False)
    cost_rows = []
    for cost in [0, 5, 20]:
        result = study if cost == 5 else run_allocation_study(prices, cost_bps=cost)
        for method, row in result.summary.iterrows():
            cost_rows.append({"cost_bps": cost, "method": method, **row.to_dict()})
    pd.DataFrame(cost_rows).to_csv(destination / "cost_sensitivity.csv", index=False)
    intervals = paired_block_intervals(study)
    intervals.to_csv(destination / "paired_block_intervals.csv")
    annual = (1 + returns).groupby(returns.index.year).prod() - 1
    annual.to_csv(destination / "calendar_returns.csv")
    rolling_corr = pd.DataFrame(
        {
            "SPY / TLT": asset_returns.SPY.rolling(252).corr(asset_returns.TLT),
            "SPY / GLD": asset_returns.SPY.rolling(252).corr(asset_returns.GLD),
            "TLT / GLD": asset_returns.TLT.rolling(252).corr(asset_returns.GLD),
        }
    )
    rolling_corr.to_csv(destination / "rolling_correlations.csv")
    for column, filename, title in [
        ("wealth", "01_累積資産の比較.png", "SPY / TLT / GLD: walk-forward, after 5 bps costs"),
        ("drawdown", "02_ドローダウンの比較.png", "Drawdown from prior peak: net of trading costs"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for method, report in study.reports.items():
            values = report.wealth if column == "wealth" else drawdown(report.net_return) * 100
            ax.plot(values.index, values, label=LABELS[method], color=COLORS[method], linewidth=1.5)
        if column == "wealth":
            ax.set_yscale("log")
            ax.set_ylabel("Wealth (initial = 1, log scale)")
        else:
            ax.set_ylabel("Drawdown (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, method in zip(axes, METHODS[1:], strict=True):
        targets = study.targets[method]
        ax.stackplot(
            targets.index,
            targets.T,
            labels=targets.columns,
            colors=["#0072b2", "#e69f00", "#009e73"],
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.set_title(LABELS[method], loc="left")
        ax.legend(loc="upper left", ncol=3)
    fig.suptitle("Monthly target weights (estimated using the past 252 trading days)")
    fig.tight_layout()
    fig.savefig(figures / "03_目標配分の推移.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    rolling_corr.plot(ax=ax, linewidth=1)
    ax.set(title="Trailing 252-day asset correlations", ylim=(-1, 1), ylabel="Correlation")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "04_相関係数の推移.png", dpi=160)
    plt.close(fig)
    summary = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "data": provenance,
        "input_sha256_verified": hashlib.sha256(path.read_bytes()).hexdigest()
        == provenance["sha256"],
        "lookback": 252,
        "cost_bps_one_way": 5,
        "risk_free_for_comparative_sharpe": 0,
        "methods": list(METHODS),
        "evaluation_start": study.summary.start.iloc[0],
        "evaluation_end": study.summary.end.iloc[0],
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 21,
        "summary": study.summary.to_dict(orient="index"),
    }
    (destination / "study.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        study.summary[
            ["annual_return", "volatility", "sharpe", "max_drawdown", "annual_turnover"]
        ].to_string()
    )
    print("\nConditional paired block intervals (strategy minus equal weight):")
    print(intervals.to_string())
    print("\nStudy saved to 分析結果/レポート/資産配分比較")


if __name__ == "__main__":
    main()
