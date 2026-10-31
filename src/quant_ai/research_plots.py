"""研究の全主要図をNotebookとPNGで共用。日付・単位・凡例を明示する。"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_ai.metrics import drawdown
from quant_ai.risk_research import RiskResearch
from quant_ai.volatility_target import CORE_METHODS

LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse volatility",
    "min_variance": "Minimum variance",
}
COLORS = {"equal_weight": "#3b4a5a", "inverse_vol": "#009e73", "min_variance": "#d55e00"}


def research_figures(bundle: RiskResearch, prices: pd.DataFrame, destination: Path) -> dict:
    """9図を返しPNG保存。各グラフに条件・評価期間を付ける。呼出側でshow/close可能。"""
    destination.mkdir(parents=True, exist_ok=True)
    figures = {}
    start = bundle.summary.loc["cap15_equal_weight", "start"]
    end = bundle.summary.loc["cap15_equal_weight", "end"]
    period = f"{start} to {end}; USD; 252-day estimates; monthly next-close; 5 bps"

    def finish(fig: plt.Figure, key: str, title: str) -> None:
        """期間・共通条件を添え、Notebookに返す図と同じものを保存する。"""
        conditions = (
            period
            if not key.startswith("09_")
            else period.replace("; 5 bps", "; costs / rates varied as labelled")
        )
        fig.suptitle(title + "\n" + conditions, fontsize=11)
        for ax in fig.axes:
            ax.grid(alpha=0.2)
            ax.set_xlabel(
                "Date"
                if isinstance(ax.xaxis.get_major_locator(), mdates.DateLocator)
                else (ax.get_xlabel() or "Strategy")
            )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(destination / f"{key}.png", dpi=150)
        figures[key] = fig

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, mode in zip(axes, ["unbounded", "cap15"], strict=True):
        for method in CORE_METHODS:
            data = bundle.daily[f"{mode}_{method}"].daily
            ax.plot(data.index, (data.wealth - 1) * 100, label=LABELS[method], color=COLORS[method])
        ax.set(ylabel="Cumulative return (%)", title=f"10% target / {mode}")
        ax.legend(ncol=3)
    finish(fig, "01_累積リターン", "Cumulative returns after financing and trading costs")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, mode in zip(axes, ["unbounded", "cap15"], strict=True):
        for method in CORE_METHODS:
            data = bundle.daily[f"{mode}_{method}"].daily
            ax.plot(
                data.index,
                drawdown(data.net_return) * 100,
                label=LABELS[method],
                color=COLORS[method],
            )
        ax.set(ylabel="Drawdown (%)", title=f"10% target / {mode}")
        ax.legend(ncol=3)
    finish(fig, "02_ドローダウン", "Drawdowns include initial capital")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, mode in zip(axes, ["unbounded", "cap15"], strict=True):
        for method in CORE_METHODS:
            data = bundle.daily[f"{mode}_{method}"].daily
            realized = data.net_return.rolling(63).std() * np.sqrt(252) * 100
            ax.plot(realized.index, realized, label=LABELS[method], color=COLORS[method])
        ax.axhline(10, color="black", linestyle="--", label="Annual target 10%")
        ax.set(ylabel="Annual volatility (%)", title=f"Trailing 63 trading days / {mode}")
        ax.legend(ncol=2)
    finish(
        fig,
        "03_ローリングボラティリティ",
        "Ex-post realized volatility (never used to set weights)",
    )

    asset_returns = prices.pct_change(fill_method=None).dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    for first, second in [("SPY", "TLT"), ("SPY", "GLD"), ("TLT", "GLD")]:
        corr = asset_returns[first].rolling(252).corr(asset_returns[second]).loc[start:end]
        ax.plot(corr.index, corr, label=f"{first} / {second}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(ylabel="Correlation", ylim=(-1, 1))
    ax.legend(ncol=3)
    finish(fig, "04_資産相関", "Trailing 252-day adjusted-return correlations")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, method in zip(axes, CORE_METHODS, strict=True):
        data = bundle.daily[f"cap15_{method}"].daily
        ax.stackplot(
            data.index,
            *[data[f"post_{asset}"] for asset in prices],
            labels=list(prices),
            colors=["#0072b2", "#e69f00", "#009e73"],
            alpha=0.8,
        )
        ax.plot(
            data.index,
            data.post_cash_weight,
            color="black",
            linewidth=0.7,
            label="Cash (+) / borrowing (-)",
        )
        ax.set(
            ylabel="Fraction of NAV", title=f"{LABELS[method]} / cap15 / actual after-close weights"
        )
        ax.legend(ncol=4, fontsize=8)
    finish(fig, "05_保有配分", "Realized holdings: risky allocation + signed cash = 1")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric, unit in zip(
        axes.flat,
        ["annual_return", "volatility", "max_drawdown", "sharpe"],
        ["CAGR (%)", "Annual volatility (%)", "Maximum DD (%)", "Sharpe (rf=0)"],
        strict=True,
    ):
        scale = 1 if metric == "sharpe" else 100
        x = np.arange(3)
        for offset, prefix, label in [
            (-0.18, "original", "Original 100% risky"),
            (0.18, "cap15", "10% target / monthly cap1.5"),
        ]:
            vals = [bundle.summary.loc[f"{prefix}_{m}", metric] * scale for m in CORE_METHODS]
            ax.bar(x + offset, vals, width=0.36, label=label)
        ax.set_xticks(x, [LABELS[m] for m in CORE_METHODS], rotation=12)
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
    finish(fig, "06_リスク統一前後", "Before / after ex-ante risk normalization")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ["unbounded", "cap15"], strict=True):
        x = np.arange(3)
        for offset, metric, label in [
            (-0.18, "forecast_mean", "Monthly mean forecast"),
            (0.18, "volatility", "Full-period realized"),
        ]:
            vals = [bundle.summary.loc[f"{mode}_{m}", metric] * 100 for m in CORE_METHODS]
            ax.bar(x + offset, vals, width=0.36, label=label)
        ax.axhline(10, color="black", linestyle="--", label="Annual target 10%")
        ax.set_xticks(x, [LABELS[m] for m in CORE_METHODS], rotation=12)
        ax.set(ylabel="Annual volatility (%)", title=mode)
        ax.legend(fontsize=8)
    finish(fig, "07_目標と実現ボラ", "Common target does not imply identical realized risk")

    fig, ax = plt.subplots(figsize=(12, 5))
    periods = list(bundle.periods.period.unique())
    x = np.arange(len(periods))
    for j, method in enumerate(CORE_METHODS):
        subset = bundle.periods[bundle.periods.case_id == f"cap15_{method}"].set_index("period")
        ax.bar(
            x + (j - 1) * 0.25,
            subset.loc[periods, "annual_return"] * 100,
            width=0.25,
            label=LABELS[method],
            color=COLORS[method],
        )
    ax.set_xticks(x, periods)
    ax.set(ylabel="Subperiod CAGR (%)", xlabel="Fixed market subperiod")
    ax.legend(ncol=3)
    finish(fig, "08_局面別成績", "10% target / monthly cap1.5 (last subperiod is incomplete)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for method in CORE_METHODS:
        subset = bundle.summary[
            (bundle.summary.family == "cost_sensitivity")
            & (bundle.summary.leverage_cap == 1.5)
            & (bundle.summary.method == method)
        ].sort_values("cost_bps")
        axes[0, 0].plot(
            subset.cost_bps,
            subset.annual_return * 100,
            "o-",
            label=LABELS[method],
            color=COLORS[method],
        )
        subset = bundle.summary[
            (bundle.summary.family == "funding_sensitivity")
            & (bundle.summary.leverage_cap == 1.5)
            & (bundle.summary.method == method)
        ].sort_values("cash_rate")
        axes[0, 1].plot(
            np.arange(3),
            subset.annual_return * 100,
            "o-",
            label=LABELS[method],
            color=COLORS[method],
        )
        data = bundle.daily[f"cap15_{method}"].daily
        axes[1, 0].plot(data.index, data.held_leverage, label=LABELS[method], color=COLORS[method])
    axes[0, 0].set(xlabel="One-way trading cost (bps)", ylabel="CAGR (%)")
    axes[0, 0].set_title("Rates 0 / 0%; trading costs varied")
    axes[0, 1].set_xticks([0, 1, 2], ["0 / 0", "2 / 3", "4 / 6"])
    axes[0, 1].set(xlabel="Deposit / borrowing annual rates (%)", ylabel="CAGR (%)")
    axes[0, 1].set_title("Trading cost 5 bps; fixed rate scenarios")
    axes[1, 0].axhline(1.5, linestyle="--", color="black", linewidth=0.7)
    axes[1, 0].set(ylabel="Held risky exposure / NAV", xlabel="Date")
    x = np.arange(3)
    for offset, metric, label in [
        (-0.18, "mean_held_leverage", "Average held"),
        (0.18, "max_realized_leverage", "Maximum observed close"),
    ]:
        axes[1, 1].bar(
            x + offset,
            [bundle.summary.loc[f"cap15_{m}", metric] for m in CORE_METHODS],
            width=0.36,
            label=label,
        )
    axes[1, 1].set_xticks(x, [LABELS[m] for m in CORE_METHODS], rotation=12)
    axes[1, 1].set_ylabel("Exposure / NAV (multiple)")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    finish(
        fig,
        "09_コスト金利とレバレッジ",
        "Monthly cap1.5 does not enforce a daily maintenance ceiling",
    )
    return figures
