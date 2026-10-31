"""戦略比較・収益分解の図と日本語報告。CAGR差・対数差・支払額を混同しない。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_ai.research_report import markdown_table
from quant_ai.return_attribution import COMPONENTS, AttributionStudy
from quant_ai.volatility_target import CORE_METHODS

METHOD_NAMES = {"equal_weight": "等ウェイト", "inverse_vol": "逆ボラ", "min_variance": "最小分散"}
METHOD_LABELS = {
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse volatility",
    "min_variance": "Minimum variance",
}
COLORS = {"equal_weight": "#3b4a5a", "inverse_vol": "#009e73", "min_variance": "#d55e00"}
PART_NAMES = {
    "allocation": "資産配分",
    "exposure": "倍率変更",
    "extra_trading_cost": "追加売買費用",
    "funding_cost": "借入費用",
    "cash_income": "預金利息",
}
PART_LABELS = {
    "allocation": "Allocation (gross)",
    "exposure": "Exposure (gross)",
    "extra_trading_cost": "Extra trading costs",
    "funding_cost": "Borrowing costs",
    "cash_income": "Cash interest",
}


def attribution_figures(study: AttributionStudy, root: Path) -> dict:
    """4図をPNG保存してNotebookにも返す。評価期間・単位・基準を表示。"""
    directory = root / "outputs/figures/戦略比較と収益分解"
    directory.mkdir(parents=True, exist_ok=True)
    source = root / "outputs/reports/戦略比較と収益分解"
    figures = {}
    period = "2006-01-03 to 2026-10-29; USD; monthly next-close; one-way trading cost 5 bps"

    def finish(fig: plt.Figure, name: str, title: str) -> None:
        """図を出力し、同じFigureを呼出側から表示できるよう保持する。"""
        fig.suptitle(title + "\n" + period, fontsize=11)
        for ax in fig.axes:
            ax.grid(alpha=0.2)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(directory / f"{name}.png", dpi=150)
        figures[name] = fig

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, prefix, title in [
        (axes[0], "raw", "100% risky, no leverage; rates 0 / 0%"),
        (axes[1], "cap15_rates0", "Vol target 10%; monthly target cap1.5; rates 0 / 0%"),
    ]:
        for method in CORE_METHODS:
            path = pd.read_parquet(source / f"path_{prefix}_{method}.parquet")
            ax.plot(path.index, path.wealth, label=METHOD_LABELS[method], color=COLORS[method])
        ax.set(yscale="log", ylabel="NAV (initial USD = 1, log scale)", title=title, xlabel="Date")
        ax.legend(ncol=3)
    finish(
        fig,
        "01_非レバとVTの資産推移",
        "Same data / same fees: unlevered versus volatility targeting",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(
        axes.flat,
        ["annual_return", "volatility", "max_drawdown", "sharpe"],
        ["CAGR (%)", "Annual volatility (%)", "Maximum drawdown (%)", "Sharpe (rf=0)"],
        strict=True,
    ):
        x = np.arange(3)
        multiplier = 1 if metric == "sharpe" else 100
        for offset, prefix, description in [
            (-0.24, "raw", "Unlevered / rates 0/0%"),
            (0.0, "cap15_rates0", "VT / rates 0/0%"),
            (0.24, "cap15_rates23", "VT / rates 2/3%"),
        ]:
            values = [
                study.performance.loc[f"{prefix}_{method}", metric] * multiplier
                for method in CORE_METHODS
            ]
            ax.bar(x + offset, values, width=0.24, label=description)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in CORE_METHODS], rotation=12)
        ax.set(ylabel=label, xlabel="Strategy")
        ax.legend(fontsize=8)
    finish(
        fig,
        "02_非レバとVTの成績比較",
        "VT: 10% forecast target, monthly cap1.5; realized risks differ",
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, rates, title in [
        (axes[0], "rates0", "Current base: deposit / borrowing = 0 / 0%"),
        (axes[1], "rates23", "Sensitivity: deposit / borrowing = 2 / 3%"),
    ]:
        x = np.arange(3)
        positive, negative = np.zeros(3), np.zeros(3)
        for component, color in zip(
            COMPONENTS, ["#009e73", "#0072b2", "#e69f00", "#cc3311", "#aa4499"], strict=True
        ):
            values = np.array(
                [
                    study.bridges.loc[f"cap15_{rates}_{method}", component] * 100
                    for method in CORE_METHODS
                ]
            )
            bottom = np.where(values >= 0, positive, negative)
            ax.bar(x, values, bottom=bottom, label=PART_LABELS[component], color=color, width=0.5)
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        total = [
            study.bridges.loc[f"cap15_{rates}_{method}", "total_cagr_difference"] * 100
            for method in CORE_METHODS
        ]
        ax.plot(x, total, "kD", label="Total CAGR difference")
        span = float(positive.max() - negative.min())
        ax.set_ylim(
            float(negative.min()) - 0.10 * max(span, 0.1),
            float(positive.max()) + 0.18 * max(span, 0.1),
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(ylabel="CAGR difference (percentage points/year)", title=title)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(np.arange(3), [METHOD_LABELS[m] for m in CORE_METHODS])
    axes[-1].set_xlabel("Underlying allocation strategy")
    finish(
        fig,
        "03_CAGR差の要因分解",
        "Fixed counterfactual bridge vs unlevered equal weight; not causal effects",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, component in zip(axes, ["allocation", "exposure"], strict=True):
        x = np.arange(3)
        positive, negative = np.zeros(3), np.zeros(3)
        for asset, color in [("SPY", "#0072b2"), ("TLT", "#e69f00"), ("GLD", "#009e73")]:
            values = np.array(
                [
                    study.asset_contributions.loc[
                        (study.asset_contributions.case_id == f"cap15_rates23_{method}")
                        & (study.asset_contributions.component == component)
                        & (study.asset_contributions.asset == asset),
                        "annual_log_contribution",
                    ].iloc[0]
                    * 100
                    for method in CORE_METHODS
                ]
            )
            ax.bar(
                x,
                values,
                bottom=np.where(values >= 0, positive, negative),
                label=asset,
                color=color,
            )
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in CORE_METHODS], rotation=12)
        ax.set(
            xlabel="Strategy",
            ylabel="Annual log-return difference (log pp/year)",
            title=f"{component}; monthly cap1.5; rates 2/3%",
        )
        ax.legend(ncol=3)
    finish(
        fig,
        "04_資産別の配分と倍率寄与",
        "Daily linked contributions vs unlevered EW (different units from CAGR bridge)",
    )
    return figures


def write_attribution_report(study: AttributionStudy, root: Path) -> Path:
    """比較基準・順序・交互作用を明示し、現在の金利0ケースを変えずに報告する。"""
    summaries = []
    for method in CORE_METHODS:
        for prefix, description in [
            ("raw", "非レバ・金利0/0%"),
            ("cap15_rates0", "VT・金利0/0%"),
            ("cap15_rates23", "VT・金利2/3%"),
        ]:
            row = study.performance.loc[f"{prefix}_{method}"]
            summaries.append(
                {
                    "戦略": METHOD_NAMES[method],
                    "条件": description,
                    "CAGR": f"{row.annual_return:.2%}",
                    "年率ボラ": f"{row.volatility:.2%}",
                    "最大DD": f"{row.max_drawdown:.2%}",
                    "Sharpe(rf=0)": f"{row.sharpe:.3f}",
                }
            )
    bridges = []
    for rate in ["rates0", "rates23", "rates46"]:
        for method in CORE_METHODS:
            row = study.bridges.loc[f"cap15_{rate}_{method}"]
            record = {"金利条件": rate, "戦略": METHOD_NAMES[method]}
            record.update({PART_NAMES[c] + "(pt)": f"{row[c] * 100:+.4f}" for c in COMPONENTS})
            record.update(
                {
                    "寄与合計(pt)": f"{row.total_cagr_difference * 100:+.4f}",
                    "最終CAGR": f"{row.portfolio_net_cagr:.2%}",
                }
            )
            bridges.append(record)
    payments = []
    for method in CORE_METHODS:
        row = study.payments.loc[f"cap15_rates23_{method}"]
        payments.append(
            {
                "戦略": METHOD_NAMES[method],
                "借入実支払額": f"{-row.funding_cost:.4f}",
                "預金実受取額": f"{row.cash_income:.4f}",
                "売買実支払額": f"{-row.trading_cost:.4f}",
                "最終資産": f"{row.terminal_wealth:.4f}",
            }
        )
    log_rows = []
    for method in CORE_METHODS:
        case = f"cap15_rates23_{method}"
        frame = study.contributions[study.contributions.case_id == case].set_index("component")
        for kind, title in [
            ("arithmetic_annual_contribution", "年率算術リターン差"),
            ("annual_log_contribution", "年率対数リターン差"),
        ]:
            record = {"戦略": METHOD_NAMES[method], "単位": title + "のポイント"}
            record.update({PART_NAMES[c]: f"{frame.loc[c, kind] * 100:+.4f}" for c in COMPONENTS})
            record["合計"] = f"{frame.loc[list(COMPONENTS), kind].sum() * 100:+.4f}"
            record["倍率内の交互作用"] = f"{frame.loc['interaction_in_exposure', kind] * 100:+.4f}"
            log_rows.append(record)
    min0 = study.bridges.loc["cap15_rates0_min_variance"]
    min23 = study.bridges.loc["cap15_rates23_min_variance"]
    sections = [
        "# 同じデータによる非レバ3戦略・VT3戦略と収益源の分解\n\n"
        "2026-10-31実行。元分析と環境は保持。",
        "## 要点\n\n"
        "主条件の最小分散は非レバCAGR "
        f"{study.performance.loc['raw_min_variance', 'annual_return']:.2%}、"
        f"VT金利0%では{min0.portfolio_net_cagr:.2%}、預金2%/借入3%では{min23.portfolio_net_cagr:.2%}。"
        "以下の定義では、非レバ等ウェイトからのCAGR増分は主に倍率変更の比較経路に現れた。"
        "これは倍率変更の独立した因果効果や投資スキルの証明ではない。",
        "## 比較条件と基準\n\n"
        "同じSPY/TLT/GLD調整価格を使用。評価2006-01-03〜2026-10-29、5239営業日。"
        "252日推定、月末判断・翌共通営業日終値、片道5bps、USD。"
        "非レバはリスク資産100%の3配分。VTは同じ配分に目標10%の倍率を乗じる。"
        "主表は月次目標上限1.5倍、無上限の結果もCSVに保存。月中ドリフトは上限を超え得る。\n\n"
        "分解の基準は全て**金利0%・非レバ・費用後の等ウェイト**。"
        "rates0は預金0/借入0%、rates23は2/3%、rates46は4/6%。後二つは固定仮想金利で歴史的金利ではない。"
        "現在の基準を金利ありへ置き換えていない。Sharpeは全表rf=0の比較指標。",
        "## 条件1・2：運用成績\n\n" + markdown_table(pd.DataFrame(summaries)),
        "## 条件3：CAGR差を比較経路で橋渡し\n\n"
        "以下は年率CAGR差のポイント。足し算できるよう比較経路を明示して定義した差で、"
        "各効果が独立な『要因別CAGR』という意味ではない。G(X)を経路XのCAGRとして、\n\n"
        "1. 資産配分：G(非レバ戦略・費用0)−G(非レバEW・費用0)。\n"
        "2. 倍率変更：G(VT・金利0・費用0)−G(非レバ戦略・費用0)。配分×倍率の交互作用を含む。\n"
        "3. 追加売買費用：[G(VT・金利0・費用5bps)−G(VT・金利0・費用0)]"
        "−[G(非レバEW・費用5bps)−G(非レバEW・費用0)]。\n"
        "4. 借入費用：G(VT・預金0/借入b・費用あり)−G(VT・金利0・費用あり)。\n"
        "5. 預金利息：G(VT・預金c/借入b・費用あり)−G(VT・預金0/借入b・費用あり)。\n\n"
        "5つの差は最終VTのCAGR−基準EWのCAGRへ厳密に一致する。金利→倍率など"
        "比較順序を変えると配賦は変わり得る。借入と預金は別の比較経路で分けた。\n\n"
        + markdown_table(pd.DataFrame(bridges)),
        "## 日次の損益恒等式と交互作用\n\n"
        "q_sは実際のリスク資産内構成、Lは実際の期首リスク資産/NAV。a_s=q_s' r、a_EW=q_EW' r。\n\n"
        "r_VT−r_EW = (a_s−a_EW) + (L−1)a_s + 預金受取/NAV − 借入支払/NAV − 追加売買費用。\n\n"
        "倍率変更項は (L−1)a_EW + (L−1)(a_s−a_EW)。後者が交互作用で、倍率効果の内数として表示する。"
        "月末の目標倍率ではなく期首の実保有倍率を使い、初回の現金状態も正しく扱う。"
        "各資産の配分/倍率寄与はasset_contributions.csv、日次寄与はattribution_*.parquet。",
        "## 複数期間：CAGR・算術差・対数差の単位を分ける\n\n"
        "日次の寄与を252/nで足した値は年率**算術**リターン差で、CAGR差ではない。"
        "別にk_t=[log(1+r_VT)−log(1+r_EW)]/(r_VT−r_EW)を用い、"
        "252/n Σ k_t c_j,tを年率**対数**差への寄与として報告する。"
        "同リターン時はk_t=1/(1+r_EW)。合計はlog(1+CAGR_VT)−log(1+CAGR_EW)へ一致する。"
        "この日次係数は[Carinoの公式実装](https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Carino.R)"
        "と同型だが、ここでは最終的な算術超過収益へ変換する全期間係数で割らず対数単位のまま出す。"
        "[複数期間の連結の論点](https://rpc.cfainstitute.org/research/financial-analysts-journal/2004/multiperiod-arithmetic-attribution)。\n\n"
        "下表は金利2/3%。CAGR橋渡しと値が異なるのは、単位・複利・経路・連結の定義が違うため。"
        "交互作用の列は倍率の内数なので合計に二重加算しない。\n\n"
        + markdown_table(pd.DataFrame(log_rows)),
        "## 実際の支払額・受取額\n\n初期資金1USDあたりの全期間金額。金利2/3%、主条件。"
        "借入実支払額とCAGRの借入効果は異なる単位で、資金経路・複利のため同じ数値にならない。\n\n"
        + markdown_table(pd.DataFrame(payments)),
        "## 解釈・限界\n\n"
        "配分と倍率は独立ではない。戦略ごとの予測ボラが倍率を決めるため、倍率項には"
        "市場タイミング、平均倍率、配分との交互作用が混在する。今回それらをスキルと断定しない。"
        "将来データは売買には使わず、複数期間の寄与計算は事後評価として行う。"
        "条件は旧結果を見た後の追加研究で、独立OOS・因果実験ではない。"
        "この分解は収益差の分解であり、Sharpeや最大DDの因果要因分解ではない。"
        "基準EWや比較順序を変えると分解も変わる。独立将来検証・実調達条件・約定ずれ・"
        "流動性・強制縮小等は引き続き未確認。",
        "## 再現・数値監査\n\n"
        f"現在VTとの日次最大差{study.audit['legacy_max_daily_difference']:.3g}、"
        f"日次分解残差{study.audit['max_daily_residual']:.3g}、"
        f"年率対数分解残差{study.audit['max_log_residual']:.3g}、"
        f"CAGR橋渡し残差{study.audit['max_cagr_bridge_residual']:.3g}。\n\n"
        "`python scripts/run_return_attribution.py`で全計算・図・報告を再生成。"
        "`--render-only`は既存数値から図と報告だけ更新する。"
        "Notebook09に実際の関数・式・再計算・図を用意。実データ結果と合成テストを区別する。",
    ]
    path = root / "outputs/reports/戦略比較と収益分解_分析レポート.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path
