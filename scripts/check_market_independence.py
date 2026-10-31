"""既存の同一経路を直接比較し、市場環境に依存しない優位性という仮説を検証。"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.market_regimes import (
    GROUPS,
    LABELS,
    market_states,
    pair_summary,
    paired_regime_bootstrap,
    rolling_comparison,
)
from quant_ai.research_report import markdown_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """固定データ・既存VT純収益で比較し、原本を変えず新規成果物を保存する。"""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    source = ROOT / "outputs/reports/戦略比較と収益分解"
    output = ROOT / "outputs/reports/市場環境別VT直接比較"
    figures = ROOT / "outputs/figures/市場環境別VT直接比較"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    price_path = ROOT / "data/processed/spy_tlt_gld_adjusted.parquet"
    provenance = json.loads(price_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert hashlib.sha256(price_path.read_bytes()).hexdigest() == provenance["sha256"]
    prices = pd.read_parquet(price_path)
    all_pairs, full_rows = {}, []
    for mode in ["cap15", "unbounded"]:
        for rates in ["rates0", "rates23", "rates46"]:
            paths = [
                pd.read_parquet(source / f"path_{mode}_{rates}_{method}.parquet")
                for method in ["min_variance", "equal_weight"]
            ]
            if not paths[0].index.equals(paths[1].index):
                raise ValueError("Evaluation dates differ")
            pair = pd.DataFrame({"minimum": paths[0].net_return, "equal": paths[1].net_return})
            all_pairs[f"{mode}_{rates}"] = pair
            full_rows.append(dict(case=f"{mode}_{rates}", **pair_summary(pair)))
    pair = all_pairs["cap15_rates0"]
    states = market_states(prices).loc[pair.index]
    assert states["all"].all()
    pair.to_parquet(output / "paired_returns.parquet")
    states.to_parquet(output / "lagged_states.parquet")
    full = pd.DataFrame(full_rows)
    full.to_csv(output / "full_comparison.csv", index=False)
    # 日付・全期間成績を保存済み成績と照合。
    old = pd.read_csv(source / "performance.csv", index_col=0)
    np.testing.assert_allclose(
        pair_summary(pair)["minimum_cagr"],
        old.loc["cap15_rates0_min_variance", "annual_return"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pair_summary(pair)["equal_cagr"],
        old.loc["cap15_rates0_equal_weight", "annual_return"],
        atol=1e-12,
    )
    rows = []
    for case, p in all_pairs.items():
        for group in GROUPS[1:]:
            mask = states[group]
            stats = pair_summary(p.loc[mask])
            # 非連続抽出日のDDは連続運用のDDではないため出力しない。
            stats.pop("minimum_mdd")
            stats.pop("equal_mdd")
            rows.append(dict(case=case, group=group, **stats))
    regimes = pd.DataFrame(rows)
    regimes.to_csv(output / "regimes.csv", index=False)
    calendar = []
    periods = {
        "2006-2012": ("2006", "2012"),
        "2013-2019": ("2013", "2019"),
        "2020-2021": ("2020", "2021"),
        "2022": ("2022", "2022"),
        "2023-2026YTD": ("2023", "2026"),
    }
    for case, p in all_pairs.items():
        for period, (start, end) in periods.items():
            calendar.append(dict(case=case, period=period, **pair_summary(p.loc[start:end])))
    calendar = pd.DataFrame(calendar)
    calendar.to_csv(output / "calendar_periods.csv", index=False)
    yearly = pd.DataFrame(
        [dict(year=year, **pair_summary(p)) for year, p in pair.groupby(pair.index.year)]
    )
    yearly.to_csv(output / "yearly.csv", index=False)
    rolling_rows = []
    for window in [252, 756, 1260]:
        rolling = rolling_comparison(pair, window)
        rolling.to_csv(output / f"rolling_{window}.csv")
        rolling_rows.append(
            dict(
                window=window,
                observations=len(rolling),
                cagr_win_fraction=float((rolling.cagr_difference > 0).mean()),
                sharpe_win_fraction=float((rolling.sharpe_difference > 0).mean()),
                minimum_cagr_gap=float(rolling.cagr_difference.min()),
                minimum_gap_end=str(rolling.cagr_difference.idxmin().date()),
            )
        )
    rolling_summary = pd.DataFrame(rolling_rows)
    rolling_summary.to_csv(output / "rolling_summary.csv", index=False)
    bootstrap = []
    for length in [21, 63, 126]:
        print(f"Paired conditional bootstrap: block {length}", flush=True)
        b = paired_regime_bootstrap(pair, states, block_days=length)
        bootstrap.append(b)
    bootstrap = pd.concat(bootstrap, ignore_index=True)
    bootstrap.to_csv(output / "bootstrap.csv", index=False)
    # 局面別の全期間対数超過収益への寄与。合計が全期間差に一致。
    logactive = np.log1p(pair.minimum) - np.log1p(pair.equal)
    contribution = []
    for group in GROUPS[1:5]:
        contribution.append(
            dict(
                group=group,
                annual_log_contribution=float(logactive[states[group]].sum() / len(pair) * 252),
            )
        )
    contribution = pd.DataFrame(contribution)
    assert abs(contribution.annual_log_contribution.sum() - logactive.mean() * 252) < 1e-12
    contribution.to_csv(output / "trend_contribution.csv", index=False)
    # 元研究のパラメータ感応度を同じ条件の戦略同士に結合。
    sensitivity = pd.read_csv(ROOT / "outputs/reports/リスク統一比較/all_scenarios.csv")
    keys = [
        "family",
        "target_volatility",
        "leverage_cap",
        "lookback",
        "cash_rate",
        "borrow_rate",
        "cost_bps",
        "start",
        "end",
        "observations",
    ]
    fields = ["annual_return", "volatility", "sharpe", "max_drawdown"]
    sensitivity = sensitivity[sensitivity.family != "original"]
    joined = sensitivity[sensitivity.method == "min_variance"][keys + fields].merge(
        sensitivity[sensitivity.method == "equal_weight"][keys + fields],
        on=keys,
        suffixes=("_minimum", "_equal"),
        validate="one_to_one",
    )
    for field in fields:
        joined[field + "_difference"] = joined[field + "_minimum"] - joined[field + "_equal"]
    joined.to_csv(output / "parameter_sensitivity.csv", index=False)
    audit = {
        "price_sha256": provenance["sha256"],
        "evaluation_start": str(pair.index[0].date()),
        "evaluation_end": str(pair.index[-1].date()),
        "observations": len(pair),
        "seed": 20261031,
        "repetitions": 10000,
        "blocks": [21, 63, 126],
        "bootstrap_reestimates_strategy": False,
        "regimes_use_previous_close": True,
        "independent_unseen_oos": False,
        "all_regimes_positive_cagr": bool(
            (regimes.query("case=='cap15_rates0'").cagr_difference > 0).all()
        ),
        "all_regimes_positive_sharpe": bool(
            (regimes.query("case=='cap15_rates0'").sharpe_difference > 0).all()
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(pair, regimes, calendar, bootstrap, output, figures)
    write_report(
        full, regimes, calendar, yearly, rolling_summary, bootstrap, contribution, joined, output
    )
    print(full.to_string(index=False))
    print(regimes.query("case=='cap15_rates0'").to_string(index=False))


def plot_results(
    pair: pd.DataFrame,
    regimes: pd.DataFrame,
    calendar: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output: Path,
    figures: Path,
) -> None:
    """相対資産・ローリング差・局面差・ブロックCIを期間と単位付きで保存する。"""
    import matplotlib.pyplot as plt

    period = f"{pair.index[0].date()} to {pair.index[-1].date()}; USD; 10% target; monthly cap 1.5"

    def save(fig: plt.Figure, name: str) -> None:
        """独立した画像へ保存。Notebookからも同じ図を閲覧する。"""
        fig.suptitle(
            period + "\n252-day estimation; next-close; 5 bps; cash/borrow 0/0%", fontsize=10
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(figures / f"{name}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    wealth = (1 + pair).cumprod()
    axes[0].plot(wealth.index, wealth.minimum, label="Minimum variance + VT")
    axes[0].plot(wealth.index, wealth.equal, label="Equal weight + VT")
    axes[0].set(title="Net wealth comparison", ylabel="Wealth (initial USD 1)")
    axes[1].plot(
        wealth.index, wealth.minimum / wealth.equal, label="Minimum / equal relative wealth"
    )
    axes[1].axhline(1, color="black", lw=0.8)
    axes[1].set(
        title="Relative wealth: a decline means minimum variance underperforms",
        ylabel="Relative wealth (ratio)",
        xlabel="Date",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    save(fig, "01_VT同士の直接比較")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for w in [252, 756, 1260]:
        rolling = pd.read_csv(output / f"rolling_{w}.csv", index_col=0, parse_dates=True)
        axes[0].plot(rolling.index, rolling.cagr_difference * 100, label=f"{w / 252:g}-year window")
        axes[1].plot(rolling.index, rolling.sharpe_difference, label=f"{w / 252:g}-year window")
    axes[0].set(title="Rolling CAGR: minimum VT minus equal VT", ylabel="CAGR difference (pp/year)")
    axes[1].set(
        title="Rolling Sharpe difference (rf=0)",
        ylabel="Sharpe difference",
        xlabel="Window end date",
    )
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.legend()
        ax.grid(alpha=0.25)
    save(fig, "02_ローリング期間の優劣")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    r = regimes.query("case=='cap15_rates0'").set_index("group").loc[list(GROUPS[1:])]
    names = [
        "Stock up / bond up",
        "Stock up / bond down",
        "Stock down / bond up",
        "Stock down / bond down",
        "Stock-bond corr > 0",
        "Stock-bond corr <= 0",
        "Stock vol > 20%",
        "Stock vol <= 20%",
    ]
    axes[0].barh(
        names, r.cagr_difference * 100, color=np.where(r.cagr_difference > 0, "teal", "tomato")
    )
    axes[0].set(
        title="Conditional geometric annualization: min minus equal", xlabel="Difference (pp/year)"
    )
    c = calendar.query("case=='cap15_rates0'")
    axes[1].barh(
        c.period, c.cagr_difference * 100, color=np.where(c.cagr_difference > 0, "teal", "tomato")
    )
    axes[1].set(title="Contiguous calendar subperiods", xlabel="CAGR difference (pp/year)")
    for ax in axes:
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(axis="x", alpha=0.25)
    save(fig, "03_市場局面と暦期間の比較")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, scale, title in zip(
        axes,
        ["cagr_difference", "sharpe_difference"],
        [100, 1],
        ["CAGR difference (pp/year)", "Sharpe difference (rf=0)"],
        strict=True,
    ):
        b = bootstrap[(bootstrap.group == "all") & (bootstrap.metric == metric)]
        point = b.estimate.to_numpy() * scale
        lower = b.lower_95.to_numpy() * scale
        upper = b.upper_95.to_numpy() * scale
        ax.vlines(
            b.block_days, lower, upper, color="teal", label="Pointwise 95% percentile interval"
        )
        ax.scatter(b.block_days, point, color="black", label="Observed difference")
        ax.axhline(0, color="red", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set(
            title=title,
            xlabel="Circular block length (trading days)",
            ylabel="Minimum VT minus equal VT",
        )
    save(fig, "04_ペアブロック再標本化の区間")


def write_report(
    full: pd.DataFrame,
    regimes: pd.DataFrame,
    calendar: pd.DataFrame,
    yearly: pd.DataFrame,
    rolling: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contribution: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: Path,
) -> None:
    """データの事実・条件付き統計・解釈・未確認事項を分けて日本語報告する。"""

    def table(df: pd.DataFrame, columns: list[str], percent: list[str]) -> str:
        """年率・差の単位をヘッダで明示して表に変換する。"""
        view = df[columns].copy()
        for col in percent:
            view[col] = view[col].map(lambda x: f"{x * 100:.3f}")
        for col in view.select_dtypes(include=["floating"]).columns:
            view[col] = view[col].map(lambda x: f"{x:.3f}")
        view = view.rename(
            columns={
                "case": "条件",
                "group": "局面",
                "observations": "日数",
                "period": "期間",
                "window": "窓(日)",
                "minimum_cagr": "最小分散(%)",
                "equal_cagr": "等ウェイト(%)",
                "cagr_difference": "CAGR差(pt)",
                "minimum_vol": "最小分散ボラ(%)",
                "equal_vol": "等ウェイトボラ(%)",
                "minimum_sharpe": "最小分散Sharpe",
                "equal_sharpe": "等ウェイトSharpe",
                "sharpe_difference": "Sharpe差",
                "minimum_mdd": "最小分散DD(%)",
                "equal_mdd": "等ウェイトDD(%)",
                "minimum_es95": "最小分散ES95(%)",
                "equal_es95": "等ウェイトES95(%)",
                "cagr_win_fraction": "CAGR勝率(%)",
                "sharpe_win_fraction": "Sharpe勝率(%)",
                "minimum_cagr_gap": "CAGR差の最小値(pt)",
                "minimum_gap_end": "最悪窓の終了日",
                "annual_log_contribution": "年率対数寄与(pt)",
            }
        )
        return markdown_table(view)

    primary = regimes.query("case=='cap15_rates0'").copy()
    primary["group"] = primary.group.map(LABELS)
    ci = bootstrap[
        (bootstrap.group == "all")
        & (bootstrap.metric.isin(["cagr_difference", "sharpe_difference"]))
    ].copy()
    for col in ["estimate", "lower_95", "upper_95", "family_lower_95", "family_upper_95"]:
        ci[col] = ci.apply(
            lambda r, col=col: f"{r[col] * (100 if r.metric == 'cagr_difference' else 1):.3f}",
            axis=1,
        )
    contrasts = bootstrap[
        (bootstrap.block_days == 63) & (bootstrap.metric == "annual_log_active_contrast")
    ].copy()
    counts = regimes.query("case=='cap15_rates0'")
    ref = full[full.case == "cap15_rates0"].iloc[0]
    parts = [
        "# 最小分散＋VTは市場環境に依存せず等ウェイト＋VTを上回るか\n\n"
        "2026-10-31実行。同じ凍結済みSPY/TLT/GLD調整価格・既存純収益を直接比較。",
        "## 判定\n\n"
        "**「市場環境に依存せず上回る」という強い仮説は支持できない。** "
        "全期間平均の優位、各観測局面での優位、未知の環境でも成立する普遍的優位は別の主張。"
        f"全期間CAGR差は{ref.cagr_difference * 100:+.3f}ポイント、"
        f"Sharpe差は{ref.sharpe_difference:+.3f}。"
        f"しかし観測した8分類のうちCAGR差が負の分類は{int((counts.cagr_difference < 0).sum())}、"
        f"Sharpe差が負の分類は{int((counts.sharpe_difference < 0).sum())}。"
        "有限の過去データから「全ての市場環境」を証明することもできない。",
        "## 比較条件と仮説の操作的定義\n\n"
        "評価2006-01-03〜2026-10-29、5239営業日。252営業日推定、月次判断、翌営業日終値約定、"
        "新保有の損益はその次の終値から。目標年率10%、主条件は月次目標倍率上限1.5、"
        "片道5bps、ドル建て。金利0/0%が主条件で、預金2%/借入3%、4%/6%、倍率無上限も比較。"
        "月中倍率はドリフトする。両者の実現ボラは一致しない。Sharpe分子は日次算術平均×252、"
        "分母は標本標準偏差×√252、rf=0。CAGRはΠ(1+r)^(252/n)−1。\n\n"
        "主にCAGRとSharpeがどの局面でも上回るかを確認し、最大DD・ESを併記する。"
        "局面間の成績差がゼロという仮説を棄却できないことを、環境非依存性の証明とは扱わない。"
        "新たな閾値や局面区分は既存結果を見た後の探索的定義で、事前登録された独立検証ではない。",
        "## データから直接確認：全期間\n\n"
        "CAGR/ボラ/DDは%、CAGR差はポイント、Sharpeは無次元。\n\n"
        + table(
            full,
            [
                "case",
                "minimum_cagr",
                "equal_cagr",
                "cagr_difference",
                "minimum_vol",
                "equal_vol",
                "minimum_sharpe",
                "equal_sharpe",
                "minimum_mdd",
                "equal_mdd",
            ],
            [
                "minimum_cagr",
                "equal_cagr",
                "cagr_difference",
                "minimum_vol",
                "equal_vol",
                "minimum_mdd",
                "equal_mdd",
            ],
        ),
        "## 前日情報で分類した市場環境\n\n"
        "株↑/↓はSPYの過去252日調整価格騰落が正/非正、債↑/↓はTLTで同様。"
        "株債相関は過去252日の日次収益相関の正/非正。株式ボラは63日年率ボラが20%超/以下。"
        "全特徴量を1営業日遅らせた。TLTトレンドは金利そのものの上昇/低下ではない。"
        "各分類内は非連続日が混じるため、CAGR欄は条件付き幾何年率換算、Sharpeは条件付き指標。"
        "局面だけで投資できる戦略の成績ではない。非連続局面のDDは表示しない。"
        "トレンド4群は互いに排他的。他の相関/ボラ群とは重複し、8群を合算しない。\n\n"
        + table(
            primary,
            [
                "group",
                "observations",
                "minimum_cagr",
                "equal_cagr",
                "cagr_difference",
                "sharpe_difference",
                "minimum_vol",
                "equal_vol",
                "minimum_es95",
                "equal_es95",
            ],
            [
                "minimum_cagr",
                "equal_cagr",
                "cagr_difference",
                "minimum_vol",
                "equal_vol",
                "minimum_es95",
                "equal_es95",
            ],
        ),
        "## 暦期間・年別・ローリング\n\n"
        "暦期間は旧研究と同じ区分。2026年は10月29日まで。各期間内で資産額を1に正規化して"
        "指標を再計算するが、配分は連続した既存運用経路を維持する。\n\n"
        + table(
            calendar.query("case=='cap15_rates0'"),
            [
                "period",
                "minimum_cagr",
                "equal_cagr",
                "cagr_difference",
                "sharpe_difference",
                "minimum_mdd",
                "equal_mdd",
            ],
            ["minimum_cagr", "equal_cagr", "cagr_difference", "minimum_mdd", "equal_mdd"],
        )
        + "\n\n"
        f"年別CAGRで上回った年は{int((yearly.cagr_difference > 0).sum())}/{len(yearly)}年。"
        "ローリング勝率は重複窓なので独立な成功回数・p値にはしない。\n\n"
        + table(
            rolling,
            [
                "window",
                "observations",
                "cagr_win_fraction",
                "sharpe_win_fraction",
                "minimum_cagr_gap",
                "minimum_gap_end",
            ],
            ["cagr_win_fraction", "sharpe_win_fraction", "minimum_cagr_gap"],
        ),
        "## 条件付き統計：ペア循環ブロック・ブートストラップ\n\n"
        "同じ日付の両戦略収益と局面ラベルをまとめて循環ブロックで抽出。各10000回、"
        "21/63/126営業日、乱数seed=20261031。各標本を5239日ちょうどに切り、"
        "CAGR差・Sharpe差・年率対数収益差を再計算する。下表は全期間の通常95%区間と、"
        "9群×3指標＋3局面差=30個の比較にBonferroni補正した参考区間。"
        "CAGR差はポイント。区間が0を跨ぐなら当該手法での方向の裏付けは弱い。"
        "補正区間の極端分位点にはMonte Carlo誤差もある。\n\n"
        + markdown_table(
            ci[
                [
                    "block_days",
                    "metric",
                    "estimate",
                    "lower_95",
                    "upper_95",
                    "family_lower_95",
                    "family_upper_95",
                ]
            ]
        )
        + "\n\n"
        "これは実現経路に条件付けた再標本化。価格・共分散・重みを再推定しない。"
        "定常性や代表性を仮定する近似であり、構造変化、過去の未記録のモデル選択、"
        "独立将来の不確実性を全て含まない。正の再標本割合は帰無仮説のp値ではない。"
        "再標本内の局面が0日、またはSharpe計算で2日未満なら当該指標は欠測とし、"
        "有効反復数をbootstrap.csvへ保存。95%未満の反復しか有効でなければ区間を表示しない。"
        "今回、全期間のCAGR差・Sharpe差は3種類のブロック長でいずれも通常95%区間が0を跨いだ。"
        "従って全期間の正の点推定についても、統計的に優位とまでは判断しない。",
        "## 局面による差の違い\n\n"
        "63日ブロックで、条件付き年率対数超過収益の「株上昇−下落」「債上昇−下落」"
        "「相関正−非正」を比較する。値と区間は年率対数ポイント。"
        "今回これら3つの通常95%区間はいずれも0を含む。観測上の成績差はあるが、"
        "母集団での局面依存を統計的に確定したという主張もできない。\n\n"
        + table(
            contrasts,
            ["group", "estimate", "lower_95", "upper_95", "family_lower_95", "family_upper_95"],
            ["estimate", "lower_95", "upper_95", "family_lower_95", "family_upper_95"],
        ),
        "## 成績がどこで生じたか\n\n"
        "4トレンド群の年率対数超過収益への寄与。各群の累計log差×252/全日数。"
        "4行の合計が全期間log差へ一致する。CAGR差ではない。\n\n"
        + table(contribution, ["group", "annual_log_contribution"], ["annual_log_contribution"])
        + "\n\n"
        f"既存のパラメータ感応度{len(sensitivity)}組も同条件で戦略間を直接結合してCSVへ保存。"
        "これらは同じ過去データ上の設定比較で、追加の独立標本ではない。",
        "## 金融工学上の解釈と未確認事項\n\n"
        "最小分散は推定共分散の下で分散を小さくする目的関数を解く。"
        "将来リターン・Sharpe・DDの最大化を保証する理論ではない。"
        "VT倍率は配分と予測ボラに依存し、相関変化・推定誤差・倍率上限・資金調達条件が"
        "両戦略へ異なる影響を与え得る。これは解釈で、個別局面の損益原因を同定した結果ではない。\n\n"
        "仮説を「この過去標本の全期間でCAGRが高かった」に限定すれば確認できる。"
        "「市場環境に依存しない」「投信として採用できる」まで拡張するには、"
        "独立将来の固定ルール検証、別市場・期間、point-in-timeデータ、実際の調達・先物/ETF実装、"
        "スプレッド・税・流動性・強制縮小・運用制約、推定手法の安定性が不足する。"
        "Yahoo調整価格と現在の3ETF選択にはデータ改訂・銘柄選択の限界がある。"
        "同一の過去成績を繰り返し参照しており、売買の過去情報制約を守っても研究選択の独立OOSにはならない。\n\n"
        "参考：[Novy-Marx, Backtesting Strategies Based on Multiple Signals](https://www.nber.org/papers/w21329)"
        "（過適合・複数比較の問題。今回の区間がそれら全てを補正するという意味ではない）。",
        "## 再現・成果物\n\n"
        "`python scripts/check_market_independence.py`。Notebook10は前日局面・直接比較・"
        "63日ブートストラップを実計算し、4図を表示する。CSV/Parquet/監査JSONは"
        "`市場環境別VT直接比較/`、図は`outputs/figures/市場環境別VT直接比較/`。"
        "単体テストは合成データによる計算検証で、本文の実市場結果と区別する。\n\n"
        "[Notebook原本](../../学習ノート/10_最小分散VTの市場環境依存性.ipynb)、"
        "実行済みNotebookは`分析結果/実行済みノート/`、"
        "テストログ・XMLはローカル生成物としてGit対象外。"
        "公開前検証は[検証記録](../../資料/公開前監査と検証.md)を参照。",
    ]
    path = output.parent / "市場環境に依存しない優位性の検証.md"
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
