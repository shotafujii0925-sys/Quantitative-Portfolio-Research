# ruff: noqa: E501  # レポートの本文テンプレートは折り返さない
"""VIXボラ目標の図とレポートを保存済みの集計から生成する。統計値は手で書かない。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/reports/VIXボラ目標"
FIGURES = ROOT / "outputs/figures/VIXボラ目標"
REPORT = ROOT / "outputs/reports/VIXボラ目標_分析レポート.md"

BASES = ["spy_only", "equal_weight", "min_variance"]
BASE_LABELS = {"spy_only": "SPYのみ", "equal_weight": "等ウェイト", "min_variance": "最小分散"}
FORECASTS = ["none", "hist252", "rv21", "vix_cal", "vix_raw"]
FORECAST_LABELS = {
    "none": "ボラ目標なし",
    "hist252": "過去252日",
    "rv21": "過去21日",
    "vix_cal": "VIX（校正）",
    "vix_raw": "VIX（校正なし）",
}
COLORS = {
    "none": "#52514e",
    "hist252": "#2a78d6",
    "rv21": "#eb6834",
    "vix_cal": "#1baf7a",
    "vix_raw": "#eda100",
}
ALLOC = "min_variance__vix_cal_alloc"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SENSITIVITY_LABELS = {
    "target8": "目標ボラ8%",
    "target12": "目標ボラ12%",
    "uncapped": "倍率上限なし",
    "cap1": "倍率上限1倍",
    "funding23": "金利2%/3%",
    "funding46": "金利4%/6%",
    "cost20": "片道20bps",
    "weekly": "週次判断",
}
plt.rcParams.update(
    {
        "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def check(condition: object, message: str) -> None:
    """本文の定性的な主張がデータと合わなければ、生成を止める。"""
    if not bool(condition):
        raise AssertionError(f"本文の主張がデータと一致しません: {message}")


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}"


def pval(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["|" + "|".join(str(c) for c in row) + "|" for row in rows]
    return "\n".join(lines)


def load() -> dict:
    data = {
        name: pd.read_csv(RESULTS / f"{name}.csv", index_col=0)
        for name in (
            "summary",
            "bootstrap",
            "primary",
            "accuracy",
            "accuracy_differences",
            "placebo",
            "periods",
            "episodes",
            "exposure_before",
            "exposures",
        )
    }
    data["provenance"] = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))
    return data


def figures(data: dict) -> None:
    """3枚の図を保存する。数値はすべて表にも載せる。"""
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = data["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [
        ("annual_return", "CAGR（%）", 100),
        ("sharpe", "Sharpe（rf=0）", 1),
        ("max_drawdown", "最大DD（%）", 100),
    ]
    width = 0.16
    for ax, (column, title, scale) in zip(axes, specs, strict=True):
        for i, forecast in enumerate(FORECASTS):
            values = [s.loc[f"{b}__{forecast}", column] * scale for b in BASES]
            ax.bar(
                np.arange(3) + (i - 2) * width,
                values,
                width * 0.92,
                color=COLORS[forecast],
                label=FORECAST_LABELS[forecast],
            )
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([BASE_LABELS[b] for b in BASES])
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(FIGURES / "01_予測ボラ別の成績.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    acc = data["accuracy"].set_index("forecast")
    names = ["hist252", "rv21", "vix_cal", "vix_raw"]
    axes[0].bar(
        [FORECAST_LABELS[n] for n in names],
        [acc.loc[n, "qlike"] for n in names],
        color=[COLORS[n] for n in names],
        width=0.6,
    )
    axes[0].set_title("SPYの次の21日ボラの予測: VIXの誤差が小さい")
    axes[0].set_ylabel("QLIKE（小さいほど良い）")
    p = data["primary"].set_index("strategy")
    keys = [f"{b}__vix_cal" for b in BASES]
    y = np.arange(3)[::-1]
    points = [
        data["summary"].loc[k, "sharpe"]
        - data["summary"].loc[k.replace("vix_cal", "rv21"), "sharpe"]
        for k in keys
    ]
    lower = [points[i] - p.loc[k, "lower_95"] for i, k in enumerate(keys)]
    upper = [p.loc[k, "upper_95"] - points[i] for i, k in enumerate(keys)]
    axes[1].errorbar(points, y, xerr=[lower, upper], fmt="o", ms=8, color="#2a78d6", capsize=3)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([BASE_LABELS[b] for b in BASES])
    axes[1].set_title("ボラ目標のSharpe差（VIX − 過去21日、95%区間）: 差は確認できない")
    axes[1].set_xlabel("Sharpe差")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_予測精度とSharpe差.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    exposure = data["exposures"].copy()
    exposure.index = pd.to_datetime(exposure.index)
    for forecast in ["hist252", "rv21", "vix_cal"]:
        axes[0].plot(
            exposure.index,
            exposure[f"spy_only__{forecast}"],
            color=COLORS[forecast],
            lw=1.5,
            label=FORECAST_LABELS[forecast],
        )
    axes[0].set_title("SPYのみの月次目標倍率")
    axes[0].set_ylabel("倍率")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    episodes = data["episodes"].pivot(index="case_id", columns="episode", values="total_return")
    order = ["金融危機の下落", "コロナの下落", "2022年の株債同時下落", "コロナ後の反発"]
    for i, forecast in enumerate(["none", "rv21", "vix_cal"]):
        values = [episodes.loc[f"spy_only__{forecast}", e] * 100 for e in order]
        axes[1].bar(
            np.arange(4) + (i - 1) * 0.26,
            values,
            0.24,
            color=COLORS[forecast],
            label=FORECAST_LABELS[forecast],
        )
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(["金融危機\n下落", "コロナ\n下落", "2022年\n下落", "コロナ後\n反発"])
    axes[1].set_title("SPYのみの局面別リターン: 下落を抑え、反発も取り逃がす")
    axes[1].set_ylabel("期間リターン（%）")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_倍率と局面別の損益.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build(data: dict) -> str:
    s = data["summary"]
    boot = data["bootstrap"]
    primary = data["primary"].set_index("strategy")
    acc = data["accuracy"].set_index("forecast")
    diffs = data["accuracy_differences"].set_index(["forecast", "baseline"])
    placebo = data["placebo"].set_index("base")
    eps = data["episodes"].pivot(index="case_id", columns="episode", values="total_return")
    before = data["exposure_before"].set_index(["case_id", "episode"])
    prov = data["provenance"]

    def b(strategy: str, baseline: str, metric: str, block: int = 63) -> pd.Series:
        rows = boot[
            (boot.strategy == strategy)
            & (boot.baseline == baseline)
            & (boot.metric == metric)
            & (boot.block_days == block)
        ]
        check(len(rows) == 1, f"bootstrap row {strategy} {baseline} {metric}")
        return rows.iloc[0]

    def ci(row: pd.Series, scale: float = 1.0, digits: int = 3) -> str:
        return f"[{row.lower_95 * scale:+.{digits}f}, {row.upper_95 * scale:+.{digits}f}]"

    # 判定条件の検証
    significant = [k for k in primary.index if primary.loc[k, "p_holm"] < 0.05]
    check(not significant, "主要な比較はHolm補正後に有意でない")
    for base in BASES:
        for block in (21, 63, 126):
            row = b(f"{base}__vix_cal", f"{base}__rv21", "sharpe", block)
            check(row.lower_95 < 0 < row.upper_95, "主要な比較の区間は全ブロック長で0を含む")
    q = diffs.loc[("vix_cal", "rv21")]
    check(q.upper_95 < 0, "VIXのQLIKEは過去21日より小さい")
    check(diffs.loc[("vix_cal", "hist252")].upper_95 < 0, "VIXのQLIKEは過去252日より小さい")
    check(acc.loc["vix_cal", "log_r2"] > acc.loc["rv21", "log_r2"], "VIXのlog R²が高い")
    check(acc.qlike.idxmin() == "vix_raw", "QLIKEは校正なしVIXが最小")
    check(acc.loc["vix_raw", "log_mse"] > acc.loc["vix_cal", "log_mse"], "log誤差は校正版が小さい")
    check(
        s.loc["spy_only__none", "annual_return"] > s.loc["spy_only__vix_cal", "annual_return"], ""
    )
    check(b("spy_only__vix_cal", "spy_only__none", "max_drawdown").lower_95 > 0, "SPYのDDは改善")
    check(
        eps.loc["spy_only__none", "コロナ後の反発"]
        > 3 * eps.loc["spy_only__vix_cal", "コロナ後の反発"],
        "コロナ後の反発はボラ目標なしが大きく上回る",
    )
    check(
        eps.loc["spy_only__vix_cal", "コロナの下落"] > eps.loc["spy_only__none", "コロナの下落"],
        "コロナの下落はボラ目標で軽減",
    )
    check(
        before.loc[("min_variance__vix_cal", "コロナの下落"), "target_exposure"] == 1.5, "上限到達"
    )
    alloc = s.loc[ALLOC]
    spy = s.loc["spy_only__none"]
    check(
        alloc.annual_return > spy.annual_return and alloc.sharpe > spy.sharpe, "点推定で両方上回る"
    )
    check(
        s.loc[f"funding46__{ALLOC}", "annual_return"] < spy.annual_return, "金利4/6%でCAGRは下回る"
    )
    check(
        s.loc[f"funding23__{ALLOC}", "annual_return"] < spy.annual_return, "金利2/3%でCAGRは下回る"
    )
    check(s.loc[f"cap1__{ALLOC}", "annual_return"] < spy.annual_return, "借入なしでCAGRは下回る")
    alloc_cagr = b(ALLOC, "spy_only__none", "annual_return")
    check(alloc_cagr.lower_95 < 0 < alloc_cagr.upper_95, "SPYとのCAGR差は0をまたぐ")
    alloc_sharpe = b(ALLOC, "spy_only__none", "sharpe")
    alloc_mdd = b(ALLOC, "spy_only__none", "max_drawdown")
    check(alloc_mdd.lower_95 > 0, "SPYより最大DDは小さい")
    other = b(ALLOC, "min_variance__vix_cal", "sharpe")
    check(other.lower_95 < 0 < other.upper_95, "配分変更の効果は区間で確認できない")
    check(
        (
            s.loc[[f"{x}__{y}" for x in BASES for y in FORECASTS], "accounting_max_error"] < 1e-9
        ).all(),
        "会計恒等式",
    )

    rows = []
    for base in BASES:
        for forecast in FORECASTS:
            r = s.loc[f"{base}__{forecast}"]
            rows.append(
                [
                    BASE_LABELS[base],
                    FORECAST_LABELS[forecast],
                    pct(r.annual_return),
                    pct(r.volatility),
                    num(r.sharpe),
                    pct(r.max_drawdown),
                    pct(r.cagr_at_10pct_vol),
                    num(r.mean_target_exposure, 2),
                    num(r.annual_turnover, 2),
                ]
            )
    rows.append(
        [
            "最小分散",
            "VIX（校正、配分も変更）",
            pct(alloc.annual_return),
            pct(alloc.volatility),
            num(alloc.sharpe),
            pct(alloc.max_drawdown),
            pct(alloc.cagr_at_10pct_vol),
            num(alloc.mean_target_exposure, 2),
            num(alloc.annual_turnover, 2),
        ]
    )
    t_main = table(
        [
            "基本配分",
            "SPYのボラ予測",
            "CAGR",
            "実現ボラ",
            "Sharpe",
            "最大DD",
            "ボラ10%換算CAGR",
            "平均倍率",
            "年率回転率",
        ],
        rows,
    )

    t_primary = table(
        ["基本配分", "Sharpe（VIX）", "Sharpe（過去21日）", "差", "95%区間", "両側p", "Holm p"],
        [
            [
                BASE_LABELS[base],
                num(s.loc[f"{base}__vix_cal", "sharpe"]),
                num(s.loc[f"{base}__rv21", "sharpe"]),
                signed(s.loc[f"{base}__vix_cal", "sharpe"] - s.loc[f"{base}__rv21", "sharpe"]),
                ci(primary.loc[f"{base}__vix_cal"]),
                pval(primary.loc[f"{base}__vix_cal", "p_two_sided"]),
                pval(primary.loc[f"{base}__vix_cal", "p_holm"]),
            ]
            for base in BASES
        ],
    )
    t_accuracy = table(
        ["予測", "QLIKE", "log誤差の2乗平均", "logバイアス", "log R²"],
        [
            [
                FORECAST_LABELS[n],
                num(acc.loc[n, "qlike"]),
                num(acc.loc[n, "log_mse"]),
                signed(acc.loc[n, "log_bias"]),
                num(acc.loc[n, "log_r2"]),
            ]
            for n in ["hist252", "rv21", "vix_cal", "vix_raw"]
        ],
    )
    t_placebo = table(
        ["基本配分", "実績Sharpe", "プラセボ中央値", "片側p", "シフト数"],
        [
            [
                BASE_LABELS[base],
                num(placebo.loc[base, "sharpe"]),
                num(placebo.loc[base, "placebo_median"]),
                pval(placebo.loc[base, "p_one_sided"]),
                int(placebo.loc[base, "n_shifts"]),
            ]
            for base in BASES
        ],
    )
    pairs = [
        ("spy_only__vix_cal", "spy_only__none", "SPYのみ: VIX − 目標なし"),
        ("min_variance__vix_cal", "min_variance__none", "最小分散: VIX − 目標なし"),
        ("min_variance__rv21", "spy_only__none", "最小分散・過去21日 − SPY買い持ち"),
        ("min_variance__vix_cal", "spy_only__none", "最小分散・VIX − SPY買い持ち"),
        (ALLOC, "spy_only__none", "最小分散・VIX配分 − SPY買い持ち"),
        (ALLOC, "min_variance__vix_cal", "最小分散: VIX配分 − VIX倍率のみ"),
    ]
    t_pairs = table(
        ["比較", "CAGR差（pt）", "Sharpe差", "最大DD差（pt、正が改善）"],
        [
            [
                label,
                ci(b(x, y, "annual_return"), 100, 2),
                ci(b(x, y, "sharpe")),
                ci(b(x, y, "max_drawdown"), 100, 1),
            ]
            for x, y, label in pairs
        ],
    )
    order = [
        "金融危機の下落",
        "金融危機後の反発",
        "コロナの下落",
        "コロナ後の反発",
        "2022年の株債同時下落",
    ]
    t_episode = table(
        ["基本配分", "予測"] + order,
        [
            [BASE_LABELS[base], FORECAST_LABELS[f]]
            + [pct(eps.loc[f"{base}__{f}", e], 1) for e in order]
            for base in BASES
            for f in ["none", "hist252", "rv21", "vix_cal"]
        ],
    )
    decline = ["金融危機の下落", "コロナの下落", "2022年の株債同時下落"]
    t_before = table(
        ["基本配分", "予測"] + decline,
        [
            [BASE_LABELS[base], FORECAST_LABELS[f]]
            + [num(before.loc[(f"{base}__{f}", e), "target_exposure"], 2) for e in decline]
            for base in BASES
            for f in ["hist252", "rv21", "vix_cal"]
        ],
    )
    vix_before = {e: num(before.loc[("spy_only__vix_cal", e), "vix"], 1) for e in decline}
    sens_rows, wins, total = [], 0, 0
    for family, label in SENSITIVITY_LABELS.items():
        row = [label]
        for base in BASES:
            x = s.loc[f"{family}__{base}__vix_cal", "sharpe"]
            y = s.loc[f"{family}__{base}__rv21", "sharpe"]
            wins += int(x > y)
            total += 1
            row.append(f"{num(x)} / {num(y)}")
        sens_rows.append(row)
    t_sens = table(
        ["条件（Sharpe: VIX / 過去21日）"] + [BASE_LABELS[base] for base in BASES], sens_rows
    )
    fund = table(
        ["条件", "CAGR", "実現ボラ", "Sharpe", "最大DD"],
        [
            [
                label,
                pct(s.loc[key, "annual_return"]),
                pct(s.loc[key, "volatility"]),
                num(s.loc[key, "sharpe"]),
                pct(s.loc[key, "max_drawdown"]),
            ]
            for label, key in [
                ("SPY買い持ち（比較対象）", "spy_only__none"),
                ("金利0%", ALLOC),
                ("金利2%/3%", f"funding23__{ALLOC}"),
                ("金利4%/6%", f"funding46__{ALLOC}"),
                ("倍率上限1倍（借入なし）", f"cap1__{ALLOC}"),
            ]
        ],
    )
    ew_old = s.loc["equal_weight__hist252"]
    mv_old = s.loc["min_variance__hist252"]

    return f"""# VIXはボラ目標の成績を改善するか

{prov["evaluation_start"]}〜{prov["evaluation_end"]}（{prov["observations"]:,}営業日）。SPY・TLT・GLD、USD。価格は既存研究の取得版ではなく2026-11-07に再取得した版です。

> 本文の統計値はすべて `scripts/build_vix_report.py` が集計から差し込んでいます。実装と本文の作成にはAI（Claude）を使用しました。過去のバックテストであり、将来の成績や投資の推奨を示すものではありません。

## 結論

**VIXはSPYの将来ボラをよく予測しますが、ボラ目標の成績では過去21日の実現ボラと区別できませんでした。** 事前に決めた判定条件は満たしていません。

- **予測精度:** QLIKEはVIX（校正）{num(acc.loc["vix_cal", "qlike"])} に対し過去21日 {num(acc.loc["rv21", "qlike"])} で、差 {signed(q["mean"])} [{signed(q.lower_95)}, {signed(q.upper_95)}] は0を除外します。
- **運用成績:** VIXと過去21日のSharpe差は、3つの基本配分のすべてで区間が0を含み、Holm補正後のpは1.000です。
- **SharpeとCAGRの両立:** 最小分散でVIXを配分にも使う版は、金利0%の点推定ではSPY買い持ちをCAGR（{pct(alloc.annual_return)} 対 {pct(spy.annual_return)}）とSharpe（{num(alloc.sharpe)} 対 {num(spy.sharpe)}）の両方で上回ります。ただしCAGR差の区間は0を含み、金利を入れるか借入をなくすとCAGRはSPYを下回ります。
- **確かに言える効果:** SPYのみでは、ボラ目標は最大DDを {pct(spy.max_drawdown, 1)} から {pct(s.loc["spy_only__vix_cal", "max_drawdown"], 1)} に縮めます。代償はCAGRの低下（{pct(s.loc["spy_only__vix_cal", "annual_return"])}）で、コロナ後の反発の大半も取り逃がします。

## 既存研究との一致の確認

`hist252` は既存のボラ目標と同じ計算です（テストで一致を確認）。再取得した価格でも、既存の公表値に近い結果になりました。

| 戦略 | 公表値（CAGR / ボラ / Sharpe / 最大DD） | 今回 |
|---|---|---|
| 等ウェイト＋VT | 9.08% / 10.69% / 0.867 / -24.16% | {pct(ew_old.annual_return)} / {pct(ew_old.volatility)} / {num(ew_old.sharpe)} / {pct(ew_old.max_drawdown)} |
| 最小分散＋VT | 10.46% / 10.89% / 0.968 / -25.39% | {pct(mv_old.annual_return)} / {pct(mv_old.volatility)} / {num(mv_old.sharpe)} / {pct(mv_old.max_drawdown)} |

差は調整済み価格の改訂によるものと考えられます。ただし原因は特定していません。

## 条件

条件と判定基準は [研究計画](../../資料/VIXボラ目標_研究計画.md) に記録しています。

- 月次判断、翌営業日終値で約定、片道5bps、目標ボラ10%、月次目標倍率の上限1.5倍、預金・借入金利0%
- SPYのボラ予測だけを置き換え、TLT・GLDのボラと相関は過去252日で推定
- VIX（校正）は VIX × exp(k)。kは判断日までに結果が確定した日の log(実現ボラ / VIX) の平均

## 全条件の成績

{t_main}

ボラ10%換算CAGRは事後のスケーリングによる記述的な値で、取引可能な戦略ではありません。

![予測ボラ別の成績](../figures/VIXボラ目標/01_予測ボラ別の成績.png)

## 主要な比較: VIX − 過去21日

63日ペア循環ブロック・ブートストラップ（{int(boot.repetitions.iloc[0]):,}回）です。21日と126日のブロックでも、すべて区間は0を含みました。

{t_primary}

**判定:** Holm補正後に有意な基本配分がないので、計画の条件1を満たしません。したがって「VIXには過去21日の実現ボラを超える価値がある」とは判定しません。区間の幅を考えると、「差がない」と証明したわけでもありません。

## 予測精度

SPYの次の21日の実現ボラに対する、月末ごとの予測（{int(acc.loc["vix_cal", "months"])}か月）です。

{t_accuracy}

QLIKE差（12か月ブロック）は、VIX（校正）− 過去21日が {signed(q["mean"])} [{signed(q.lower_95)}, {signed(q.upper_95)}]、p={pval(q.p_two_sided)} です。VIX（校正）− 過去252日は {signed(diffs.loc[("vix_cal", "hist252"), "mean"])} [{signed(diffs.loc[("vix_cal", "hist252"), "lower_95"])}, {signed(diffs.loc[("vix_cal", "hist252"), "upper_95"])}] です。

校正なしのVIXはQLIKEが最小ですが、QLIKEは過小予測を重く罰する指標です。logバイアスは {signed(acc.loc["vix_raw", "log_bias"])} と大きく、log誤差の2乗平均では校正版より劣るので、実現ボラを高めに見積もっています。そのため倍率が下がり、実現ボラも {pct(s.loc["spy_only__vix_raw", "volatility"])}（SPYのみ）と目標を下回ります。

![予測精度とSharpe差](../figures/VIXボラ目標/02_予測精度とSharpe差.png)

予測が良くても成績に結びつかなかった理由は、この研究では特定していません。考えられるのは、ボラ目標の成績がボラ予測の精度だけでなく、倍率とその後のリターンの関係にも左右されること、月次判断では短期のボラ予測の利点が薄れること、過去21日でも主な変動を捉えていることです。週次判断でも結論は変わりませんでした。

## VIXのプラセボ

VIXを循環シフトし、市場との対応だけを壊した場合との比較です。

{t_placebo}

SPYのみと最小分散では、VIXは時期をずらした系列より良い結果でした（補正なし）。VIXに情報がないわけではなく、過去21日の実現ボラと比べた上乗せが確認できないという結果です。

## SharpeとCAGRの両立

差の95%区間です（63日ブロック、補正なしの参考値）。SPY買い持ちとの比較は、最初の結果を見た後に追加しました。

{t_pairs}

配分もVIXで変える版の条件別の成績です。

{fund}

金利0%で両方の点推定が上回るのは、平均 {num(alloc.mean_target_exposure, 2)} 倍の倍率を無料で持てる前提によるところが大きいです。金利シナリオは期間全体で固定した仮定で、実際の金利の推移ではありません。また、この版は複数の候補のうち最も高かった点推定であり、選択の影響を含みます。SPY買い持ちとの差のうち、区間で確認できたのは最大DDの改善です（CAGR差 {ci(alloc_cagr, 100, 2)} pt、Sharpe差 {ci(alloc_sharpe)}、最大DD差 {ci(alloc_mdd, 100, 1)} pt）。

## ショック前の倍率と局面別の損益

各下落の直前の判断日の目標倍率です。その日のVIXはそれぞれ {vix_before["金融危機の下落"]}、{vix_before["コロナの下落"]}、{vix_before["2022年の株債同時下落"]} でした。

{t_before}

局面は結果の分類だけに使っており、売買の判断には使っていません。

{t_episode}

- **弱点の確認:** 下落の直前はVIXが低く、最小分散では倍率が上限の1.5倍に達していた局面があります。計画時の懸念のとおりで、VIXはショックを予告しません。
- **下落の軽減:** SPYのみでは、コロナの下落を {pct(eps.loc["spy_only__none", "コロナの下落"], 1)} から {pct(eps.loc["spy_only__vix_cal", "コロナの下落"], 1)} に抑えました。
- **反発の取り逃がし:** その後1年の反発は、ボラ目標なしの {pct(eps.loc["spy_only__none", "コロナ後の反発"], 1)} に対し {pct(eps.loc["spy_only__vix_cal", "コロナ後の反発"], 1)} でした。

![倍率と局面別の損益](../figures/VIXボラ目標/03_倍率と局面別の損益.png)

## 感度分析

報告のみで、ここから設定を選んでいません。VIXのSharpeが過去21日を上回ったのは、{total}条件のうち{wins}条件です。

{t_sens}

## 限界

- 既存の結果を知ったうえでの事後研究で、独立した検証ではありません。評価期間も既存研究と同じです。
- VIXの傾きを1に固定した根拠は、同じ期間と重なる別研究の結果です。
- ブートストラップは得られた収益系列に条件付きで、配分の再推定を含みません。主要な比較以外は多重比較を補正していません。
- 金利は固定シナリオで、実際の金利の推移、税、マージンコール、月中の強制縮小は含みません。
- Sharpeはrf=0です。金利シナリオでは預金利息が分子に入るため、異なる金利シナリオ間のSharpeを直接比較しないでください。
- SPY・TLT・GLDは米国上場ETFで、2006年以降の特定の期間に依存します。TLTとGLDの過去の成績が将来も続くとは限りません。
- 調整済み価格は再取得で変わります。元データは配布していません。

## 再現

```bash
uv run --no-sync python scripts/fetch_allocation_data.py
uv run --no-sync python scripts/fetch_vix_data.py
uv run --no-sync python scripts/run_vix_study.py
uv run --no-sync python scripts/build_vix_report.py
```

取得版: 価格 SHA256 `{prov["prices"]["sha256"]}`、VIX SHA256 `{prov["vix"]["sha256"]}`。
"""


def main() -> None:
    data = load()
    figures(data)
    text = build(data)
    REPORT.write_text(text, encoding="utf-8", newline=chr(10))
    print(f"Saved {REPORT.name}: {len(text):,} characters")


if __name__ == "__main__":
    main()
