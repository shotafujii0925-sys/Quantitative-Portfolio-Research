"""計算済みの数値から日本語レポートを生成。事実・統計・解釈・推測を区別する。"""

from pathlib import Path

import pandas as pd

from quant_ai.risk_research import RiskResearch

METHOD_NAMES = {"equal_weight": "等ウェイト", "inverse_vol": "逆ボラ", "min_variance": "最小分散"}


def markdown_table(frame: pd.DataFrame) -> str:
    """追加依存なしで表示済み文字列の表をMarkdownへ変換する。"""
    header = "|" + "|".join(map(str, frame.columns)) + "|"
    separator = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    lines = ["|" + "|".join(str(value) for value in row) + "|" for row in frame.to_numpy()]
    return "\n".join([header, separator, *lines])


def write_research_report(bundle: RiskResearch, root: Path) -> Path:
    """推測を実証結果として扱わず、同じ予測目標でも実現リスクが異なる点を強調する。"""
    primary = bundle.summary[bundle.summary.family.isin(["original", "primary"])]
    displayed = []
    for name, row in primary.iterrows():
        condition = (
            "統一前"
            if name.startswith("original")
            else ("10%・無上限" if name.startswith("unbounded") else "10%・月次上限1.5")
        )
        displayed.append(
            {
                "条件": condition,
                "戦略": METHOD_NAMES[row.method],
                "CAGR": f"{row.annual_return:.2%}",
                "実現年率ボラ": f"{row.volatility:.2%}",
                "Sharpe(rf=0)": f"{row.sharpe:.3f}",
                "最大DD": f"{row.max_drawdown:.2%}",
                "平均保有倍率": f"{row.mean_held_leverage:.3f}",
                "最大実現倍率": f"{row.max_realized_leverage:.3f}",
            }
        )
    full_risk = []
    for name, row in primary.iterrows():
        recovery = (
            "未回復・打切り"
            if pd.isna(row.trough_to_recovery_days)
            else f"{int(row.trough_to_recovery_days)}暦日"
        )
        full_risk.append(
            {
                "ケース": name,
                "Sortino(MAR=0)": f"{row.sortino:.3f}",
                "Calmar": f"{row.calmar:.3f}",
                "日次ES95損失": f"{row.daily_es95_loss:.2%}",
                "最大DDの底→回復": recovery,
                "回復日": row.recovery_date,
                "ボラ−目標": "対象外"
                if pd.isna(row.realized_minus_target_volatility)
                else f"{row.realized_minus_target_volatility * 100:+.2f}pt",
            }
        )
    costs = []
    for name, row in primary.iterrows():
        costs.append(
            {
                "ケース": name,
                "年率売買/NAV": f"{row.annual_turnover:.3f}",
                "累計売買費用/初期資金": f"{row.total_trade_cost_initial_capital:.4f}",
                "無コストCAGR": f"{row.gross_path_annual_return:.2%}",
                "費用によるCAGR低下": f"{row.cost_cagr_drag * 100:.3f}pt",
                "借入がある日": f"{row.borrowing_day_fraction:.1%}",
                "上限超過ドリフト日": f"{row.cap_drift_breach_day_fraction:.1%}",
            }
        )
    sensitivity = []
    for _, row in bundle.summary[
        bundle.summary.family.isin(
            [
                "target_sensitivity",
                "lookback_sensitivity",
                "funding_sensitivity",
                "cost_sensitivity",
            ]
        )
        & (bundle.summary.method == "min_variance")
        & (bundle.summary.leverage_cap == 1.5)
    ].iterrows():
        sensitivity.append(
            {
                "比較群": row.family,
                "窓": row.lookback,
                "目標": f"{row.target_volatility:.0%}",
                "片道bps": row.cost_bps,
                "預金/借入年率": f"{row.cash_rate:.0%}/{row.borrow_rate:.0%}",
                "CAGR": f"{row.annual_return:.2%}",
                "ボラ": f"{row.volatility:.2%}",
                "最大DD": f"{row.max_drawdown:.2%}",
                "累計借入費用/初期資金": f"{row.total_funding_cost_initial_capital:.4f}",
                "開始": row.start,
            }
        )
    statistical = bundle.bootstrap[
        (bundle.bootstrap.strategy.str.endswith("min_variance"))
        & (bundle.bootstrap.metric.isin(["annual_return", "volatility", "sharpe", "max_drawdown"]))
    ].copy()
    for column in ["lower_95", "upper_95"]:
        statistical[column] = statistical[column].map(lambda x: f"{x:.4f}")
    period_view = bundle.periods[bundle.periods.case_id.str.startswith("cap15_")][
        ["case_id", "period", "annual_return", "volatility", "max_drawdown"]
    ].copy()
    for column in ["annual_return", "volatility", "max_drawdown"]:
        period_view[column] = period_view[column].map(lambda x: f"{x:.2%}")
    reconciliation_max = max(
        v["daily_return_max_abs_difference"] for v in bundle.reconciliation.values()
    )
    accounting = float(primary.accounting_max_error.max())
    return_intervals = bundle.bootstrap[bundle.bootstrap.metric.isin(["annual_return", "sharpe"])]
    all_cross_zero = bool(
        ((return_intervals.lower_95 <= 0) & (return_intervals.upper_95 >= 0)).all()
    )
    empirical_uncertainty = (
        "今回、CAGR差とSharpe差は全ての比較・両ブロック長で区間が0をまたいだ。"
        "リターン・Sharpeの優位を統計的に確立したとは言えない。"
        if all_cross_zero
        else "一部の区間は0をまたがないが、条件付き・多重比較・後付け設計という制約が残る。"
    )
    comparison_lines = []
    for prefix in ["unbounded", "cap15"]:
        baseline = primary.loc[f"{prefix}_equal_weight"]
        minimum = primary.loc[f"{prefix}_min_variance"]
        comparison_lines.append(
            f"- {prefix}の最小分散−等ウェイト：CAGR差"
            f"{(minimum.annual_return - baseline.annual_return) * 100:+.3f}pt、"
            f"ボラ差{(minimum.volatility - baseline.volatility) * 100:+.3f}pt、"
            f"最大DD差{(minimum.max_drawdown - baseline.max_drawdown) * 100:+.3f}pt"
            "（DD差は正が改善）。"
        )
    contents = [
        "# リスク水準統一によるSPY・TLT・GLD戦略比較\n\n2026-10-31実行。実データによる追加研究。",
        "## 結論の要点\n\n"
        f"10%目標・月次上限1.5倍では、等ウェイトのCAGRは"
        f"{primary.loc['cap15_equal_weight', 'annual_return']:.2%}、最小分散は"
        f"{primary.loc['cap15_min_variance', 'annual_return']:.2%}。一方、実現ボラは"
        f"{primary.loc['cap15_equal_weight', 'volatility']:.2%}と"
        f"{primary.loc['cap15_min_variance', 'volatility']:.2%}で完全には一致せず、最大DDは"
        f"{primary.loc['cap15_equal_weight', 'max_drawdown']:.2%}と"
        f"{primary.loc['cap15_min_variance', 'max_drawdown']:.2%}。"
        "点推定の収益増加だけで低リスク・優位な商品とは判断しない。" + empirical_uncertainty,
        "## 研究の位置付け\n\n目標8/10/12%、1.5倍制約、推定窓・金利条件は既存の2006〜2026年結果を"
        "見た後に設定した。過去情報のみのhistorical walk-forwardだが、"
        "独立した未見将来の検証ではない。"
        "いずれかの戦略の優位を前提にしない。**同一の予測目標と、同一の実現リスクは別である。**",
        "## データと実行条件\n\n"
        "元のYahoo/yfinance調整価格スナップショットを再利用。USD、配当再投資相当、"
        "価格観測2004-11-18〜2026-10-29。評価2006-01-03〜2026-10-29、5239営業日。"
        "10月30日は元取得時に3ETFとも欠損のため除外済み。新たな取得・補完はしていない。\n\n"
        f"入力SHA256：`{bundle.provenance['sha256']}`。"
        "基本は252日推定、月次判断、翌共通営業日終値約定、片道5bps、預金/借入0%。"
        "窓比較126/252/504日は全条件で2007年開始にそろえる。"
        "金利は固定仮想シナリオ2%/3%・4%/6%（年実効、ACT/365）。実際の歴史的金利ではない。",
        "## 計算式と約定\n\n"
        "基本配分b、過去の資産共分散Σからσhat=sqrt(252*b'Σ*b)。L=目標/σhat、"
        "w=L*b。制約ケースはL=min(L,1.5)。未来の実現ボラを重みに使用しない。\n\n"
        "終値直前の保有金額H^-とキャッシュC^-を、旧保有の価格変化・預金利息・借入利息から計算。"
        "V^+=V^-−cΣ|w_i V^+−H_i^-|を解き、約定後保有H_i^+=w_i V^+、"
        "C^+=V^+−ΣH_i^+とする。負のCは借入。"
        "売買コストはNAVではなく各資産の実売買金額にかかる。"
        "[数値解法SciPy 1.15.3 brentq](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.brentq.html)。\n\n"
        "**1.5倍は月次約定時の目標上限。日中・月中の厳格な維持上限ではない。**"
        "月中の下落・利息で倍率が増える場合を記録した。マージンコール・日次強制縮小は未実装。",
        "## データから直接確認した結果\n\n"
        + markdown_table(pd.DataFrame(displayed))
        + "\n\n"
        + "\n".join(comparison_lines),
        "## 下方リスクと回復\n\n"
        + markdown_table(pd.DataFrame(full_risk))
        + "\n\n回復は最大DDの底からピーク回復までの暦日。ピーク日・底日・回復日と"
        "ピークからの期間もCSVに保存。未回復を0日とせず打切りとして扱う。",
        "## 費用とレバレッジ\n\n"
        + markdown_table(pd.DataFrame(costs))
        + "\n\n累計費用は初期資金1に対する金額で、各時点NAV比率を単純に足した値とは違う。"
        "無コスト成績は同じ金利・同じ目標ルールを費用0で再実行した経路。"
        "CAGRやDDをレバレッジで単純乗算していない。",
        "## 指標の定義\n\n"
        "CAGR=(Π(1+r_net))^(252/n)−1。Sharpe(rf=0)=sqrt(252)*mean(r_net)/std(r_net,ddof=1)。"
        "分子は日次の算術平均で、CAGRではない。金利感応度には預金機会収益を日次で引いた"
        "sharpe_excess_cashも保存。SortinoはMAR0、sqrt(252)*mean(r)/sqrt(mean(min(r,0)^2))、"
        "下方偏差の分母は全観測数。Calmar=CAGR/|最大DD|。"
        "日次ES95損失=下位ceil(5%*n)個の平均リターンの符号反転。年率化しない。"
        "実現レバレッジは各日の旧保有・約定前・約定後NAV比率から観測。"
        "予測との乖離は全期間ボラ−目標と63日実現ボラRMSEを併記。",
        "## 感応度分析\n\n以下は最小分散・上限1.5倍の抜粋。"
        "他戦略・無上限を含む全条件は`リスク統一比較/all_scenarios.csv`。"
        "良い設定を選び直す材料とはしない。\n\n" + markdown_table(pd.DataFrame(sensitivity)),
        "## 固定市場局面\n\n区間内CAGR・区間内DDである。"
        "この表は10%・月次上限1.5倍。\n\n" + markdown_table(period_view),
        "## 統計的に支持できる範囲\n\n21/63営業日の循環ブロックを全戦略で共有して各2000回再標本化。"
        "下表は最小分散−同じ上限の等ウェイト、95%記述的区間。CAGR・DDは小数単位。\n\n"
        + markdown_table(statistical)
        + "\n\n"
        + empirical_uncertainty
        + " 区間が0をまたぐ差を優位性の証拠としない。"
        "0をまたがない場合でも、これは既存リターンに条件付けた区間であり、"
        "配分再推定・非定常性・研究者の事後選択・多重比較を十分反映しない。"
        "**一般的な戦略優位性や将来収益を統計的に確立したとは扱わない。**",
        "## 金融理論による解釈\n\n"
        "期待リスクで揃えると、元の低ボラ戦略はより多くのリスク資産を持つことがある。"
        "そのため元の分散低減が、そのまま絶対損失低減になるわけではない。"
        "借入と変動するNAV、複利、推定誤差、月中ドリフトのためCAGR・最大DDは倍率に単純比例しない。"
        "目標低下と上限制約は資金調達や回転率の影響も変える。",
        "## 推測にとどまる説明\n\n"
        "株式・長期債の相関変化や特定局面の金利環境が差を生んだ可能性はある。"
        "相関図は記述であり、今回因果寄与を分解したものではない。"
        "同じ標本共分散で最小分散を選び、その共分散で倍率も決める構造では、"
        "推定誤差により予測リスクが楽観的になり得る。目標超過の要因別寄与は未分解。"
        "レジーム分類やパラメータを結果に合わせて変更すればさらに選択バイアスが増える。",
        "## 投資信託として採用できるか\n\n"
        "**現時点では採用可と判断できない。研究用候補としての比較まで。**\n\n"
        "不足しているのは、変更後に独立した将来/別市場での確認、実際の時変金利・資金調達契約、"
        "流動性と売買代金・スプレッド・市場インパクト、税・FX・販売管理報酬、"
        "引け前の注文数量確定と終値からの約定ずれ、"
        "上限超過時の縮小/マージンルール、ストレステスト、解約資金・商品規約との整合、"
        "運用担当と独立リスク管理の体制である。各商品の法的・約款上の実行可能性は未確認。"
        "金融庁の投資運用業向け監督指針もリスク管理・流動性等の確認を含むが、"
        "この研究がそれを満たすと判定したものではない。"
        "[金融庁・投資運用業の監督指針](https://www.fsa.go.jp/common/law/guide/kinyushohin/06.html)。",
        "## 整合性・再現\n\n"
        f"元の100%投資3戦略との日次純リターン最大差は{reconciliation_max:.3g}、"
        f"NAV会計恒等式の最大残差は{accounting:.3g}。"
        "差は浮動小数点の計算順序の範囲で、旧結果を変更していない。"
        f"異なる経路は{bundle.provenance['distinct_paths']}本、"
        f"比較群を含む集計行は{bundle.provenance['reported_scenarios']}行。\n\n"
        "`python scripts/run_risk_normalized_study.py`で全計算・図・本報告を生成。"
        "`--render-only`は保存済み数値から図と報告だけを生成する。"
        "実データ結果と合成単体テストは区別する。テストログはoutputs/reportsに保存。"
        "Notebook07はコード・式・結果、Notebook08は既存分析の可視化と実装案内。",
    ]
    if (root / "outputs/reports/戦略比較と収益分解_分析レポート.md").exists():
        contents.append(
            "## 関連する追加比較\n\n"
            "[非レバ3戦略・VT3戦略と収益分解](戦略比較と収益分解_分析レポート.md)では、"
            "同じデータを使い、資産配分・倍率変更・借入費用・預金利息・追加売買費用を区別した。"
            "元の本報告の数値は維持している。"
        )
    target = root / "outputs/reports/リスク統一比較_分析レポート.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join(contents) + "\n", encoding="utf-8")
    return target
