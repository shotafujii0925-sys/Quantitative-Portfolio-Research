"""新規依存・再取得なしで、リスク統一比較の全計算・図・日本語レポートを出力する。"""

import argparse
import os
from pathlib import Path

from quant_ai.risk_research import load_risk_research, run_risk_research, save_risk_research

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """実データ分析を実行。旧分析は読取のみ。生成先はoutputs/。"""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        bundle = load_risk_research(ROOT)
    else:
        bundle = run_risk_research(ROOT)
        save_risk_research(bundle, ROOT)
    import matplotlib.pyplot as plt
    import pandas as pd

    from quant_ai.research_plots import research_figures
    from quant_ai.research_report import write_research_report

    prices = pd.read_parquet(ROOT / "data/processed/spy_tlt_gld_adjusted.parquet")
    research_figures(bundle, prices, ROOT / "outputs/figures/リスク統一比較")
    plt.close("all")
    print(write_research_report(bundle, ROOT), flush=True)
    print(
        bundle.summary[bundle.summary.family.isin(["original", "primary"])][
            [
                "annual_return",
                "volatility",
                "sharpe",
                "sortino",
                "max_drawdown",
                "daily_es95_loss",
                "mean_held_leverage",
                "max_realized_leverage",
            ]
        ].to_string()
    )
    print("Saved numerical results to outputs/reports/リスク統一比較", flush=True)


if __name__ == "__main__":
    main()
