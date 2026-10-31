"""同じデータで非レバ3戦略・VT3戦略を比較し、収益源を追加分析する。"""

import argparse
import os
from pathlib import Path

from quant_ai.return_attribution import load_attribution_study, run_attribution_study

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """既存結果を変更せずoutputs/reports/戦略比較と収益分解へ新規出力。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    study = load_attribution_study(ROOT) if args.render_only else run_attribution_study(ROOT)
    import matplotlib.pyplot as plt

    from quant_ai.attribution_outputs import attribution_figures, write_attribution_report

    attribution_figures(study, ROOT)
    plt.close("all")
    print(write_attribution_report(study, ROOT))
    print(
        study.performance[
            [
                "method",
                "mode",
                "rate_scenario",
                "annual_return",
                "volatility",
                "max_drawdown",
                "sharpe",
            ]
        ].to_string()
    )
    print(study.bridges.to_string())


if __name__ == "__main__":
    main()
