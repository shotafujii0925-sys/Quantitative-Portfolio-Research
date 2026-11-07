"""VIXボラ目標の全計算を実行し、ローカルへ保存する。設定は研究計画に固定する。"""

import os
from pathlib import Path

from quant_ai.vix_research import run_vix_study, save_vix_study

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """計画の全条件を計算する。結果から設定を選び直さない。"""
    os.environ["MPLBACKEND"] = "Agg"
    bundle = run_vix_study(ROOT)
    save_vix_study(bundle, ROOT)
    print(bundle["primary"].to_string(), flush=True)
    print(f"Saved to {ROOT.name}/outputs/reports/VIXボラ目標", flush=True)


if __name__ == "__main__":
    main()
