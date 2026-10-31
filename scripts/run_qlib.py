"""プロジェクト直下で python scripts/run_qlib.py。合成データのみを使用。"""

from pathlib import Path

from quant_ai.qlib_workflow import build_sample_provider, run_qlib_demo

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_qlib_demo(
        build_sample_provider(root / "data/qlib_synthetic"), root / "分析結果/レポート"
    )
    print(result["risk"])
