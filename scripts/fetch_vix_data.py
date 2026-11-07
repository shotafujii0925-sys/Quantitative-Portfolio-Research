"""VIXの日次終値を取得し、前処理の記録を残す。合成データに代替しない。"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_ai.data_loader import download_prices

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """既存の価格研究と同じ窓で取得する。上書きは明示した場合だけ行う。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2004-11-18")
    parser.add_argument("--end", default="2026-10-30", help="Exclusive end; default ends on Oct 29")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace local snapshot"
    )
    args = parser.parse_args()
    target = ROOT / "data/processed/vix_close.parquet"
    if target.exists() and not args.overwrite:
        raise FileExistsError("Existing snapshot preserved; use --overwrite only intentionally")
    raw = download_prices(["^VIX"], args.start, args.end, cache_dir=ROOT / "data/raw")
    close = raw["Close"]
    series = close["^VIX"] if isinstance(close, pd.DataFrame) else close
    vix = series.dropna().rename("VIX").to_frame()
    target.parent.mkdir(parents=True, exist_ok=True)
    vix.to_parquet(target)
    metadata = {
        "kind": "yahoo_observed_vix_close",
        "provenance": raw.attrs["provenance"],
        "missing": int(series.isna().sum()),
        "start": str(vix.index[0].date()),
        "end": str(vix.index[-1].date()),
        "rows": len(vix),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
