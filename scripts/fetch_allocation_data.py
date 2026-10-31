"""比較研究に使う小容量の日次データを取得し、前処理の記録を残す。"""

import argparse
import hashlib
import json
from pathlib import Path

from quant_ai.data_loader import close_prices, download_prices

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """日付・欠損を監査し、実データのみを保存。合成データに代替しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2004-11-18")
    parser.add_argument("--end", default="2026-10-30", help="Exclusive end; default ends on Oct 29")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace local snapshot"
    )
    args = parser.parse_args()
    target = ROOT / "data/processed/spy_tlt_gld_adjusted.parquet"
    if target.exists() and not args.overwrite:
        raise FileExistsError("Existing snapshot preserved; use --overwrite only intentionally")
    raw = download_prices(["SPY", "TLT", "GLD"], args.start, args.end, cache_dir=ROOT / "data/raw")
    close = raw["Close"].reindex(columns=["SPY", "TLT", "GLD"])
    missing = close.isna().sum().to_dict()
    prices = close_prices(raw, missing="drop").reindex(columns=["SPY", "TLT", "GLD"])
    target.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(target)
    metadata = {
        "kind": "yahoo_observed_adjusted_prices",
        "currency": "USD",
        "provenance": raw.attrs["provenance"],
        "missing_by_asset": missing,
        "dropped_dates": [str(d.date()) for d in close.index if d not in prices.index],
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "rows": len(prices),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
