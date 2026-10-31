"""少量の実データ取得、公式サンプル容量照会、公式converterの実行検証。"""

import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quant_ai.data_loader import close_prices, download_prices, synthetic_prices

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """各検証を分離し、失敗した接続・実行をJSONに記録。大規模取得しない。"""
    result = {"checked_at_utc": datetime.now(UTC).isoformat()}
    try:
        frame = download_prices(
            ["SPY", "MSFT"], "2024-01-01", "2024-02-01", cache_dir=ROOT / "data/raw"
        )
        prices = close_prices(frame)
        result["yahoo"] = {
            "status": "passed",
            "rows": len(prices),
            "tickers": list(prices.columns),
            "provenance": frame.attrs["provenance"],
        }
    except Exception as exc:
        result["yahoo"] = {"status": "failed", "error": repr(exc)}
    try:
        url = "https://api.github.com/repos/SunsetWolf/qlib_dataset/releases/tags/v2"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        assets = [
            {"name": a["name"], "compressed_bytes": a["size"], "url": a["browser_download_url"]}
            for a in response.json()["assets"]
        ]
        result["official_dataset"] = {
            "status": "metadata_only",
            "downloaded": False,
            "source": url,
            "assets": assets,
            "extracted_bytes": "unknown; inspect zip before extraction",
        }
    except Exception as exc:
        result["official_dataset"] = {"status": "failed", "downloaded": False, "error": repr(exc)}
    try:
        url = "https://raw.githubusercontent.com/microsoft/qlib/v0.9.7/scripts/dump_bin.py"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        folder = ROOT / "分析結果/公式変換ツール"
        folder.mkdir(parents=True, exist_ok=True)
        script = folder / "dump_bin.py"
        script.write_bytes(response.content)
        csv_dir = folder / "csv"
        csv_dir.mkdir(exist_ok=True)
        prices = synthetic_prices(80, 2)
        for symbol in prices:
            close = prices[symbol] / prices[symbol].iloc[0]
            pd.DataFrame(
                {
                    "date": prices.index.strftime("%Y-%m-%d"),
                    "close": close,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": 100000,
                    "factor": 1 / prices[symbol].iloc[0],
                }
            ).to_csv(csv_dir / f"{symbol}.csv", index=False)
        provider = folder / "provider"
        command = [
            sys.executable,
            str(script),
            "dump_all",
            "--data_path",
            str(csv_dir),
            "--qlib_dir",
            str(provider),
            "--include_fields",
            "open,close,high,low,volume,factor",
            "--max_workers",
            "1",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
        )
        (folder / "conversion.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"Converter exit={completed.returncode}; see conversion.log")
        values = np.fromfile(provider / "features/syn00/close.day.bin", dtype="<f4")
        np.testing.assert_allclose(values[1:], prices.iloc[:, 0] / prices.iloc[0, 0], rtol=1e-6)
        result["official_converter"] = {
            "status": "passed",
            "source": url,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "data": "synthetic_only",
            "rows": len(values) - 1,
        }
    except Exception as exc:
        result["official_converter"] = {"status": "failed", "error": repr(exc)}
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    torch = next(p for p in lock["package"] if p["name"] == "torch")
    wheels = []
    for wheel in torch.get("wheels", []):
        if "cp311" not in wheel["url"] or "aarch64" in wheel["url"]:
            continue
        entry = {"url": wheel["url"], "bytes": wheel.get("size")}
        try:
            response = requests.head(wheel["url"], allow_redirects=True, timeout=30)
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            entry["bytes"] = int(length) if length else None
            entry["size_method"] = "HTTP HEAD, no wheel download"
        except Exception as exc:
            entry["size_error"] = repr(exc)
        wheels.append(entry)
    result["torch_optional"] = {
        "version": torch["version"],
        "installed": False,
        "cpu_wheels": wheels,
        "note": "Wheel sizes exclude dependencies and extracted files",
    }
    output = ROOT / "分析結果/レポート/external_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
