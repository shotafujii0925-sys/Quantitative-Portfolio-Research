"""容量を確認した小さな公式CSV配布だけを取得し、公式converterで検証する。"""

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """2MiB圧縮・20MiB展開上限。CSVのみを明示保存し、任意パス展開しない。"""
    output = ROOT / "分析結果/レポート/official_sample_check.json"
    result = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "data_kind": "official_distributed_csv_sample_not_a_profitability_test",
    }
    try:
        meta = requests.get(
            "https://api.github.com/repos/SunsetWolf/qlib_dataset/releases/tags/v2", timeout=30
        )
        meta.raise_for_status()
        asset = next(a for a in meta.json()["assets"] if a["name"] == "csv_data_cn.zip")
        if asset["size"] > 2 * 1024**2:
            raise ValueError("Sample exceeds the small-download limit")
        response = requests.get(asset["browser_download_url"], timeout=30, stream=True)
        response.raise_for_status()
        blob = bytearray()
        for chunk in response.iter_content(65536):
            blob.extend(chunk)
            if len(blob) > 2 * 1024**2:
                raise ValueError("Response exceeded download limit")
        root = ROOT / "data/qlib_official_sample"
        marker = root / "OFFICIAL_SAMPLE.json"
        if root.exists() and any(root.iterdir()) and not marker.exists():
            raise FileExistsError("Unknown sample directory; refusing to overwrite")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"source": asset["browser_download_url"]}), encoding="utf-8")
        csv_dir = root / "csv"
        csv_dir.mkdir(exist_ok=True)
        names = []
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = archive.infolist()
            if sum(x.file_size for x in entries) > 20 * 1024**2:
                raise ValueError("Sample exceeds extracted-size limit")
            for info in entries:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive path")
                if info.is_dir() or path.suffix.lower() != ".csv":
                    continue
                if path.name.lower() in names:
                    raise ValueError("Duplicate CSV name")
                data = archive.read(info)
                frame = pd.read_csv(io.BytesIO(data))
                required = {"date", "open", "close", "high", "low", "volume"}
                if not required.issubset(frame.columns):
                    raise ValueError(f"Unexpected columns: {list(frame.columns)}")
                raw_dir = root / "raw"
                raw_dir.mkdir(exist_ok=True)
                (raw_dir / path.name.lower()).write_bytes(data)
                frame = frame.sort_values("date").reset_index(drop=True)
                if frame["date"].duplicated().any():
                    raise ValueError("Duplicate sample dates")
                if "factor" not in frame:
                    if "adjclose" not in frame:
                        raise ValueError("No adjustment information")
                    # Yahoo adjclose/close由来の事後調整。約定用の実価格ではない。
                    factor = (frame["adjclose"] / frame["close"]) / frame["adjclose"].iloc[0]
                    if not np.isfinite(factor).all() or (factor <= 0).any():
                        raise ValueError("Invalid sample adjustment factor")
                    for field in ("open", "close", "high", "low"):
                        frame[field] *= factor
                    frame["volume"] /= factor
                    frame["factor"] = factor
                frame.to_csv(csv_dir / path.name.lower(), index=False)
                names.append(path.name.lower())
        if not names:
            raise ValueError("No CSV in sample")
        script = ROOT / "分析結果/公式変換ツール/dump_bin.py"
        expected = "b8f34c57ce1ef4b1772f3909735e66058f21b25bd7ab8a5f16318822401fe53f"
        if hashlib.sha256(script.read_bytes()).hexdigest() != expected:
            raise ValueError("Run check_external.py first; official converter hash mismatch")
        provider = root / "provider"
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
        (ROOT / "分析結果/レポート/official_sample_conversion.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"Converter exit {completed.returncode}")
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D

        qlib.init(
            provider_uri=str(provider),
            region=REG_CN,
            kernels=1,
            expression_cache=None,
            dataset_cache=None,
            logging_level=30,
        )
        instruments = D.list_instruments(D.instruments("all"), as_list=True)
        features = D.features(instruments[:2], ["$close", "$close/Ref($close,5)-1"])
        if features.dropna().empty:
            raise ValueError("No valid official sample features")
        features.head(20).to_csv(ROOT / "分析結果/レポート/official_sample_features.csv")
        result.update(
            status="passed",
            source=asset["browser_download_url"],
            compressed_bytes=len(blob),
            extracted_bytes=sum(x.file_size for x in entries),
            csv_files=names,
            instruments=instruments,
            feature_rows=len(features),
            archive_sha256=hashlib.sha256(blob).hexdigest(),
            transformation="factor=(adjclose/close)/first_adjclose; OHLC*=factor; volume/=factor",
            note="Official distributed demo: provenance/completeness not independently audited",
        )
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
