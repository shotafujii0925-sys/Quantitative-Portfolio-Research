"""Yahoo Finance取得・明示的欠損処理・Parquetキャッシュと出典記録。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    cache_dir: Path = Path("data/raw"),
    adjusted: bool = True,
    refresh: bool = False,
    max_age_days: float = 7,
) -> pd.DataFrame:
    """日次OHLCVを取得。endは排他的、auto_adjust=Trueは配当・分割調整。

    UTC取得時刻、全パラメータ、出典、yfinance版をJSON記録。過去の構成銘柄や
    point-in-time調整系列を保証しない。raw Closeを総収益率と解釈しない。
    """
    if not tickers or max_age_days < 0 or pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("Invalid tickers, interval or cache age")
    request = {
        "tickers": sorted(set(tickers)),
        "start": start,
        "end_exclusive": end,
        "auto_adjust": adjusted,
        "interval": "1d",
        "actions": True,
        "yfinance_version": yf.__version__,
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, meta = cache_dir / f"{key}.parquet", cache_dir / f"{key}.json"
    if path.exists() and meta.exists() and not refresh:
        info = json.loads(meta.read_text(encoding="utf-8"))
        age = (datetime.now(UTC) - datetime.fromisoformat(info["retrieved_at_utc"])).total_seconds()
        if age <= max_age_days * 86400:
            frame = pd.read_parquet(path)
            frame.attrs["provenance"] = info
            return frame
    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    frame = yf.download(
        request["tickers"],
        start=start,
        end=end,
        auto_adjust=adjusted,
        actions=True,
        progress=False,
        threads=False,
        timeout=20,
        group_by="column",
        multi_level_index=True,
    )
    if frame.empty:
        raise RuntimeError("Yahoo returned no data; synthetic data is a separate workflow")
    frame = frame.sort_index()
    if not frame.index.is_unique:
        raise ValueError("Duplicate dates from provider")
    missing = [
        t
        for t in request["tickers"]
        if t not in frame["Close"].columns or frame["Close"][t].dropna().empty
    ]
    if missing:
        raise RuntimeError(f"Missing tickers: {missing}")
    info = {
        **request,
        "source": "Yahoo Finance via yfinance",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(frame),
        "warning": "Retrospectively adjusted; not point-in-time; licensing applies",
    }
    frame.to_parquet(path)
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.attrs["provenance"] = info
    return frame


def close_prices(
    frame: pd.DataFrame,
    missing: Literal["raise", "drop", "ffill"] = "raise",
    max_gap: int = 1,
) -> pd.DataFrame:
    """Close抽出。欠損は明示選択、ffillは過去値のみ最大max_gap日。

    ffillは休場・売買停止を取引可能と誤認し得る。バックテストでは原則raise。
    """
    prices = frame["Close"].copy()
    if missing == "drop":
        prices = prices.dropna(how="any")
    elif missing == "ffill":
        prices = prices.ffill(limit=max_gap)
    elif missing != "raise":
        raise ValueError("Unknown missing policy")
    if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Unresolved missing/nonpositive prices")
    return prices


def synthetic_prices(rows: int = 700, assets: int = 5, seed: int = 42) -> pd.DataFrame:
    """再現可能な合成市場 P_t=100 exp(Σ(β f_t+ε_t))。実市場の証拠ではない。"""
    if rows < 10 or assets < 1:
        raise ValueError("rows>=10 and assets>=1 required")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.009, (rows, 1))
    innovations = rng.normal(0, 0.006, (rows, assets))
    logs = market * np.linspace(0.7, 1.3, assets) + innovations
    return pd.DataFrame(
        100 * np.exp(np.cumsum(logs, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=rows),
        columns=[f"SYN{i:02d}" for i in range(assets)],
    )
