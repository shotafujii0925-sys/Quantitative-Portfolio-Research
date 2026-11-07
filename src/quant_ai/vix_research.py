"""VIXボラ目標の研究。計画は資料/VIXボラ目標_研究計画.mdに固定し、結果から設定を選ばない。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.downside_metrics import evaluate_leveraged
from quant_ai.leveraged_backtest import LeveragedResult, leveraged_backtest
from quant_ai.vix_volatility import (
    BASES,
    FORECASTS,
    align_vix,
    base_weight_schedule,
    forecast_table,
    qlike,
    rescaled_cagr,
    vix_schedule,
)

OUTPUT = "outputs/reports/VIXボラ目標"
EPISODES = {
    "金融危機の下落": ("2007-10-09", "2009-03-09"),
    "金融危機後の反発": ("2009-03-10", "2010-03-09"),
    "コロナの下落": ("2020-02-19", "2020-03-23"),
    "コロナ後の反発": ("2020-03-24", "2021-03-23"),
    "2022年の株債同時下落": ("2022-01-03", "2022-10-12"),
}
PERIODS = {
    "2006-2012": ("2006", "2012"),
    "2013-2019": ("2013", "2019"),
    "2020-2026": ("2020", "2026"),
}
METRICS = ("annual_return", "volatility", "sharpe", "max_drawdown")
PRIMARY_BLOCK = 63
SENSITIVITY_FORECASTS = ("hist252", "rv21", "vix_cal")
REALLOCATED = "min_variance__vix_cal_alloc"


def paired_bootstrap(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str]],
    block: int,
    repetitions: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """同じ循環ブロックで全列を再標本化し、差の95%区間と両側pを返す。

    p = min(1, 2*min(P(差<=0), P(差>=0)))。得られた収益系列に条件付きで配分を再推定しない。
    指標はCAGR、年率ボラ、Sharpe(rf=0)、最大DD。
    """
    if block < 1 or repetitions < 100 or len(frame) < 2:
        raise ValueError("Positive block, at least 100 repetitions and 2 observations required")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Finite solvent returns required")
    n, columns = values.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, columns, len(METRICS)))
    for i in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        sample = values[((starts[:, None] + np.arange(block)) % n).ravel()[:n]]
        log_wealth = np.log1p(sample).cumsum(axis=0)
        vol = sample.std(axis=0, ddof=1) * np.sqrt(252)
        peak = np.maximum.accumulate(np.maximum(log_wealth, 0), axis=0)
        draws[i] = np.column_stack(
            [
                np.expm1(log_wealth[-1] * 252 / n),
                vol,
                sample.mean(axis=0) * 252 / vol,
                np.expm1(log_wealth - peak).min(axis=0),
            ]
        )
    rows = []
    for strategy, baseline in pairs:
        difference = (
            draws[:, frame.columns.get_loc(strategy)] - draws[:, frame.columns.get_loc(baseline)]
        )
        for k, metric in enumerate(METRICS):
            d = difference[:, k]
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": baseline,
                    "metric": metric,
                    "block_days": block,
                    "repetitions": repetitions,
                    "lower_95": float(np.quantile(d, 0.025)),
                    "upper_95": float(np.quantile(d, 0.975)),
                    "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                }
            )
    return pd.DataFrame(rows)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm補正。小さい順に(m-i)倍し、単調性を保って1で打ち切る。"""
    order = p_values.sort_values().index
    m = len(p_values)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * float(p_values[key])))
        adjusted[key] = running
    return pd.Series(adjusted).reindex(p_values.index)


def bootstrap_mean_difference(
    difference: np.ndarray,
    block: int = 12,
    repetitions: int = 10000,
    seed: int = 42,
) -> dict:
    """平均差の循環ブロック区間と両側p。月次の予測損失の差に使う（12か月ブロック）。"""
    n = len(difference)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        means[i] = difference[((starts[:, None] + np.arange(block)) % n).ravel()[:n]].mean()
    return {
        "mean": float(difference.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))),
    }


def forecast_accuracy(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SPYの次の21日実現ボラに対する予測精度と、QLIKE差の区間。"""
    names = ["hist252", "rv21", "vix_cal", "vix_raw"]
    realized = table["realized"].to_numpy()
    rows = []
    for name in names:
        forecast = table[name].to_numpy()
        error = np.log(forecast) - np.log(realized)
        rows.append(
            {
                "forecast": name,
                "months": len(table),
                "qlike": float(qlike(forecast, realized).mean()),
                "log_mse": float((error**2).mean()),
                "log_bias": float(error.mean()),
                "log_r2": float(np.corrcoef(np.log(forecast), np.log(realized))[0, 1] ** 2),
            }
        )
    differences = []
    for a, b in [("vix_cal", "rv21"), ("vix_cal", "hist252"), ("rv21", "hist252")]:
        d = qlike(table[a].to_numpy(), realized) - qlike(table[b].to_numpy(), realized)
        differences.append({"forecast": a, "baseline": b, **bootstrap_mean_difference(d)})
    return pd.DataFrame(rows), pd.DataFrame(differences)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """価格とVIXを読み込み、記録されたハッシュと一致することを確認する。"""
    provenance = {}
    frames = {}
    for key, name in [("prices", "spy_tlt_gld_adjusted"), ("vix", "vix_close")]:
        path = root / f"data/processed/{name}.parquet"
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Snapshot hash differs from the recorded source: {name}")
        frames[key] = pd.read_parquet(path)
        provenance[key] = info
    prices = frames["prices"][["SPY", "TLT", "GLD"]]
    vix = align_vix(frames["vix"]["VIX"], prices)
    return prices, vix, provenance


def run_case(
    prices: pd.DataFrame,
    vix: pd.Series,
    base_cache: dict,
    base: str,
    forecast: str,
    *,
    target: float = 0.10,
    cap: float | None = 1.5,
    cash: float = 0.0,
    borrow: float = 0.0,
    cost: float = 5.0,
    frequency: str = "M",
    reallocate: bool = False,
) -> tuple[LeveragedResult, pd.DataFrame, dict]:
    """1条件を計算し、純収益・監査表・指標を返す。"""
    if base not in base_cache:
        base_cache[base] = base_weight_schedule(prices, base, 252, "2006-01-01")
    targets, audit = vix_schedule(
        prices,
        vix,
        base,
        forecast,
        target_volatility=target,
        leverage_cap=cap,
        frequency=frequency,
        reallocate=reallocate,
        base_targets=base_cache[base],
    )
    result = leveraged_backtest(
        prices.loc[targets.index[0] :], targets, cost_bps=cost, cash_rate=cash, borrow_rate=borrow
    )
    stats = evaluate_leveraged(result, target if forecast != "none" else None, cap)
    stats["cagr_at_10pct_vol"] = rescaled_cagr(result.daily.net_return)
    stats["mean_target_exposure"] = float(audit.target_exposure.mean())
    stats["cap_binding_fraction"] = float(audit.cap_binding.mean())
    return result, audit, stats


def placebo_sharpe(
    prices: pd.DataFrame,
    vix: pd.Series,
    base_cache: dict,
    base: str,
    step: int = 21,
    minimum_shift: int = 252,
) -> dict:
    """VIXを循環シフトし、市場との対応だけを壊す。p=(1+#{Sharpe>=実績})/(1+N)。片側。"""
    _, _, actual = run_case(prices, vix, base_cache, base, "vix_cal")
    shifts = np.arange(minimum_shift, len(vix) - minimum_shift, step)
    values = np.empty(len(shifts))
    for i, shift in enumerate(shifts):
        rolled = pd.Series(np.roll(vix.to_numpy(), shift), index=vix.index, name="VIX")
        _, _, stats = run_case(prices, rolled, base_cache, base, "vix_cal")
        values[i] = stats["sharpe"]
    return {
        "base": base,
        "sharpe": actual["sharpe"],
        "placebo_median": float(np.median(values)),
        "p_one_sided": float((1 + (values >= actual["sharpe"]).sum()) / (1 + len(values))),
        "n_shifts": len(values),
    }


def run_vix_study(root: Path, *, repetitions: int = 10000) -> dict:
    """計画どおりの全条件を計算し、CSVとJSONの素材を辞書で返す。"""
    prices, vix, provenance = load_inputs(root)
    base_cache: dict = {}
    rows, daily, audits = [], {}, {}

    def record(case_id: str, family: str, base: str, forecast: str, **kwargs) -> None:
        result, audit, stats = run_case(prices, vix, base_cache, base, forecast, **kwargs)
        settings = {
            "target": 0.10,
            "cap": 1.5,
            "cash": 0.0,
            "borrow": 0.0,
            "cost": 5.0,
            "frequency": "M",
            **kwargs,
        }
        settings.pop("reallocate", None)
        rows.append(
            {"case_id": case_id, "family": family, "base": base, "forecast": forecast}
            | settings
            | stats
        )
        if family == "primary":
            daily[case_id], audits[case_id] = result, audit

    for base in BASES:
        for forecast in FORECASTS:
            record(f"{base}__{forecast}", "primary", base, forecast)
    record(REALLOCATED, "primary", "min_variance", "vix_cal", reallocate=True)
    print("Primary cases complete", flush=True)

    sensitivity = {
        "target8": {"target": 0.08},
        "target12": {"target": 0.12},
        "uncapped": {"cap": None},
        "cap1": {"cap": 1.0},
        "funding23": {"cash": 0.02, "borrow": 0.03},
        "funding46": {"cash": 0.04, "borrow": 0.06},
        "cost20": {"cost": 20.0},
        "weekly": {"frequency": "W"},
    }
    for name, kwargs in sensitivity.items():
        for base in BASES:
            for forecast in SENSITIVITY_FORECASTS:
                record(f"{name}__{base}__{forecast}", name, base, forecast, **kwargs)
    # 計画後に追加: SPY買い持ちとの比較のため、配分も変更する版の金利と上限の感度を見る。
    for name in ("funding23", "funding46", "cap1"):
        record(
            f"{name}__{REALLOCATED}",
            name,
            "min_variance",
            "vix_cal",
            reallocate=True,
            **sensitivity[name],
        )
    print("Sensitivity complete", flush=True)
    summary = pd.DataFrame(rows).set_index("case_id")

    frame = pd.DataFrame({key: result.daily.net_return for key, result in daily.items()})
    primary_pairs = [(f"{b}__vix_cal", f"{b}__rv21") for b in BASES]
    secondary = [
        (f"{b}__{x}", f"{b}__{y}")
        for b in BASES
        for x, y in [("vix_cal", "hist252"), ("vix_cal", "none"), ("rv21", "hist252")]
        + [("hist252", "none")]
    ] + [(REALLOCATED, "min_variance__vix_cal")]
    # 計画後に追加: 「SharpeとCAGRの両方」を確かめるためのSPY買い持ちとの比較。
    secondary += [
        (name, "spy_only__none")
        for name in ("min_variance__rv21", "min_variance__vix_cal", REALLOCATED)
    ]
    bootstrap = pd.concat(
        [
            paired_bootstrap(frame, primary_pairs + secondary, block, repetitions)
            for block in (21, PRIMARY_BLOCK, 126)
        ],
        ignore_index=True,
    )
    primary = bootstrap[
        (bootstrap.block_days == PRIMARY_BLOCK)
        & (bootstrap.metric == "sharpe")
        & bootstrap.strategy.isin([p[0] for p in primary_pairs])
        & bootstrap.baseline.str.endswith("__rv21")
    ].copy()
    primary["p_holm"] = holm(primary.p_two_sided)
    print("Bootstrap complete", flush=True)

    table = forecast_table(prices, vix)
    accuracy, accuracy_differences = forecast_accuracy(table)

    placebo = pd.DataFrame([placebo_sharpe(prices, vix, base_cache, base) for base in BASES])
    print("Placebo complete", flush=True)

    period_rows, episode_rows = [], []
    for key, returns in frame.items():
        for name, (start, end) in PERIODS.items():
            part = returns.loc[start:end]
            volatility = part.std(ddof=1) * np.sqrt(252)
            period_rows.append(
                {
                    "case_id": key,
                    "period": name,
                    "annual_return": float(np.expm1(np.log1p(part).mean() * 252)),
                    "volatility": float(volatility),
                    "sharpe": float(part.mean() * 252 / volatility),
                }
            )
        for name, (start, end) in EPISODES.items():
            part = returns.loc[start:end]
            episode_rows.append(
                {"case_id": key, "episode": name, "total_return": float((1 + part).prod() - 1)}
            )
    exposure_rows = []
    for name, (start, _) in EPISODES.items():
        if "下落" not in name:
            continue
        for key, audit in audits.items():
            before = audit.loc[audit.execution <= pd.Timestamp(start)]
            exposure_rows.append(
                {
                    "case_id": key,
                    "episode": name,
                    "decision": str(before.index[-1].date()),
                    "vix": float(before.vix.iloc[-1]),
                    "target_exposure": float(before.target_exposure.iloc[-1]),
                }
            )
    exposures = pd.concat(
        [audit.target_exposure.rename(key) for key, audit in audits.items()], axis=1
    )
    provenance.update(
        research_is_post_selection=True,
        bootstrap_reestimates_weights=False,
        evaluation_start=str(frame.index[0].date()),
        evaluation_end=str(frame.index[-1].date()),
        observations=len(frame),
    )
    return {
        "summary": summary,
        "bootstrap": bootstrap,
        "primary": primary,
        "accuracy": accuracy,
        "accuracy_differences": accuracy_differences,
        "placebo": placebo,
        "periods": pd.DataFrame(period_rows),
        "episodes": pd.DataFrame(episode_rows),
        "exposure_before": pd.DataFrame(exposure_rows),
        "exposures": exposures,
        "frame": frame,
        "provenance": provenance,
    }


def save_vix_study(bundle: dict, root: Path) -> None:
    """集計はCSV、日次はParquetでローカルへ保存する。公開範囲は公開監査で管理する。"""
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for key, value in bundle.items():
        if key in {"frame", "provenance"}:
            continue
        value.to_csv(output / f"{key}.csv")
    bundle["frame"].to_parquet(output / "daily_net_returns.parquet")
    (output / "provenance.json").write_text(
        json.dumps(bundle["provenance"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
