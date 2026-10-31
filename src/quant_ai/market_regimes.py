"""最小分散VT対等ウェイトVTの市場局面依存性。将来の局面を売買に使わない。"""

import numpy as np
import pandas as pd

from quant_ai.downside_metrics import expected_shortfall
from quant_ai.metrics import performance

GROUPS = (
    "all",
    "stock_up_bond_up",
    "stock_up_bond_down",
    "stock_down_bond_up",
    "stock_down_bond_down",
    "correlation_positive",
    "correlation_nonpositive",
    "stock_vol_high",
    "stock_vol_low",
)
LABELS = {
    "all": "全期間",
    "stock_up_bond_up": "株↑・債↑",
    "stock_up_bond_down": "株↑・債↓",
    "stock_down_bond_up": "株↓・債↑",
    "stock_down_bond_down": "株↓・債↓",
    "correlation_positive": "株債相関 > 0",
    "correlation_nonpositive": "株債相関 ≤ 0",
    "stock_vol_high": "株式ボラ > 20%",
    "stock_vol_low": "株式ボラ ≤ 20%",
}


def market_states(prices: pd.DataFrame) -> pd.DataFrame:
    """t日の局面はt-1終値までの情報。252日騰落・相関と63日SPYボラ。

    トレンドP[t-1]/P[t-253]-1、相関corr(r_SPY,r_TLT)の252日窓、
    ボラsqrt(252)*std(r_SPY,63日)。閾値は0/20%で事後研究として固定する。
    TLT価格トレンドは債券の収益環境であって、金利変化そのものではない。
    """
    if not {"SPY", "TLT", "GLD"}.issubset(prices.columns):
        raise ValueError("SPY/TLT/GLD required")
    r = prices.pct_change(fill_method=None)
    trend = prices.pct_change(252, fill_method=None).shift(1)
    state = pd.DataFrame(index=prices.index)
    state["stock_trend"] = trend.SPY
    state["bond_trend"] = trend.TLT
    state["correlation"] = r.SPY.rolling(252).corr(r.TLT).shift(1)
    state["stock_vol"] = (r.SPY.rolling(63).std(ddof=1) * np.sqrt(252)).shift(1)
    valid = state.notna().all(axis=1)
    state["all"] = valid
    for stock_up in [True, False]:
        for bond_up in [True, False]:
            key = f"stock_{'up' if stock_up else 'down'}_bond_{'up' if bond_up else 'down'}"
            state[key] = valid & ((trend.SPY > 0) == stock_up) & ((trend.TLT > 0) == bond_up)
    state["correlation_positive"] = valid & (state.correlation > 0)
    state["correlation_nonpositive"] = valid & (state.correlation <= 0)
    state["stock_vol_high"] = valid & (state.stock_vol > 0.20)
    state["stock_vol_low"] = valid & (state.stock_vol <= 0.20)
    return state


def validate_pair(pair: pd.DataFrame) -> None:
    """同じ日付の純収益を厳密に比較し、欠損を暗黙に落とさない。"""
    if list(pair.columns) != ["minimum", "equal"] or len(pair) < 2:
        raise ValueError("Two ordered columns minimum/equal and at least two dates required")
    if not pair.index.is_unique or not pair.index.is_monotonic_increasing:
        raise ValueError("Unique increasing dates required")
    if not np.isfinite(pair.to_numpy()).all() or (pair <= -1).any().any():
        raise ValueError("Finite solvent paired returns required")


def pair_summary(pair: pd.DataFrame) -> dict:
    """同一期間のCAGR差・Sharpe差・リスク。log差=252*mean(log(1+rm)-log(1+re))。

    非連続局面抽出後はCAGRを条件付き幾何年率換算と読む。実行可能な局面戦略ではない。
    """
    validate_pair(pair)
    a, b = (performance(pair[c]) for c in pair)
    return {
        "observations": len(pair),
        "minimum_cagr": float(a.annual_return),
        "equal_cagr": float(b.annual_return),
        "cagr_difference": float(a.annual_return - b.annual_return),
        "minimum_sharpe": float(a.sharpe),
        "equal_sharpe": float(b.sharpe),
        "sharpe_difference": float(a.sharpe - b.sharpe),
        "minimum_vol": float(a.volatility),
        "equal_vol": float(b.volatility),
        "minimum_mdd": float(a.max_drawdown),
        "equal_mdd": float(b.max_drawdown),
        "minimum_es95": expected_shortfall(pair.minimum),
        "equal_es95": expected_shortfall(pair.equal),
        "annual_log_active": float(np.log1p(pair.minimum).sub(np.log1p(pair.equal)).mean() * 252),
    }


def rolling_comparison(pair: pd.DataFrame, window: int) -> pd.DataFrame:
    """過去window日のCAGR差とSharpe差。重複窓の勝率を独立な検定に使わない。"""
    validate_pair(pair)
    if window < 2 or window > len(pair):
        raise ValueError("Window must be between two and sample length")
    cagr = np.expm1(np.log1p(pair).rolling(window).mean() * 252)
    sharpe = pair.rolling(window).mean() / pair.rolling(window).std(ddof=1) * np.sqrt(252)
    return pd.DataFrame(
        {
            "cagr_difference": cagr.minimum - cagr.equal,
            "sharpe_difference": sharpe.minimum - sharpe.equal,
        }
    ).iloc[window - 1 :]


def paired_regime_bootstrap(
    pair: pd.DataFrame,
    states: pd.DataFrame,
    *,
    block_days: int = 63,
    repetitions: int = 10000,
    seed: int = 20261031,
) -> pd.DataFrame:
    """局面と両戦略収益を同じ循環ブロックで再標本化し条件付き区間を返す。

    ブロック内の連続性・戦略間依存を保つ。n日ちょうどで末尾ブロックを切る。
    価格・配分・局面の再推定は行わず、保存された実現経路に条件付ける。
    定常性等が必要で、構造変化・事後選択・未知の将来を保証しない。
    9群×3指標+3局面差=30個にBonferroni補正した両側区間も表示する。
    再標本化で正になった割合を帰無仮説のp値とは呼ばない。
    """
    validate_pair(pair)
    if not pair.index.equals(states.index) or not states["all"].all():
        raise ValueError("Every paired date must have lagged market states")
    if not 1 <= block_days <= len(pair) or repetitions < 100:
        raise ValueError("Valid block length and at least 100 repetitions required")
    mask = states[list(GROUPS)].to_numpy(dtype=float)
    if (mask.sum(axis=0) < 2).any():
        raise ValueError("Each group needs at least two observations")
    v = pair.to_numpy()
    # 群別の十分統計をブロック内で先に足し、配列展開の容量を抑える。
    features = np.column_stack([np.ones(len(v)), np.log1p(v), v, v * v])
    values = mask[:, :, None] * features[:, None, :]
    n = len(v)
    extended = np.concatenate([values, values[:block_days]])
    cumulative = np.concatenate([np.zeros((1, len(GROUPS), 7)), extended.cumsum(axis=0)])
    starts = np.arange(n)
    full = cumulative[starts + block_days] - cumulative[starts]
    quotient, remainder = divmod(n, block_days)
    partial = cumulative[starts + remainder] - cumulative[starts] if remainder else None
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, len(GROUPS) * 3 + 3))

    def statistics(s: np.ndarray) -> np.ndarray:
        """count,sum logm,sum loge,sum rm,sum re,sum rm²,sum re² → 比較指標。"""
        # 稀な局面は再標本内に0日しか現れない場合がある。値を捏造せずNaNとする。
        count = np.where(s[..., 0] > 0, s[..., 0], np.nan)
        logs = s[..., 1:3] / count[..., None] * 252
        means = s[..., 3:5] / count[..., None]
        denominator = np.where(count > 1, count - 1, np.nan)
        variance = (s[..., 5:7] - s[..., 3:5] ** 2 / count[..., None]) / denominator[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = means / np.sqrt(np.maximum(variance, 0)) * np.sqrt(252)
        metrics = np.stack(
            [
                logs[..., 0] - logs[..., 1],
                np.expm1(logs[..., 0]) - np.expm1(logs[..., 1]),
                sr[..., 0] - sr[..., 1],
            ],
            axis=-1,
        ).reshape(*s.shape[:-2], -1)
        contrasts = []
        # 株上昇対下落、債上昇対下落、相関正対非正の条件付きlog超過収益差。
        for positive, negative in [([1, 2], [3, 4]), ([1, 3], [2, 4]), ([5], [6])]:
            pos, neg = s[..., positive, :].sum(axis=-2), s[..., negative, :].sum(axis=-2)
            pos_count = np.where(pos[..., 0] > 0, pos[..., 0], np.nan)
            neg_count = np.where(neg[..., 0] > 0, neg[..., 0], np.nan)
            contrasts.append(
                ((pos[..., 1] - pos[..., 2]) / pos_count - (neg[..., 1] - neg[..., 2]) / neg_count)
                * 252
            )
        return np.concatenate([metrics, np.stack(contrasts, axis=-1)], axis=-1)

    for offset in range(0, repetitions, 64):
        batch = min(64, repetitions - offset)
        selected = rng.integers(n, size=(batch, quotient))
        sums = full[selected].sum(axis=1)
        if partial is not None:
            sums += partial[rng.integers(n, size=batch)]
        draws[offset : offset + batch] = statistics(sums)
    point = statistics(values.sum(axis=0))
    names = [
        (g, m)
        for g in GROUPS
        for m in ["annual_log_active", "cagr_difference", "sharpe_difference"]
    ]
    names += [
        (g, "annual_log_active_contrast")
        for g in [
            "stock_up_minus_down",
            "bond_up_minus_down",
            "positive_minus_nonpositive_correlation",
        ]
    ]
    family = len(names)
    rows = []
    for j, (group, metric) in enumerate(names):
        finite = draws[:, j][np.isfinite(draws[:, j])]
        if len(finite) < repetitions * 0.95:
            low = high = family_low = family_high = np.nan
        else:
            low, high, family_low, family_high = np.quantile(
                finite, [0.025, 0.975, 0.05 / (2 * family), 1 - 0.05 / (2 * family)]
            )
        rows.append(
            dict(
                group=group,
                metric=metric,
                estimate=float(point[j]),
                lower_95=low,
                upper_95=high,
                family_lower_95=family_low,
                family_upper_95=family_high,
                valid_repetitions=len(finite),
                repetitions=repetitions,
                family_size=family,
                block_days=block_days,
                seed=seed,
            )
        )
    return pd.DataFrame(rows)
