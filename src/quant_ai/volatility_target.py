"""過去の資産共分散による月次ボラティリティ目標。将来の実現ボラは利用しない。"""

import numpy as np
import pandas as pd

from quant_ai.returns import simple_returns
from quant_ai.risk_allocation import Method, monthly_targets

CORE_METHODS: tuple[Method, ...] = ("equal_weight", "inverse_vol", "min_variance")


def volatility_schedule(
    prices: pd.DataFrame,
    method: Method,
    *,
    lookback: int = 252,
    evaluation_start: str = "2006-01-01",
    target_volatility: float | None = 0.10,
    leverage_cap: float | None = None,
    base_targets: pd.DataFrame | None = None,
    base_audit: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """b_tで求めたσhat=sqrt(252*b'Σ_t*b)、L=目標/σhat、w=L*b。

    capは月次約定後の目標Lの上限。月中の価格変化に伴うLの増加は制約しない。
    target_volatility=Noneなら元の100%投資戦略を再現する。
    予測ボラは当日までの固定組成ポートフォリオの標本共分散から推定する。
    将来の実現ボラで再スケールしない。キャッシュ・借入の金利変動リスクは予測から除く。
    """
    if target_volatility is not None and target_volatility <= 0:
        raise ValueError("Target volatility must be positive")
    if leverage_cap is not None and leverage_cap <= 0:
        raise ValueError("Leverage cap must be positive")
    if base_targets is None or base_audit is None:
        base_targets, base_audit = monthly_targets(prices, method, lookback, evaluation_start)
    dates = base_audit.index[base_audit.execution >= pd.Timestamp(evaluation_start)]
    if dates.empty:
        raise ValueError("No executable target dates")
    returns = simple_returns(prices)
    rows, audit_rows = [], []
    for date in dates:
        history = returns.loc[:date].iloc[-lookback:]
        if len(history) != lookback:
            raise ValueError("Insufficient history")
        base = base_targets.loc[date].to_numpy()
        sigma = float(np.sqrt(base @ (history.cov().to_numpy() * 252) @ base))
        if sigma <= 1e-12:
            raise ValueError("Predicted zero volatility: exposure is undefined")
        requested = 1.0 if target_volatility is None else target_volatility / sigma
        exposure = requested if leverage_cap is None else min(requested, leverage_cap)
        rows.append(pd.Series(base * exposure, index=prices.columns, name=date))
        record = base_audit.loc[date].to_dict()
        record.update(
            decision=date,
            unscaled_predicted_volatility=sigma,
            requested_exposure=requested,
            target_exposure=exposure,
            predicted_volatility=sigma * exposure,
            cap_binding=exposure < requested - 1e-12,
            cash_target=1 - exposure,
            borrowing_target=max(exposure - 1, 0),
        )
        audit_rows.append(record)
    targets = pd.DataFrame(rows)
    targets.index.name = "decision"
    return targets, pd.DataFrame(audit_rows).set_index("decision")
