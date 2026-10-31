"""翌終値執行・自己資金制約・現金/借入を明示する追加エンジン。旧APIは変更しない。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from quant_ai.returns import validate_frame


@dataclass
class LeveragedResult:
    """日次NAV・損益・売買・約定前後の比率と、初期資金の基準日。"""

    daily: pd.DataFrame
    initial_date: pd.Timestamp


def leveraged_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
    cash_rate: float = 0.0,
    borrow_rate: float = 0.0,
) -> LeveragedResult:
    """月末targets[t]を次の観測日終値で約定。空売りなし、合計比率>1は借入。

    H_i^- = H_i,prev * P_i,t/P_i,prev、C^- = Cprev + I_cash - I_borrow。
    年実効金利をACT/365で付利。約定前V^- = sum(H^-)+C^-。
    z=V^+/V^-を z+c*sum(abs(z*w-H^-/V^-))=1 から求める。
    H^+=w*V^+, C^+=V^+-sum(H^+)。売買コストは各資産の実際の売買金額×c。
    旧保有の価格変化・借入利息を受けてから終値で取引する。金利は固定シナリオ。
    capはtargets生成側の月次目標制約。月中のドリフトや強制決済はここでは制限しない。
    NAV<=0は破綻として停止し、欠損やリターンを都合よく補完しない。
    """
    validate_frame(prices, positive=True)
    validate_frame(targets)
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex is required for ACT/365 accrual")
    if not prices.columns.equals(targets.columns) or not targets.index.isin(prices.index).all():
        raise ValueError("Target dates/assets must match observed prices")
    if not prices.columns.is_unique or (targets < 0).any().any():
        raise ValueError("Unique long-only assets required")
    if not 0 <= cost_bps < 10000 or not -1 < cash_rate <= borrow_rate:
        raise ValueError("Require 0<=cost<10000bps and -1<cash_rate<=borrow_rate")
    if targets.index[0] != prices.index[0]:
        raise ValueError("First price date must be the first decision date")
    c = cost_bps / 10000
    if c * targets.sum(axis=1).max() >= 1:
        raise ValueError("Costs and leverage do not guarantee a unique self-financing solution")
    values = prices.to_numpy(dtype=float)
    growth = values[1:] / values[:-1]
    scheduled = {date: row.to_numpy(dtype=float) for date, row in targets.iterrows()}
    holdings = np.zeros(prices.shape[1])
    cash, nav = 1.0, 1.0
    rows = []
    for i in range(1, len(prices)):
        previous_nav = nav
        held_weights = holdings / previous_nav
        held_cash = cash / previous_nav
        days = (prices.index[i] - prices.index[i - 1]).total_seconds() / 86400
        if days <= 0:
            raise ValueError("Price dates must increase")
        deposit_daily = np.expm1(np.log1p(cash_rate) * days / 365)
        loan_daily = np.expm1(np.log1p(borrow_rate) * days / 365)
        cash_income = max(cash, 0) * deposit_daily
        funding_cost = max(-cash, 0) * loan_daily
        old_holdings = holdings.copy()
        pre_holdings = holdings * growth[i - 1]
        asset_pnl = float((pre_holdings - old_holdings).sum())
        pre_cash = cash + cash_income - funding_cost
        pre_nav = float(pre_holdings.sum() + pre_cash)
        if pre_nav <= 0:
            raise RuntimeError(f"Insolvent before trading at {prices.index[i]}")
        pre_weights = pre_holdings / pre_nav
        decision = prices.index[i - 1]
        target = scheduled.get(decision)
        trade_cost, trade_amount = 0.0, 0.0
        if target is not None:
            z = brentq(
                lambda z, w=target, pre=pre_weights: z + c * np.abs(z * w - pre).sum() - 1,
                0.0,
                1.0,
                xtol=1e-14,
            )
            nav = pre_nav * z
            holdings = target * nav
            trade_amount = float(np.abs(holdings - pre_holdings).sum())
            trade_cost = c * trade_amount
            cash = nav - float(holdings.sum())
        else:
            holdings, cash, nav = pre_holdings, pre_cash, pre_nav
        expected = previous_nav + asset_pnl + cash_income - funding_cost - trade_cost
        error = nav - expected
        if nav <= 0 or abs(error) > 1e-10 * max(1, abs(nav)):
            raise RuntimeError("Self-financing NAV identity failed")
        row = {
            "net_return": nav / previous_nav - 1,
            "gross_return": pre_nav / previous_nav - 1,
            "asset_return": asset_pnl / previous_nav,
            "cash_income": cash_income,
            "funding_cost": funding_cost,
            "trade_cost": trade_cost,
            "trade_amount": trade_amount,
            "turnover": trade_amount / pre_nav,
            "trade_cost_return": trade_cost / previous_nav,
            "funding_cost_return": funding_cost / previous_nav,
            "cash_income_return": cash_income / previous_nav,
            "cash_opportunity_return": deposit_daily,
            "wealth": nav,
            "pre_nav": pre_nav,
            "held_leverage": float(held_weights.sum()),
            "pre_leverage": float(pre_weights.sum()),
            "post_leverage": float(holdings.sum() / nav),
            "held_cash_weight": held_cash,
            "post_cash_weight": cash / nav,
            "borrowed": max(-cash, 0),
            "rebalance": target is not None,
            "accounting_error": error,
        }
        for j, name in enumerate(prices.columns):
            row[f"held_{name}"] = held_weights[j]
            row[f"pre_{name}"] = pre_weights[j]
            row[f"post_{name}"] = holdings[j] / nav
        rows.append(row)
    return LeveragedResult(pd.DataFrame(rows, index=prices.index[1:]), prices.index[0])
