"""CAPMと線形ファクター回帰。説明変数は同期間の収益率であり予測値ではない。"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from quant_ai.returns import validate_frame


def factor_regression(
    asset_excess: pd.Series, factors: pd.DataFrame, hac_lags: int = 5
) -> RegressionResultsWrapper:
    """r_i-r_f=α+Σβ_k f_k+ε。OLS推定、HAC標準誤差を使用する。

    超過収益・ファクターは同じ頻度と時点。暗黙の欠損除去や補完をしない。
    """
    if not asset_excess.index.equals(factors.index):
        raise ValueError("Indexes must match exactly")
    validate_frame(asset_excess)
    validate_frame(factors)
    if hac_lags < 0 or len(factors) <= factors.shape[1] + 2:
        raise ValueError("Insufficient observations or invalid HAC lag")
    design = sm.add_constant(factors, has_constant="add")
    import numpy as np

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Rank-deficient factor design")
    return sm.OLS(asset_excess, design).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def capm_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_daily: float | pd.Series = 0.0,
    hac_lags: int = 5,
) -> RegressionResultsWrapper:
    """CAPM r_i,t-r_f,t=α_i+β_i(r_m,t-r_f,t)+ε_t。αは1観測期間あたり。"""
    if not asset_returns.index.equals(market_returns.index):
        raise ValueError("Asset/market dates must match")
    if isinstance(risk_free_daily, pd.Series) and not asset_returns.index.equals(
        risk_free_daily.index
    ):
        raise ValueError("Risk-free dates must match")
    return factor_regression(
        asset_returns - risk_free_daily,
        (market_returns - risk_free_daily).to_frame("market_excess"),
        hac_lags,
    )
