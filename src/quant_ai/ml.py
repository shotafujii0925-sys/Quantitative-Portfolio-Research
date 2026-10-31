"""過去特徴量、ラベル時点を考慮した分割、学習期間だけでの標準化。"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_ai.returns import simple_returns


def make_features(prices: pd.Series, horizon: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    """終値tまでの特徴量X_t。y_t=P_{t+h}/P_{t+1}-1 (h>=2)。

    翌終値執行から先の収益を教師にする。h=2が1期間保有。
    未来値はyにのみ使用。末尾の未知ラベルは除外する。
    """
    if horizon < 2:
        raise ValueError("horizon>=2 for next-close execution")
    returns = simple_returns(prices)
    features = pd.DataFrame(
        {
            "return_1": returns,
            "momentum_5": prices.pct_change(5, fill_method=None),
            "momentum_20": prices.pct_change(20, fill_method=None),
            "volatility_20": returns.rolling(20).std(),
            "price_to_ma20": prices / prices.rolling(20).mean() - 1,
        }
    )
    label = (prices.shift(-horizon) / prices.shift(-1) - 1).rename("target")
    valid = features.notna().all(axis=1) & label.notna()
    return features.loc[valid], label.loc[valid]


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.6,
    valid_fraction: float = 0.2,
    gap: int = 2,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """先頭train→gap→validation→gap→test。gap>=教師の最大先読み日数。

    境界のgap行を除外し、訓練ラベルが次区間開始前に確定するようにする。
    ランダム分割しない。入力は連続営業日の特徴量を想定。
    """
    if not x.index.equals(y.index) or not x.index.is_monotonic_increasing:
        raise ValueError("Aligned chronological data required")
    if not x.index.is_unique or gap < 1:
        raise ValueError("Unique dates and gap>=1 required")
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 - train_fraction:
        raise ValueError("Invalid fractions")
    a, b = int(len(x) * train_fraction), int(len(x) * (train_fraction + valid_fraction))
    slices = {"train": slice(0, a - gap), "valid": slice(a, b - gap), "test": slice(b, None)}
    result = {name: (x.iloc[s], y.iloc[s]) for name, s in slices.items()}
    if any(len(pair[0]) < 5 for pair in result.values()):
        raise ValueError("Too few observations after purge")
    return result


def compare_models(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """固定ハイパーパラメータで訓練しvalid/test RMSEを報告。

    testで選択・調整しない。標準化は学習期間のfitのみ。全予測を返し運用評価に使う。
    """
    models = {
        "zero": DummyRegressor(strategy="constant", constant=0),
        "historical_mean": DummyRegressor(strategy="mean"),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.0001, max_iter=10000)),
        "random_forest": RandomForestRegressor(
            n_estimators=80, max_depth=3, min_samples_leaf=15, random_state=42, n_jobs=1
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=60,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            min_child_samples=20,
            verbosity=-1,
            random_state=42,
            n_jobs=1,
            deterministic=True,
        ),
    }
    scores, predictions = [], {}
    for name, model in models.items():
        model.fit(*splits["train"])
        row = {"model": name}
        for segment in ("valid", "test"):
            x, y = splits[segment]
            pred = pd.Series(model.predict(x), index=x.index, name=name)
            row[f"{segment}_rmse"] = float(np.sqrt(mean_squared_error(y, pred)))
            if segment == "test":
                predictions[name] = pred
        scores.append(row)
    return pd.DataFrame(scores).set_index("model"), predictions
