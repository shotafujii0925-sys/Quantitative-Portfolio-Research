"""ネットワーク不要のQlib教材。合成OHLCV・特徴量・LGBModel・公式バックテスト。"""

import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

from quant_ai.data_loader import synthetic_prices


def build_sample_provider(root: Path, rows: int = 420, assets: int = 6) -> Path:
    """Qlib日次形式を生成。各binはlittle-endian float32の[開始添字,値...]。

    公式dump_bin.pyと同じ格納仕様の最小実装。初日価格を1に規格化しfactorで復元。
    一定の既知銘柄集合なので、生存者バイアスを評価できない合成データである。
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "SYNTHETIC_ONLY.json"
    if any(root.iterdir()) and not marker.exists():
        raise FileExistsError("Refusing to overwrite an unknown provider directory")
    prices = synthetic_prices(rows=rows, assets=assets)
    dates = prices.index.strftime("%Y-%m-%d")
    (root / "calendars").mkdir(exist_ok=True)
    (root / "instruments").mkdir(exist_ok=True)
    (root / "calendars/day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")
    instruments = []
    for symbol in prices.columns:
        close = prices[symbol].to_numpy()
        factor = np.full(rows, 1 / close[0])
        previous = np.r_[close[0], close[:-1]]
        fields = {
            "close": close * factor,
            "open": previous * factor,
            "high": np.maximum(close, previous) * 1.003 * factor,
            "low": np.minimum(close, previous) * 0.997 * factor,
            "volume": np.full(rows, 1e6) / factor,
            "factor": factor,
        }
        directory = root / "features" / symbol.lower()
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in fields.items():
            np.r_[0, values].astype("<f4").tofile(directory / f"{name}.day.bin")
        instruments.append(f"{symbol}\t{dates[0]}\t{dates[-1]}")
    (root / "instruments/all.txt").write_text("\n".join(instruments) + "\n", encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "kind": "synthetic",
                "seed": 42,
                "rows": rows,
                "assets": assets,
                "source": "quant_ai.synthetic_prices",
                "not_empirical": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def run_qlib_demo(provider: Path, output_dir: Path | None = None) -> dict:
    """学習/検証/テストを時系列分離し、Qlib LGBModelとTopkDropoutを実行。

    X_tはRef(close,1),Meanなど過去のみ。y_t=close[t+2]/close[t+1]-1。
    2営業日purgeを置く。testスコアは翌終値で執行、以降の運用収益を評価。
    """
    import qlib
    from qlib.constant import REG_US
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    provider = provider.resolve()
    qlib.init(
        provider_uri=str(provider),
        region=REG_US,
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        logging_level=30,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": unquote((provider.parent / "mlruns").as_uri()),
                "default_exp_name": "synthetic_only",
            },
        },
    )
    calendar = pd.DatetimeIndex(D.calendar())
    instruments = D.list_instruments(D.instruments("all"), as_list=True)
    feature_expr = [
        "$close/Ref($close,1)-1",
        "$close/Ref($close,5)-1",
        "$close/Mean($close,20)-1",
        "Std($close/Ref($close,1)-1,20)",
    ]
    feature_names = ["return_1", "momentum_5", "ma20_ratio", "vol20"]
    sample = D.features(
        instruments,
        feature_expr,
        start_time=str(calendar[25].date()),
        end_time=str(calendar[30].date()),
    )
    a, b = int(len(calendar) * 0.6), int(len(calendar) * 0.8)
    segments = {
        "train": (calendar[25], calendar[a - 3]),
        "valid": (calendar[a], calendar[b - 3]),
        "test": (calendar[b], calendar[-4]),
    }
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=calendar[0],
        end_time=calendar[-1],
        data_loader={
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (feature_expr, feature_names),
                    "label": (["Ref($close,-2)/Ref($close,-1)-1"], ["LABEL0"]),
                }
            },
        },
        infer_processors=[{"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}}],
        learn_processors=[{"class": "DropnaLabel"}],
    )
    dataset = DatasetH(handler=handler, segments=segments)
    model = LGBModel(
        loss="mse",
        num_boost_round=40,
        early_stopping_rounds=5,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        num_threads=1,
        seed=42,
        deterministic=True,
        verbosity=-1,
    )
    model.fit(dataset)
    predictions = model.predict(dataset, segment="test")
    strategy = TopkDropoutStrategy(
        signal=predictions, topk=2, n_drop=1, hold_thresh=1, only_tradable=True
    )
    # Qlibは当日の取引に前営業日のscoreを参照。初回scoreの翌日から開始。
    report, positions = backtest_daily(
        start_time=calendar[b + 1],
        end_time=calendar[-2],
        strategy=strategy,
        account=100000,
        benchmark=instruments[0],
        exchange_kwargs={
            "freq": "day",
            "deal_price": "close",
            "limit_threshold": None,
            "trade_unit": 1,
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 0,
        },
    )
    net = report["return"] - report["cost"]
    evaluation = risk_analysis(net, freq="day")
    excess = risk_analysis(net - report["bench"], freq="day")
    results = {
        "kind": "synthetic_only",
        "qlib_version": qlib.__version__,
        "features": sample,
        "predictions": predictions,
        "report": report,
        "risk": evaluation,
        "excess_risk": excess,
        "positions_count": len(positions),
        "segments": {k: [str(v[0].date()), str(v[1].date())] for k, v in segments.items()},
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output_dir / "qlib_synthetic_features.csv")
        predictions.to_csv(output_dir / "qlib_synthetic_predictions.csv")
        report.to_csv(output_dir / "qlib_synthetic_backtest.csv")
        evaluation.to_csv(output_dir / "qlib_synthetic_risk.csv")
        (output_dir / "qlib_summary.json").write_text(
            json.dumps(
                {
                    "kind": results["kind"],
                    "qlib_version": results["qlib_version"],
                    "prediction_rows": len(predictions),
                    "backtest_rows": len(report),
                    "segments": results["segments"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return results
