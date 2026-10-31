"""実環境の起動・import・設定・Docker接続状態を記録。ビルド成功とは区別する。"""

import importlib
import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """機械可読の環境情報を分析結果/レポート/environment.jsonへ保存。"""
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "statsmodels.api": "statsmodels",
        "yfinance": "yfinance",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
        "qlib": "pyqlib",
        "mlflow": "mlflow",
        "jupyterlab": "jupyterlab",
        "ipykernel": "ipykernel",
        "pytest": "pytest",
        "ruff": "ruff",
    }
    imports = {}
    for module, distribution in modules.items():
        importlib.import_module(module)
        imports[distribution] = version(distribution)
    configs = [
        ".vscode/settings.json",
        ".vscode/extensions.json",
        ".devcontainer/devcontainer.json",
    ]
    for config in configs:
        json.loads((ROOT / config).read_text(encoding="utf-8"))
    docker = {"cli": shutil.which("docker"), "build_executed": False}
    try:
        import docker as docker_sdk

        client = docker_sdk.from_env(timeout=5)
        docker.update(engine_status="available", engine_version=client.version())
        client.close()
    except Exception as exc:
        docker.update(engine_status="unavailable", error=repr(exc))
    result = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "imports": imports,
        "json_configs": "passed",
        "docker": docker,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }
    target = ROOT / "分析結果/レポート/environment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
