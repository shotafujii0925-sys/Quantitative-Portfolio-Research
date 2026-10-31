"""全教材を実kernelで実行し分析結果/実行済みノートとJSONに結果を残す。"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """失敗Notebook名を記録し、成功を偽らず非ゼロ終了する。"""
    for name in ("matplotlib", "jupyter-runtime", "ipython"):
        (ROOT / ".cache" / name).mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache/jupyter-runtime")
    os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    destination = ROOT / "分析結果/実行済みノート"
    destination.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "分析結果/レポート/notebook_execution.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    results_by_name = {row["notebook"]: row for row in previous.get("results", [])}
    paths = sorted((ROOT / "学習ノート").glob("*.ipynb"))
    if sys.argv[1:]:
        unknown = set(sys.argv[1:]) - {path.name for path in paths}
        if unknown:
            raise ValueError(f"Unknown notebooks: {unknown}")
        paths = [path for path in paths if path.name in sys.argv[1:]]
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb, timeout=240, kernel_name="quant-ai", resources={"metadata": {"path": str(ROOT)}}
        )
        print(f"Executing {path.name}", flush=True)
        entry = {
            "notebook": path.name,
            "data": nb.metadata.get("quant_ai", {}).get("data_kind", "synthetic_only"),
            "executed_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            client.execute()
            entry["status"] = "passed"
            entry["executed_code_cells"] = sum(c.cell_type == "code" for c in nb.cells)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        nbformat.write(nb, destination / path.name)
        results_by_name[path.name] = entry
        print(f"{path.name}: {entry['status']}", flush=True)
    results = sorted(results_by_name.values(), key=lambda row: row["notebook"])
    report = {"updated_at_utc": datetime.now(UTC).isoformat(), "results": results}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
