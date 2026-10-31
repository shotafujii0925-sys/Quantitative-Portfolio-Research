"""公開インデックス・全履歴・Notebook・リンクを検査。検出値自体は表示しない。"""

import argparse
import json
import re
import struct
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUTS = {
    "outputs/reports/リスク統一比較_分析レポート.md",
    "outputs/reports/戦略比較と収益分解_分析レポート.md",
    "outputs/reports/市場環境に依存しない優位性の検証.md",
    "outputs/figures/市場環境別VT直接比較/01_VT同士の直接比較.png",
    "outputs/figures/市場環境別VT直接比較/02_ローリング期間の優劣.png",
    "outputs/figures/市場環境別VT直接比較/03_市場局面と暦期間の比較.png",
    "outputs/figures/市場環境別VT直接比較/04_ペアブロック再標本化の区間.png",
    "outputs/figures/戦略比較と収益分解/02_非レバとVTの成績比較.png",
    "outputs/figures/戦略比較と収益分解/03_CAGR差の要因分解.png",
}
PATTERNS = {
    "credential token": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        r"|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "private home path": re.compile(
        r'(?i)(?:[a-z]:[\\/]+[U]sers[\\/]+|/[U]sers/|/[h]ome/)[^\s"\x27]+'
    ),
    "email": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "Japanese mobile": re.compile(r"(?<!\d)0[789]0[- ]\d{4}[- ]\d{4}(?!\d)"),
}


def git_bytes(*args: str) -> bytes:
    """引数配列でGitを呼び、シェル展開せず内容を読む。"""
    return subprocess.check_output(["git", *args], cwd=ROOT)


def allowed(path: str) -> bool:
    """公開許可の範囲。元データと未審査の生成出力は許可しない。"""
    p = Path(path)
    if p.name == ".gitkeep":
        return True
    if path.startswith("outputs/"):
        return path in PUBLIC_OUTPUTS
    if path.startswith(("data/", "分析結果/", ".venv", ".cache/", ".python/", ".uv-cache/")):
        return False
    if p.suffix.lower() in {
        ".log",
        ".xml",
        ".csv",
        ".parquet",
        ".pkl",
        ".pickle",
        ".bin",
        ".zip",
        ".db",
    }:
        return False
    if path.startswith(("src/", "tests/", "scripts/")):
        return p.suffix in {".py", ".ps1"}
    if path.startswith("学習ノート/"):
        return p.suffix == ".ipynb"
    if path.startswith("資料/"):
        return p.suffix == ".md"
    if path.startswith(".github/workflows/"):
        return p.suffix in {".yml", ".yaml"}
    if path.startswith((".vscode/", ".devcontainer/")):
        return p.suffix == ".json" or p.name == "Dockerfile"
    return "/" not in path and (
        p.suffix in {".md", ".toml", ".lock"}
        or p.name
        in {
            ".gitignore",
            ".gitattributes",
            ".dockerignore",
            ".python-version",
            "LICENSE",
        }
    )


def scan_text(text: str) -> list[str]:
    """秘密らしいパターン・個人ホームを検出し、値は返さない。完全な漏洩検出器ではない。"""
    issues = []
    for kind, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if kind == "email":
            matches = [x for x in matches if not x.endswith("@users.noreply.github.com")]
        if matches:
            issues.append(kind)
    return issues


def scan_blob(path: str, content: bytes) -> list[str]:
    """ファイル許可・サイズ・テキスト・Notebook・PNGメタデータを検査する。"""
    issues = []
    if not allowed(path):
        issues.append("not in publication scope")
    if len(content) > 5 * 1024 * 1024:
        issues.append("file exceeds 5 MiB")
    if path.endswith(".png"):
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return issues + ["invalid PNG"]
        offset = 8
        while offset + 12 <= len(content):
            length = struct.unpack(">I", content[offset : offset + 4])[0]
            kind = content[offset + 4 : offset + 8]
            data = content[offset + 8 : offset + 8 + length]
            if kind in {b"tEXt", b"iTXt"}:
                issues.extend(scan_text(data.decode("utf-8", errors="replace")))
            elif kind == b"zTXt":
                issues.append("compressed PNG metadata requires manual review")
            offset += length + 12
        return issues
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return issues + ["unexpected binary content"]
    issues.extend(scan_text(text))
    if path.endswith(".ipynb"):
        notebook = json.loads(text)
        for cell in notebook.get("cells", []):
            if cell.get("outputs") or cell.get("execution_count") is not None:
                issues.append("notebook output/execution count")
            if cell.get("attachments"):
                issues.append("notebook attachments require review")
            if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
                issues.append("hidden notebook code")
    return sorted(set(issues))


def main() -> None:
    """ステージ済み版を監査。--historyはブランチ・タグの全到達履歴も検査する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    # 親の別リポジトリを誤って監査/公開しない。
    top = Path(git_bytes("rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != ROOT:
        raise RuntimeError("A dedicated project repository is required")
    paths = git_bytes("ls-files", "-z").decode("utf-8").rstrip("\0").split("\0")
    if not paths or paths == [""]:
        raise RuntimeError("No indexed files to audit")
    indexed = set(paths)
    problems = []
    size = 0
    for path in paths:
        content = git_bytes("show", f":{path}")
        size += len(content)
        for issue in scan_blob(path, content):
            problems.append({"path": path, "issue": issue})
        if path.endswith((".md", ".ipynb")):
            text = content.decode("utf-8-sig")
            if path.endswith(".ipynb"):
                text = "\n".join(
                    "".join(c.get("source", []))
                    for c in json.loads(text)["cells"]
                    if c["cell_type"] == "markdown"
                )
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                link = unquote(link.strip("<>").split("#")[0])
                if not link or re.match(r"https?://|mailto:", link):
                    continue
                target = (ROOT / Path(path).parent / link).resolve()
                if (
                    not target.is_relative_to(ROOT)
                    or target.relative_to(ROOT).as_posix() not in indexed
                ):
                    problems.append({"path": path, "issue": f"unpublished local link: {link}"})
    objects = 0
    if args.history:
        for line in git_bytes("rev-list", "--objects", "--all").decode("utf-8").splitlines():
            fields = line.split(" ", 1)
            oid = fields[0]
            kind = git_bytes("cat-file", "-t", oid).strip()
            if kind == b"blob":
                if len(fields) != 2:
                    raise RuntimeError("Unnamed history blob requires manual review")
                objects += 1
                for issue in scan_blob(fields[1], git_bytes("cat-file", "blob", oid)):
                    problems.append({"path": fields[1], "issue": f"history: {issue}"})
            elif kind == b"commit":
                for issue in scan_text(git_bytes("cat-file", "commit", oid).decode("utf-8")):
                    problems.append({"path": "commit metadata", "issue": issue})
    result = {
        "status": "failed" if problems else "passed",
        "indexed_files": len(paths),
        "indexed_bytes": size,
        "history_blobs_checked": objects,
        "issues": problems,
        "scope": "Heuristic content, scope and link audit; not a full DLP or legal guarantee",
    }
    report = ROOT / "outputs/reports/公開前監査.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
