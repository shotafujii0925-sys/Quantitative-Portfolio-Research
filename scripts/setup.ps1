param([switch]$WithTorch)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.python'
uv python install 3.11.14
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
if ($WithTorch) { uv sync --locked --group dev --extra torch }
else { uv sync --locked --group dev }
if ($LASTEXITCODE -ne 0) { throw 'Dependency synchronization failed' }
# プロジェクト内kernelspecのみ。ユーザー全体のJupyter設定を変更しない。
& .\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name quant-ai --display-name 'Python 3.11 (quant-ai)'
if ($LASTEXITCODE -ne 0) { throw 'Kernel installation failed' }
