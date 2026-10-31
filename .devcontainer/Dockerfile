FROM ghcr.io/astral-sh/uv:0.10.10 AS uv
FROM python:3.11.14-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 vscode
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}" \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
WORKDIR /workspaces/quant-ai
COPY . .
RUN mkdir -p 分析結果/レポート 分析結果/図 data/raw data/processed \
    && uv sync --locked --group dev \
    && python -m ipykernel install --prefix /opt/venv --name quant-ai --display-name 'Python 3.11 (quant-ai)' \
    && python -m pytest -q \
    && ruff check src tests scripts \
    && chown -R vscode:vscode /opt/venv /workspaces/quant-ai
USER vscode
CMD ["bash"]
