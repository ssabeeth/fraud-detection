# The scoring API. The model is copied in at build time from docker/model/ (a LightGBM
# bundle) and the frozen policy from reports/policy_frozen.json. CI builds with a model
# trained on the synthetic fixtures, so the public image never contains anything learnt
# from the competition data; see docs/deploy_azure.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra serve

COPY docker/model /app/model
COPY reports/policy_frozen.json /app/policy_frozen.json

ENV PATH=/app/.venv/bin:$PATH \
    FRAUD_MODEL_DIR=/app/model \
    FRAUD_POLICY=/app/policy_frozen.json

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "fraud.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
