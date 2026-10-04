# Reproducible research/worker image. The same image runs the CLI, a Ray head and Ray workers,
# so research and production execute byte-identical code and dependencies.
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app

# Dependency layer (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ray --extra s3 --no-install-project

COPY src ./src
COPY jobs ./jobs
ARG CODE_VERSION=dev
ENV PAIRLAB_CODE_VERSION=${CODE_VERSION} PAIRLAB_HOME=/workspace
RUN uv sync --frozen --no-dev --extra ray --extra s3
WORKDIR /workspace
ENTRYPOINT ["pairlab"]
CMD ["--help"]
