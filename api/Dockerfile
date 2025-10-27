# Builder stage
FROM python:3.11-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Install to a specific directory that we can copy later
# cf. https://github.com/astral-sh/uv/issues/8085#issuecomment-2438256688
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

RUN apk add --no-cache gcc python3-dev musl-dev linux-headers
# Install the project's dependencies using the lockfile and settings
# https://docs.astral.sh/uv/guides/integration/docker/#using-uv-temporarily
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.5,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable

# Runtime stage
FROM python:3.11-alpine AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
# Set the path to use our installed packages
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHON_PATH="/opt/venv/lib/python3.11/site-packages"

LABEL maintainer="F-G Fernandez <fg-feedback@protonmail.com>"

# Install only curl for healthcheck
RUN apk add --no-cache curl

# Copy installed dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Copy project code
COPY pyproject.toml /app/pyproject.toml
COPY app /app/app

# Entrypoint
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD ["curl", "-f", "http://localhost:8080/health", "--max-time", "3"]
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8080", "--proxy-headers", "--use-colors", "--log-level=info"]
