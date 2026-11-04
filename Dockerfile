FROM python:3.12-slim AS base

# ONNX Runtime needs libgomp when the jailbreak detector loads its models.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ARG FILTER
RUN case "$FILTER" in \
      pii|prompt-injection|jailbreak) ;; \
      *) echo "FILTER must be pii, prompt-injection, or jailbreak" >&2; exit 1 ;; \
    esac

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    UV_CONCURRENT_DOWNLOADS=4 \
    JAILBREAK_MODEL_CACHE=/opt/models

WORKDIR /app
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --extra "$FILTER" --no-editable \
    && mkdir -p /opt/models

# Download and load the pinned ONNX models during the jailbreak image build.
# The runtime image can then run without network access to Hugging Face.
RUN if [ "$FILTER" = "jailbreak" ]; then \
      /opt/venv/bin/python jailbreak.py; \
    fi

FROM base AS runtime
ARG FILTER
ENV FILTER=$FILTER \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    JAILBREAK_MODEL_CACHE=/opt/models \
    JAILBREAK_LOCAL_FILES_ONLY=true

RUN useradd --system --uid 10001 --create-home fragai
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=fragai:fragai /opt/models /opt/models
USER fragai

CMD ["sh", "-c", "exec \"${FILTER}-filter-mcp\""]
