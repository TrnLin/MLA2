FROM docker.io/library/node:22-bookworm-slim AS frontend
WORKDIR /build/fe
COPY fe/package.json fe/package-lock.json ./
RUN npm ci --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000
COPY fe/ ./
RUN npm run build

FROM docker.io/library/python:3.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FASHION_PROJECT_ROOT=/app/core FASHION_STATIC_DIR=/app/fe/dist \
    FASHION_UPLOAD_DIR=/app/runtime/uploads OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core/pyproject.toml core/pyproject.toml
COPY be/pyproject.toml be/pyproject.toml
COPY core/src/ core/src/
COPY be/src/ be/src/
RUN python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir ./core ./be
COPY core/ core/
COPY --from=frontend /build/fe/dist/ fe/dist/
RUN useradd --uid 10001 --create-home fashion \
    && mkdir -p /app/runtime/uploads /app/core/tmp \
    && chown -R fashion:fashion /app
USER fashion
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"
CMD ["python", "-m", "uvicorn", "fashion_api.hosted:create_hosted_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "128"]
