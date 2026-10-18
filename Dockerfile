FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface KAP_ANALYST_ROOT=/app

WORKDIR /app
# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY kap_analyst ./kap_analyst
RUN pip install ".[ml]"

COPY data/manifest.json data/manifest.json
COPY data/eval data/eval

# PDFs and the index are built on first start and kept in the /app/data volume.
VOLUME ["/app/data"]
EXPOSE 8000
CMD ["sh", "-c", "[ -f data/index/meta.json ] || (kap-analyst download && kap-analyst index); kap-analyst serve --host 0.0.0.0 --port 8000"]
