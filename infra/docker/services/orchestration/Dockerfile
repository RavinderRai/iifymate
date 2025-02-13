FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    httpx==0.25.0

COPY ml_features /app/ml_features

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

EXPOSE 8002

CMD ["uvicorn", "ml_features.orchestration.src.app:app", "--host", "0.0.0.0", "--port", "8002"]