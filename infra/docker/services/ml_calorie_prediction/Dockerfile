FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ml_features/ml_calorie_estimation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_features /app/ml_features

ENV PYTHONPATH=/app
ENV ML_PROJECT_ROOT=/app/ml_features/ml_calorie_estimation

EXPOSE 8000

CMD ["uvicorn", "ml_features.ml_calorie_estimation.pipeline.predict:app", "--host", "0.0.0.0", "--port", "8000"]