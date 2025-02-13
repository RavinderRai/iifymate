FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY ml_features/llm_calorie_estimation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_features /app/ml_features

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "ml_features.llm_calorie_estimation.src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]