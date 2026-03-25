FROM python:3.12-slim

WORKDIR /app

# Install football-core shared library
COPY football-core/ /football-core/
RUN pip install --no-cache-dir --progress-bar off /football-core/

# Install training dependencies
COPY training/requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy training scripts
COPY training/train/ train/
COPY training/retrain_scheduler.py .

ENV MODELS_DIR=/app/models

# Default: run retrain scheduler (override per CronJob)
CMD ["python", "retrain_scheduler.py"]
