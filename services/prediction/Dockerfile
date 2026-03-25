FROM python:3.12-slim

WORKDIR /app

# Install football-core shared library
COPY football-core/ /football-core/
RUN pip install --no-cache-dir --progress-bar off /football-core/

# Install service dependencies
COPY services/prediction/requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy prediction service code
COPY services/prediction/predict.py .

ENV MODELS_DIR=/app/models

EXPOSE 8001
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8001"]
