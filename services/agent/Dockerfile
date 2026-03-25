FROM python:3.12-slim

WORKDIR /app

# Install football-core shared library
COPY football-core/ /football-core/
RUN pip install --no-cache-dir --progress-bar off /football-core/

# Install service dependencies
COPY services/agent/requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy agent service code (maintains src/ layout for sys.path.append)
COPY services/agent/agent_api.py .
COPY services/agent/src/ src/
COPY services/agent/knowledge/ knowledge/

EXPOSE 8002
CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8002"]
