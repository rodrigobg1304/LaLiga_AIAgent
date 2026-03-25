FROM python:3.12-slim

WORKDIR /app

# Install football-core shared library
COPY football-core/ /football-core/
RUN pip install --no-cache-dir --progress-bar off /football-core/

# Install service dependencies
COPY services/streamlit/requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy Streamlit app
COPY services/streamlit/streamlit_app.py .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
