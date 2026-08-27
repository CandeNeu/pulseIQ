FROM python:3.11-slim


COPY setup.py .
COPY pulseiq/ pulseiq
COPY models/ models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run provides $PORT (defaults to 8080).
CMD uvicorn pulseiq.api.fast:app --host 0.0.0.0 --port ${PORT:-8080}
