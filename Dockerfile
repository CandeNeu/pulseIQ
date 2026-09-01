FROM python:3.11-slim
COPY setup.py .
COPY pulseiq/ pulseiq
COPY requirements.txt .
RUN pip install .
CMD uvicorn pulseiq.api.fast:app --host 0.0.0.0 --port ${PORT:-8080}
