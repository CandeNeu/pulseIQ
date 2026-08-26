FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the demo model in. For a real model, prefer downloading from GCS at
# startup instead of copying into the image (keeps images small + versioned).
RUN python train_example.py

# Cloud Run provides $PORT (defaults to 8080).
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
