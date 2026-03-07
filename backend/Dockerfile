FROM python:3.11-slim

ARG REPO=https://github.com/harryfrzz/neo.git
ARG BRANCH=main

# System deps: git (clone), ffmpeg (video/audio), tesseract (OCR), build tools for faiss/whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    tesseract-ocr \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth=1 --branch "$BRANCH" "$REPO" . \
    && pip install --no-cache-dir -r backend/requirements.txt

WORKDIR /app/backend

# Persist vector store, DB, and model registry across restarts
VOLUME ["/app/backend/data"]

EXPOSE 8000

CMD ["fastapi", "run", "ExposeAPI.py", "--host", "0.0.0.0", "--port", "8000"]
