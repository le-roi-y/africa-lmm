FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY configs ./configs

RUN mkdir -p /app/data/uploads /app/data/processed /app/data/evaluation

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
