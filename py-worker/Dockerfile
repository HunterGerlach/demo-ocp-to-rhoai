FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf-cache

WORKDIR /app
COPY requirements.txt .

# Install all dependencies (CUDA-free)
RUN pip install --no-cache-dir pika psycopg2-binary huggingface_hub scikit-learn numpy joblib pillow librosa soundfile

COPY app.py .
EXPOSE 8000
CMD ["python", "app.py"]
