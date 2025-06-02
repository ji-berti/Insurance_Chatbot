FROM python:3.10-slim

# Evita prompts de instalación y mejora tiempos de build
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala herramientas necesarias para compilación de dependencias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Añade timeout y salida detallada
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --timeout 100 --no-cache-dir --verbose

COPY . .

EXPOSE 3000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3000"]
