FROM python:3.10-slim-bookworm

# Directory di lavoro
WORKDIR /MLOps

# Copia solo requirements e installa dipendenze
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Copia il resto del codice
COPY . .

# Imposta la porta predefinita (può essere sovrascritta nel docker-compose)
EXPOSE 8011

ENV PYTHONPATH="${PYTHONPATH}:/MLOps:/gaussianPuff:/CorrectionDispersion_PIML"

# Comando di default: ingestion service
CMD ["uvicorn", "api_ingestion:app", "--host", "0.0.0.0", "--port", "8011"]
