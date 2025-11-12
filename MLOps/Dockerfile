FROM python:3.10-slim-bookworm

# Directory di lavoro
WORKDIR /MLOps

# Copia solo requirements e installa dipendenze
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Imposta la porta predefinita (può essere sovrascritta nel docker-compose)
EXPOSE 8011

# Comando di default: ingestion service
CMD ["uvicorn", "api_ingestion:app", "--host", "0.0.0.0", "--port", "8011"]
