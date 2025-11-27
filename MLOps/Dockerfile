FROM python:3.10-slim-bookworm

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTORCH_NUM_THREADS=1
ENV PYTORCH_NUM_INTEROP_THREADS=1

WORKDIR /MLOps

COPY requirements.txt .

# usa pip stabile e non aggressivo
RUN pip install --no-cache-dir pip==23.2.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# 🔥 RIMUOVI IL CMD QUI SOTTO
# CMD ["uvicorn", "api_ingestion:app", "--host", "0.0.0.0", "--port", "8011"]

# Lascia il comando vuoto: il docker-compose lo sovrascriverà correttamente
CMD ["bash"]
