FROM python:3.10-slim-bookworm

WORKDIR /MLOps

COPY requirements.txt .

# usa pip stabile e non aggressivo
RUN pip install --no-cache-dir pip==23.2.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8011

CMD ["uvicorn", "api_ingestion:app", "--host", "0.0.0.0", "--port", "8011"]
