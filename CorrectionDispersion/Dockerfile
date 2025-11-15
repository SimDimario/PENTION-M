FROM python:3.10-slim-bookworm

WORKDIR /CorrectionDispersion

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libspatialindex-dev \
    libproj-dev proj-bin proj-data \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir pip==23.2.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api_correction:app", "--host", "0.0.0.0", "--port", "8001"]
