FROM python:3.10-slim-bookworm

# --- Install system dependencies for OSMNX ---
RUN apt-get update && apt-get install -y \
    libspatialindex-dev \
    gcc \
    g++ \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /PentionSystem_M

COPY requirements.txt .

RUN pip install --no-cache-dir pip==23.2.1 && \
    pip install --no-cache-dir osmnx==1.9.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005

CMD ["uvicorn", "ui_pention_m:app", "--host", "0.0.0.0", "--port", "8005", "--app-dir", "/PentionSystem_M"]
