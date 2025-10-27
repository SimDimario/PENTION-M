FROM python:3.10-slim-bookworm

WORKDIR /gaussianPuff

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .

EXPOSE 8002

CMD ["uvicorn", "api_gaussian:app", "--host", "0.0.0.0", "--port", "8002"]
