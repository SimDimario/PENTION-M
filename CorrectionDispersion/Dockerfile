FROM python:3.10-slim-bookworm

WORKDIR /CorrectionDispersion

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api_correction:app", "--host", "0.0.0.0", "--port", "8001"]
