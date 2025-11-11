FROM python:3.10-slim-bookworm

WORKDIR /EmissionSourceLocalization_PIML

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010

CMD ["uvicorn", "api_source_localization_piml:app", "--host", "0.0.0.0", "--port", "8010"]
