FROM python:3.10-slim

WORKDIR /EmissionSourceLocalization

COPY ./EmissionSourceLocalization/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./EmissionSourceLocalization/ .

EXPOSE 8003

CMD ["uvicorn", "api_source_localization:app", "--host", "0.0.0.0", "--port", "8003"]
