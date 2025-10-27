FROM python:3.10-slim-bookworm

WORKDIR /ClassificatoreNPS

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "api_clf_nps:app", "--host", "0.0.0.0", "--port", "8000"]
