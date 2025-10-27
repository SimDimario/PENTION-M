FROM python:3.10-slim

WORKDIR /ClassificatoreNPS

COPY ./ClassificatoreNPS/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ClassificatoreNPS/ .

EXPOSE 8000

CMD ["uvicorn", "api_clf_nps:app", "--host", "0.0.0.0", "--port", "8000"]
