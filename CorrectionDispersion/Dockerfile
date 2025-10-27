FROM python:3.10-slim

WORKDIR /CorrectionDispersion

COPY ./CorrectionDispersion/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./CorrectionDispersion/ .

EXPOSE 8001

CMD ["uvicorn", "api_correction:app", "--host", "0.0.0.0", "--port", "8001"]
