FROM python:3.10-slim-bookworm

WORKDIR /PentionSystem

# 👇 Cambiato il path
COPY ./PentionSystem/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 👇 Cambiato anche questo
COPY ./PentionSystem/ .

COPY ./gaussianPuff /PentionSystem/gaussianPuff

EXPOSE 8004

CMD ["python", "-m", "streamlit", "run", "application.py", "--server.port=8004", "--server.address=0.0.0.0"]
