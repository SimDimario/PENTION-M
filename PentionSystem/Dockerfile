FROM python:3.10-slim-bookworm

WORKDIR /PentionSystem

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ Copia modulo esterno
COPY ../gaussianPuff /PentionSystem/gaussianPuff

EXPOSE 8004

CMD ["python", "-m", "streamlit", "run", "application.py", "--server.port=8004", "--server.address=0.0.0.0"]
