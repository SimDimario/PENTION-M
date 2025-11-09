# Usa una base leggera
FROM python:3.10-slim-bookworm

# Imposta la working directory nel container
WORKDIR /CorrectionDispersion_PIML

# Copia solo requirements per usare la cache di Docker
COPY requirements.txt .

# Installa le dipendenze (questo step resta in cache finché requirements.txt non cambia)
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo il resto del codice dopo (invalida solo questo step, non quello delle librerie)
COPY . .

# Espone la porta API
EXPOSE 8008

# Avvia il server FastAPI
CMD ["uvicorn", "api_correction_piml:app", "--host", "0.0.0.0", "--port", "8008"]
