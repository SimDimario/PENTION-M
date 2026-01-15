FROM python:3.10-slim-bookworm

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTORCH_NUM_THREADS=1
ENV PYTORCH_NUM_INTEROP_THREADS=1

WORKDIR /MLOps

COPY requirements.txt .

RUN pip install --no-cache-dir pip==23.2.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
