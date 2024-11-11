FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install torch transformers langdetect google-cloud-storage

WORKDIR /app

COPY translation_script.py .

CMD ["python3", "translation_script.py"]


FROM python:3.9-slim

WORKDIR /app

COPY translation_script.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "translation_script.py"]


FROM python:3.9-slim

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
