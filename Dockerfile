# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Installation des dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du projet
COPY . .

# Définir le PYTHONPATH pour que pytest trouve le module 'api'
ENV PYTHONPATH=/app

# Exécution des tests pour valider l'image
RUN pytest tests/

# Droits d'exécution pour le script de démarrage
RUN chmod +x scripts/start_worker.sh

# Création des répertoires pour les volumes
RUN mkdir -p models prefect-data logs

# Le port par défaut (sera surchargé par docker-compose)
EXPOSE 8000