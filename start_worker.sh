#!/bin/bash

# 1. Attendre que le serveur Prefect soit prêt
echo "Vérification de la connexion au serveur Prefect..."
until curl -s "$PREFECT_API_URL/health" > /dev/null; do
  echo "Serveur Prefect indisponible, attente..."
  sleep 5
done

echo "Serveur Prefect disponible."

# Appliquer le déploiement basé sur le fichier YAML
echo "Application du déploiement YAML..."
prefect deployment apply mnist_pipeline-deployment.yaml

# 3. Lancer le worker pour écouter les jobs
echo "Lancement du Worker..."
prefect worker start --pool "default-agent-pool"