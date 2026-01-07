from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import sqlite3
import json
import os
from pathlib import Path
from loguru import logger
import pandas as pd


app = FastAPI()

# Configuration du modèle
BASE_PATH = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_PATH / 'models' / 'model_mnist.keras'
model = tf.keras.models.load_model(str(MODEL_PATH))

# Configure logging
LOG_PATH = BASE_PATH / 'logs' / 'api.log'
logger.add(str(LOG_PATH), rotation="1 MB", retention="1 days", level="DEBUG")


DATABASE_PATH = BASE_PATH / 'prefect-data' / 'mnist_data.db'
class PredictionInput(BaseModel):
    image: list # Liste de listes (28x28) ou liste aplatie (784)

class ImprovementInput(BaseModel):
    image: list
    predicted_digit: int
    actual_digit: int

def reload_model():
    """Charge la dernière version du modèle en mémoire."""
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info(f"Modèle chargé avec succès depuis {MODEL_PATH}")

@app.post("/predict")
async def predict(data: PredictionInput):
    logger.debug("Requête reçue sur /predict")
    
    if model is None:
        logger.error("Modèle non disponible")
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur")
    
    try:
        # Prétraitement
        img = np.array(data.image).astype('float32')
        logger.debug(f"Shape reçue : {img.shape}")

        img = np.array(data.image).reshape(1, 28, 28, 1) / 255.0
        
        # Inférence
        preds = model.predict(img)
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        result = {"digit": digit, "confidence": confidence}
        logger.info(f"Prédiction réussie : {digit} ({confidence:.2f})")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/improvement")
async def improvement(data: ImprovementInput):
    # Stockage en base
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO corrections (image_json, predicted_digit, actual_digit) VALUES (?, ?, ?)",
        (json.dumps(data.image), data.predicted_digit, data.actual_digit)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Feedback enregistré"}

@app.post("/reload")
async def trigger_reload():
    """Endpoint appelé par Prefect après un réentraînement."""
    reload_model()
    return {"status": "model reloaded"}

@app.get("/metrics")
async def get_metrics():
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        # Utilisation de pandas pour simplifier l'agrégation côté API
        df_corr = pd.read_sql("SELECT actual_digit, is_trained FROM corrections", conn)
        
        # Récupération de l'historique d'entraînement (si la table existe)
        try:
            df_hist = pd.read_sql("SELECT accuracy, trained_at FROM training_history ORDER BY trained_at ASC", conn)
            history = df_hist.to_dict(orient="records")
        except Exception:
            history = []
            
        conn.close()

        # Construction de la réponse agrégée
        return {
            "total_corrections": len(df_corr),
            "new_images_count": int(len(df_corr[df_corr['is_trained'] == 0])),
            "distribution": df_corr['actual_digit'].value_counts().to_dict(),
            "training_history": history
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques : {e}")
        raise HTTPException(status_code=500, detail=str(e))