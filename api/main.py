from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import sqlite3
import json
import os
from pathlib import Path
from loguru import logger
import pandas as pd
from api.config import settings

# --- Modèles Pydantic ---
class PredictionInput(BaseModel):
    image: list 

class ImprovementInput(BaseModel):
    image: list
    predicted_digit: int
    actual_digit: int

# --- Logique de chargement ---
def load_mnist_model(model_path: Path):
    """Utilitaire pour charger le modèle avec gestion d'erreur."""
    if not model_path.exists():
        logger.error(f"Fichier modèle introuvable : {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(str(model_path))
        logger.info(f"Modèle chargé avec succès.")
        return model
    except Exception as e:
        logger.error(f"Erreur technique lors du chargement du modèle : {e}")
        return None

app = FastAPI()

# Initialisation préventive des attributs de state
app.state.model = None
app.state.model_path = settings.BASE_PATH / settings.MODEL_PATH
app.state.db_path = settings.BASE_PATH / settings.DB_PATH



# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialisation des constantes de chemin dans app.state
    if not hasattr(app.state, "model_path"):
        app.state.model_path = settings.BASE_PATH / settings.MODEL_PATH

    if not hasattr(app.state, "db_path"):
        app.state.db_path = settings.BASE_PATH / settings.DB_PATH
    
    # 2. Configuration du logging
    log_file = settings.BASE_PATH / 'logs' / 'api.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="1 MB", retention="1 days", level="DEBUG")
    
    logger.info(f"Démarrage de l'API sur le port {settings.API_PORT}")
    logger.info(f"Base de données sqlite : {app.state.db_path}")

    # 3. Chargement initial du modèle
    app.state.model = load_mnist_model(app.state.model_path)
    
    yield  # L'API répond aux requêtes ici
    
    # 4. Nettoyage à la fermeture
    logger.info("Arrêt de l'API...")

app.router.lifespan_context = lifespan

def get_db_path(request: Request):
    """Dépendance pour récupérer le chemin de la base de données."""
    return request.app.state.db_path

# --- Endpoints ---

@app.post("/predict")
async def predict(request: Request, data: PredictionInput):
    # Récupération du modèle depuis le state de l'app via request
    model = request.app.state.model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible sur le serveur")
    
    try:
        img = np.array(data.image).reshape(1, 28, 28, 1).astype('float32') / 255.0
        preds = model.predict(img)
        
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        logger.info(f"Prédiction : {digit} ({confidence:.2f})")
        return {"digit": digit, "confidence": confidence}
    except Exception as e:
        logger.error(f"Erreur inference : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de l'image")

@app.post("/improvement")
async def improvement(request: Request, data: ImprovementInput):
    db_path = get_db_path(request=request)

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO corrections (image_json, predicted_digit, actual_digit) VALUES (?, ?, ?)",
            (json.dumps(data.image), data.predicted_digit, data.actual_digit)
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Feedback enregistré"}
    except Exception as e:
        logger.error(f"Erreur DB : {e}")
        raise HTTPException(status_code=500, detail="Impossible d'enregistrer la correction")

@app.post("/reload")
async def trigger_reload(request: Request):
    """Endpoint appelé par Prefect pour forcer le rechargement du modèle."""
    new_model = load_mnist_model(request.app.state.model_path)
    if new_model:
        request.app.state.model = new_model
        return {"status": "success", "message": "Modèle rechargé en mémoire"}
    raise HTTPException(status_code=500, detail="Échec du rechargement du modèle")

@app.get("/metrics")
async def get_metrics(request: Request):
    db_path = get_db_path(request=request)
    try:
        conn = sqlite3.connect(str(db_path))
        df_corr = pd.read_sql("SELECT actual_digit, is_trained FROM corrections", conn)
        
        try:
            df_hist = pd.read_sql("SELECT accuracy, trained_at FROM training_history ORDER BY trained_at ASC", conn)
            history = df_hist.to_dict(orient="records")
        except:
            history = []
            
        conn.close()

        return {
            "total_corrections": len(df_corr),
            "new_images_count": int(len(df_corr[df_corr['is_trained'] == 0]) if not df_corr.empty else 0),
            "distribution": df_corr['actual_digit'].value_counts().to_dict() if not df_corr.empty else {},
            "training_history": history
        }
    except Exception as e:
        logger.error(f"Erreur metrics : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la lecture des statistiques")