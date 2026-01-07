import sqlite3
import json
import numpy as np
import tensorflow as tf
import optuna
import requests
from pathlib import Path
from prefect import flow, task
from loguru import logger

# --- Configuration des chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "prefect-data" / "mnist_data.db"
MODEL_PATH = BASE_DIR / "models" / "model_mnist.keras"
API_URL = "http://localhost:8000"

# Configuration du log
logger.add(BASE_DIR / "logs" / "prefect.log", rotation="1 MB", level="INFO")

# --- Tâches ---

@task(retries=2)
def fetch_data_from_db():
    """Récupère les nouvelles corrections et les données historiques."""
    logger.info("Extraction des données depuis SQLite...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 1. Nouvelles données
    cursor.execute("SELECT id, image_json, actual_digit FROM corrections WHERE is_trained = 0")
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return None, None, []

    ids = [r[0] for r in rows]
    X_new = np.array([json.loads(r[1]) for r in rows]).reshape(-1, 28, 28, 1) / 255.0
    y_new = np.array([r[2] for r in rows])
    
    # 2. Replay Buffer (MNIST original) pour éviter l'oubli catastrophique
    (X_orig, y_orig), _ = tf.keras.datasets.mnist.load_data()
    X_orig = X_orig.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    # On mélange 2000 images originales avec les nouvelles
    idx = np.random.choice(len(X_orig), 2000, replace=False)
    X_final = np.concatenate([X_orig[idx], X_new])
    y_final = np.concatenate([y_orig[idx], y_new])
    
    conn.close()
    return X_final, y_final, ids

@task
def optimize_hyperparameters(X, y):
    """Recherche la meilleure architecture avec Optuna."""
    def objective(trial):
        n_filters = trial.suggest_int("n_filters", 32, 64)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_filters, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Entraînement rapide pour évaluation
        model.fit(X, y, epochs=3, batch_size=32, verbose=0, validation_split=0.2)
        return model.evaluate(X, y, verbose=0)[1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    logger.info(f"Meilleurs paramètres trouvés : {study.best_params}")
    return study.best_params, study.best_value

@task
def final_train_and_save(X, y, best_params, ids, accuracy):
    """Entraîne le modèle final, le sauvegarde et met à jour les stats."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(best_params['n_filters'], (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X, y, epochs=5, verbose=0)
    model.save(str(MODEL_PATH))
    
    # Mise à jour de la DB
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # 1. Marquer comme entraîné
    cursor.executemany("UPDATE corrections SET is_trained = 1 WHERE id = ?", [(i,) for i in ids])
    # 2. Enregistrer l'historique
    cursor.execute(
        "INSERT INTO training_history (accuracy, best_params, data_count) VALUES (?, ?, ?)",
        (float(accuracy), json.dumps(best_params), len(ids))
    )
    conn.commit()
    conn.close()
    
    # Notifier l'API
    requests.post(f"{API_URL}/reload")
    logger.success("Modèle mis à jour et API notifiée.")

# --- Flow ---

@flow(name="MNIST_Continuous_Learning")
def mnist_pipeline():
    X, y, ids = fetch_data_from_db()
    
    if X is not None and len(ids) >= 5: # Seuil de déclenchement (ex: 5 nouvelles images)
        best_params, accuracy = optimize_hyperparameters(X, y)
        final_train_and_save(X, y, best_params, ids, accuracy)
    else:
        logger.info("Pas assez de nouvelles données pour un réentraînement.")

if __name__ == "__main__":
    mnist_pipeline()