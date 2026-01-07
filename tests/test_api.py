import pytest
from fastapi.testclient import TestClient
from api.main import app
import numpy as np
import json
import sqlite3
from pathlib import Path

client = TestClient(app)

# Configuration d'une base de données de test temporaire
TEST_DB = Path("prefect-data/test_mnist_data.db")

@pytest.fixture(autocmd=True)
def setup_test_db():
    """Prépare une base propre avant chaque test."""
    conn = sqlite3.connect(str(TEST_DB))
    conn.execute("DROP TABLE IF EXISTS corrections")
    conn.execute('''
        CREATE TABLE corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_json TEXT NOT NULL,
            predicted_digit INTEGER NOT NULL,
            actual_digit INTEGER NOT NULL,
            is_trained BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    # On force l'API à utiliser la base de test (si votre code le permet via settings)
    yield
    if TEST_DB.exists():
        TEST_DB.unlink()

# --- Scénario 1 : Réponse positive de l'API ---
def test_predict_success():
    # Simulation d'une image 28x28 (liste de listes)
    fake_image = np.zeros((28, 28)).tolist()
    response = client.post("/predict", json={"image": fake_image})
    
    assert response.status_code == 200
    data = response.json()
    assert "digit" in data
    assert "confidence" in data
    assert isinstance(data["digit"], int)

# --- Scénario 2 : Réponse négative (Erreur de validation) ---
def test_predict_bad_data():
    # Envoi d'une chaîne de caractères au lieu d'une liste
    response = client.post("/predict", json={"image": "not_an_image"})
    
    # FastAPI renvoie 422 pour une erreur de schéma Pydantic
    assert response.status_code == 422 

# --- Scénario 3 : Envoi d'une correction et stockage ---
def test_improvement_storage():
    fake_image = np.zeros((28, 28)).tolist()
    payload = {
        "image": fake_image,
        "predicted_digit": 3,
        "actual_digit": 5
    }
    
    # Appel de l'API
    response = client.post("/improvement", json=payload)
    assert response.status_code == 200
    
    # Vérification directe dans la base de données
    # Note : Assurez-vous que DATABASE_PATH dans main.py pointe vers TEST_DB ici
    conn = sqlite3.connect(str(TEST_DB))
    cursor = conn.cursor()
    cursor.execute("SELECT actual_digit FROM corrections WHERE predicted_digit = 3")
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row[0] == 5