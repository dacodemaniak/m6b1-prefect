from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Chemins par défaut
    BASE_PATH: Path = Path(__file__).resolve().parent.parent
    
    # Ports et URLs (chargés depuis .env)
    API_PORT: int = 8000
    IHM_PORT: int = 8501
    API_URL: str = "http://api:8000" # Nom du service Docker par défaut
    
    # Chemins des données
    MODEL_PATH: str = "models/model_mnist.keras"
    DB_PATH: str = "prefect-data/mnist_data.db"
    
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8"
    )

settings = Settings()