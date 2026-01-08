# tests/conftest.py
import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))