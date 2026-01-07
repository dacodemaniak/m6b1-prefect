import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
import cv2
from loguru import logger
from pathlib import Path
import plotly.express as px
import pandas as pd
from api.config import settings

# Configuration
API_URL = settings.API_URL


LOG_PATH = settings.BASE_PATH / 'logs' / 'app.log'
logger.add(str(LOG_PATH), rotation="1 MB", retention="1 days", level="DEBUG")

st.set_page_config(page_title="MNIST Recog & Feedback", layout="centered")
st.title("🖋️ Reconnaissance de Chiffres")

tab1, tab2 = st.tabs(["🎨 Dessin & Prédiction", "📊 Dashboard Performance"])

with tab1:
    st.write("Dessinez un chiffre ci-dessous (0-9).")

    # 1. Le Canvas de dessin
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Initialisation de l'état pour stocker la prédiction
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    # 2. Logique de Prédiction
    if canvas_result.image_data is not None:
        # Prétraitement de l'image du canvas
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, (28, 28))

        if st.button("🔍 Prédire le chiffre"):
            logger.debug("Envoi de l'image à l'API...")
            # Envoi à l'API FastAPI
            payload = {"image": img_resized.tolist()}
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                logger.debug(f"Réponse API reçue - Statut : {response.status_code}")
                if response.status_code == 200:
                    st.session_state.prediction = response.json()
                    logger.info(f"Résultat décodé : {st.session_state.prediction}")
                else:
                    logger.error(f"Erreur API {response.status_code} : {response.text}")
                    st.error(f"L'API a renvoyé une erreur : {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")

    # 3. Affichage et Correction
    if st.session_state.prediction:
        pred = st.session_state.prediction
        st.success(f"### Résultat : {pred['digit']} (Confiance : {pred['confidence']:.2%})")
        
        st.divider()
        
        st.subheader("🛠️ Un problème ? Corrigez l'IA")
        correct_val = st.selectbox("Quel était le vrai chiffre ?", list(range(10)), index=pred['digit'])
        
        if st.button("🚀 Envoyer l'amélioration"):
            improvement_payload = {
                "image": img_resized.tolist(),
                "predicted_digit": pred['digit'],
                "actual_digit": correct_val
            }
            res = requests.post(f"{API_URL}/improvement", json=improvement_payload)
            if res.status_code == 200:
                st.toast("Merci ! Données enregistrées pour le réentraînement.", icon="✅")
                st.session_state.prediction = None # Reset
with tab2:
    st.header("📊 Dashboard Performance & Données")
    
    try:
        # Récupération des données via l'API
        res = requests.get(f"{API_URL}/metrics")
        if res.status_code == 200:
            metrics = res.json()
            
            # 1. Métriques clés
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Images en attente", metrics["new_images_count"])
            with col2:
                st.metric("Total corrections", metrics["total_corrections"])
            
            # 2. Graphique de distribution
            if metrics["distribution"]:
                st.subheader("Répartition des chiffres corrigés")
                # Conversion du dict en DataFrame pour Plotly
                dist_df = pd.DataFrame([
                    {"Chiffre": k, "Nombre": v} for k, v in metrics["distribution"].items()
                ])
                fig_dist = px.bar(dist_df, x="Chiffre", y="Nombre", 
                                 range_x=[-0.5, 9.5],
                                 color_discrete_sequence=['#ff4b4b'])
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # 3. Historique de précision
            if metrics["training_history"]:
                st.subheader("Évolution de la précision (Optuna)")
                hist_df = pd.DataFrame(metrics["training_history"])
                fig_line = px.line(hist_df, x="trained_at", y="accuracy", markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("En attente du premier cycle de réentraînement Prefect...")
                
        else:
            st.error("Impossible de récupérer les métriques depuis l'API.")
            
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")