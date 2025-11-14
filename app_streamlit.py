import streamlit as st
import joblib
import numpy as np

# --- 1. Cargar el Modelo y el Escalador ---
# Asume que estos archivos fueron generados por clustering_salud_mental.py
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Archivos 'kmeans_model.pkl' o 'scaler.pkl' no encontrados. Asegúrate de ejecutar 'clustering_salud_mental.py' primero.")
    st.stop()

# --- 2. Definición de Descripciones de Clusters ---
# (AJUSTA ESTAS DESCRIPCIONES según tu análisis final del Cluster Summary)
CLUSTER_DESCRIPTIONS = {
    0: {
        "title": "Grupo 0: Riesgo Globalmente Bajo (Zona Segura)",
        "desc": "Población con las tasas de suicidio más bajas y baja mortalidad por enfermedades crónicas. Foco en la prevención general.",
        "color": "green"
    },
    1: {
        "title": "Grupo 1: Crisis Mental Aguda (Alta Probabilidad de Suicidio)",
        "desc": "Presenta tasas de suicidio extremadamente altas (dominadas por hombres). Requiere atención prioritaria e inmediata en salud mental.",
        "color": "red"
    },
    2: {
        "title": "Grupo 2: Riesgo Latente y Comorbilidad Física",
        "desc": "Tasas de suicidio medias, pero la mortalidad por enfermedades crónicas es la más alta. Necesita una intervención integrada (física y mental).",
        "color": "orange"
    },
    3: {
        "title": "Grupo 3: Alto Riesgo Social (Buena Salud Física)",
        "desc": "Altas tasas de suicidio con muy baja mortalidad por enfermedades crónicas. El riesgo es predominantemente social, económico o de apoyo comunitario.",
        "color": "blue"
    }
}

# --- 3. Interfaz de Streamlit ---
st.set_page_config(page_title="Clasificador de Riesgo de Salud Mental", layout="wide")

st.title("🧠 Clasificador de Grupo de Atención (K-Means)")
st.markdown("Utilice el modelo de clustering K-Means para clasificar un país o región en un grupo de riesgo basado en **tasas por 100,000 habitantes**.")

# 4. Formulario de Entrada
with st.form("input_form"):
    st.header("Parámetros de Entrada")
    col1, col2 = st.columns(2)

    with col1:
        suicide_rate_total = st.number_input(
            "Tasa de Suicidio (Ambos Sexos)", 
            min_value=0.0, value=9.94, step=0.1
        )
        suicide_rate_male = st.number_input(
            "Tasa de Suicidio (Hombres)", 
            min_value=0.0, value=15.0, step=0.1
        )
        
    with col2:
        suicide_rate_female = st.number_input(
            "Tasa de Suicidio (Mujeres)", 
            min_value=0.0, value=5.0, step=0.1
        )
        prob_dying_total = st.number_input(
            "Prob. de Morir 30-70 por Enfermedad (%)", 
            min_value=0.0, value=25.0, step=0.1
        )

    submit_button = st.form_submit_button("Clasificar Grupo de Riesgo")

# 5. Lógica de Predicción
if submit_button:
    # 5.1. Recopilar datos
    features = np.array([
        [suicide_rate_total, suicide_rate_male, suicide_rate_female, prob_dying_total]
    ])
    
    # 5.2. Escalar datos
    scaled_features = scaler.transform(features)
    
    # 5.3. Predecir cluster
    prediction = kmeans_model.predict(scaled_features)[0]
    
    # 5.4. Obtener resultado
    result = CLUSTER_DESCRIPTIONS.get(prediction)

    st.subheader(f"Resultado de Clasificación: {result['title']}")
    st.markdown(f"**Cluster ID:** `<span style='color: {result['color']}; font-weight: bold;'>{prediction}</span>`", unsafe_allow_html=True)
    st.info(result['desc'])
    