# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ── Cargar modelo ────────────────────────────────────────────────────────────
filename = 'modelo-class.pkl'
obj      = pickle.load(open(filename, 'rb'))
modelo   = obj[0]
encoder  = obj[1]
variables = obj[2]   # 9 columnas que espera el modelo
scaler   = obj[3]    # MinMaxScaler entrenado SOLO sobre age y avg_glucose_level

# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title('Predicción para ataque cerebrovascular')

age               = st.slider('Edad',               min_value=1,     max_value=82,    value=20,    step=1)
avg_glucose_level = st.slider('Nivel glucosa prom.', min_value=55.12, max_value=271.74, value=100.0, step=1.0)
hypertension      = st.selectbox('Hipertensión',    ['No', 'Yes'])
heart_disease     = st.selectbox('Enfermedad cardíaca', ['No', 'Yes'])
ever_married      = st.selectbox('Alguna vez casado/a', ['Yes', 'No'])
smoking_status    = st.selectbox('Estado de fumador', [
    "never smoked", "Unknown", "formerly smoked", "smokes"
])

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("Predecir"):

    # 1. Construir DataFrame con las 9 columnas que espera el modelo
    df = pd.DataFrame([[0] * len(variables)], columns=variables)

    # 2. Escalar SOLO las variables numéricas (el scaler fue entrenado con 2 features)
    numericas = pd.DataFrame([[age, avg_glucose_level]],
                             columns=['age', 'avg_glucose_level'])
    numericas_scaled = scaler.transform(numericas)

    df.at[0, 'age']               = numericas_scaled[0][0]
    df.at[0, 'avg_glucose_level'] = numericas_scaled[0][1]

    # 3. Variables binarias (one-hot)
    if hypertension == 'Yes':
        df.at[0, 'hypertension_Yes'] = 1

    if heart_disease == 'Yes':
        df.at[0, 'heart_disease_Yes'] = 1

    if ever_married == 'Yes':
        df.at[0, 'ever_married_Yes'] = 1

    # 4. Smoking status — buscar la columna correcta por coincidencia parcial
    smoking_map = {
        "formerly smoked": "smoking_status_'formerly smoked'",
        "never smoked":    "smoking_status_'never smoked'",
        "Unknown":         "smoking_status_Unknown",
        "smokes":          "smoking_status_smokes",
    }
    col_smoking = smoking_map.get(smoking_status)
    if col_smoking and col_smoking in df.columns:
        df.at[0, col_smoking] = 1

    # 5. Verificación de seguridad
    assert df.shape[1] == modelo.n_features_in_, (
        f"Mismatch de features: df tiene {df.shape[1]}, "
        f"modelo espera {modelo.n_features_in_}"
    )

    # 6. Predecir
    pred = modelo.predict(df)

    if pred[0] == 1:
        st.error("⚠️ Alto riesgo de ataque cerebrovascular")
    else:
        st.success("✅ Bajo riesgo de ataque cerebrovascular")

st.warning("⚠️ Este modelo es una herramienta de apoyo y no reemplaza el diagnóstico médico.")
