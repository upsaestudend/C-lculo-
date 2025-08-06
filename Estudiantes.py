import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuración inicial
st.set_page_config(page_title="Modelo de Cálculo", layout="centered")
st.title("📘 Predicción de Calificación en Cálculo usando Regresión Lineal")

# Opción para subir archivo o usar GitHub
st.sidebar.header("📁 Cargar Dataset")
origen = st.sidebar.radio("Selecciona el origen de los datos:", ["Subir archivo", "Desde GitHub"])

df = None

if origen == "Subir archivo":
    archivo = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            st.stop()
    else:
        st.warning("📂 Esperando que subas un archivo.")
        st.stop()

elif origen == "Desde GitHub":
    url = st.sidebar.text_input("🔗 Pega aquí el enlace RAW al CSV en GitHub:")
    if url:
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo desde GitHub: {e}")
            st.stop()
    else:
        st.warning("🔎 Ingresa una URL válida para cargar el dataset.")
        st.stop()

# Validación del DataFrame
if df is None or df.empty:
    st.error("⚠️ El dataset no se cargó correctamente o está vacío.")
