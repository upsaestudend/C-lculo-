import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración inicial
st.set_page_config(page_title="Predicción Nota Cálculo", layout="centered")
st.title("📘 Predicción de Calificación en Cálculo")
st.markdown("Modelo Ridge + Fórmula Ponderada 60% Diagnóstico / 40% Otras Materias")

# Cargar dataset
@st.cache_data
def cargar_dataset():
    try:
        df = pd.read_csv("dataset_estudiantes_final.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error cargando dataset: {e}")
        return None

df = cargar_dataset()
if df is None or df.empty:
    st.stop()

# Renombrar columnas
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Diagnostico': 'diagnostico',
    'Calificacion_Calculo': 'calculo'
})

# Separar variables
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# Entrenamiento modelo Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
y_pred = modelo_ridge.predict(X_test)
y_pred = np.clip(y_pred, 0, 100)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Métricas del Modelo Ridge")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("R²", f"{r2:.2f}")

# Coeficientes
st.subheader("📌 Coeficientes del Modelo Ridge")
st.dataframe(pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo_ridge.coef_
}))

# Gráfico Real vs Predicho
st.subheader("📈 Real vs Predicho (Ridge)")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal', ax=ax)
ax.set_xlabel("Valor Real")
ax.set_ylabel("Valor Predicho")
ax.legend()
st.pyplot(fig)

# Mostrar dataset completo (opcional)
if st.checkbox("👀 Mostrar dataset completo"):
    st.dataframe(df)

# Mostrar tabla de predicciones (opcional)
if st.checkbox("📋 Mostrar tabla de predicciones (Ridge)"):
    st.dataframe(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}))

# -----------------------------------------------
# 🔹 Formulario personalizado - Modelo Ridge
st.subheader("🔍 Predicción Personalizada (Modelo Ridge)")
with st.form("formulario_ridge"):
    aritmetica = st.number_input("Aritmética", 0.0, 100.0, key="arit_r")
    algebra = st.number_input("Álgebra", 0.0, 100.0, key="alg_r")
    geometria = st.number_input("Geometría Plana", 0.0, 100.0, key="geo_r")
    trigonometria = st.number_input("Trigonometría", 0.0, 100.0, key="tri_r")
    progresiones = st.number_input("Progresiones", 0.0, 100.0, key="pro_r")
    diagnostico = st.number_input("Diagnóstico", 0.0, 100.0, key="diag_r")
    submit_ridge = st.form_submit_button("Predecir con Ridge")

    if submit_ridge:
        entrada = [[aritmetica, algebra, geometria, trigonometria, progresiones, diagnostico]]
        prediccion = modelo_ridge.predict(entrada)[0]
        prediccion = np.clip(prediccion, 0, 100)
        st.success(f"📈 Nota predicha en Cálculo (Ridge): {prediccion:.2f}")

# -----------------------------------------------
# 🔸 Fórmula Personalizada 60% Diagnóstico / 40% Otras Materias
st.subheader("🔍 Predicción Personalizada (Fórmula 60/40)")
with st.form("formulario_manual"):
    aritmetica2 = st.number_input("Aritmética", 0.0, 100.0, key="arit_m")
    algebra2 = st.number_input("Álgebra", 0.0, 100.0, key="alg_m")
    geometria2 = st.number_input("Geometría Plana", 0.0, 100.0, key="geo_m")
    trigonometria2 = st.number_input("Trigonometría", 0.0, 100.0, key="tri_m")
    progresiones2 = st.number_input("Progresiones", 0.0, 100.0, key="pro_m")
    diagnostico2 = st.number_input("Diagnóstico", 0.0, 100.0, key="diag_m")
    submit_manual = st.form_submit_button("Predecir con Fórmula 60/40")

    if submit_manual:
        promedio_5 = np.mean([aritmetica2, algebra2, geometria2, trigonometria2, progresiones2])
        nota_final = 0.6 * diagnostico2 + 0.4 * promedio_5
        nota_final = np.clip(nota_final, 0, 100)
        st.success(f"📈 Nota predicha en Cálculo (60/40): {nota_final:.2f}")
