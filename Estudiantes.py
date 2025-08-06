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

# ------------------------------------------------
# 🔍 Predicción Personalizada (Ridge + Fórmula 60/40)
st.subheader("🔍 Predicción Personalizada (Ambos Modelos)")
with st.form("formulario_unico"):
    aritmetica = st.number_input("Aritmética", 0.0, 100.0)
    algebra = st.number_input("Álgebra", 0.0, 100.0)
    geometria = st.number_input("Geometría Plana", 0.0, 100.0)
    trigonometria = st.number_input("Trigonometría", 0.0, 100.0)
    progresiones = st.number_input("Progresiones", 0.0, 100.0)
    diagnostico = st.number_input("Diagnóstico", 0.0, 100.0)
    submit = st.form_submit_button("Predecir")

    if submit:
        entrada = [[aritmetica, algebra, geometria, trigonometria, progresiones, diagnostico]]

        # ➤ Ridge
        pred_ridge = modelo_ridge.predict(entrada)[0]
        pred_ridge = np.clip(pred_ridge, 0, 100)

        # ➤ Fórmula 60/40
        promedio_5 = np.mean([aritmetica, algebra, geometria, trigonometria, progresiones])
        pred_manual = 0.6 * diagnostico + 0.4 * promedio_5
        pred_manual = np.clip(pred_manual, 0, 100)

        # Mostrar resultados
        st.success(f"📈 Nota predicha en Cálculo (Ridge): {pred_ridge:.2f}")
        st.info(f"📊 Nota predicha con Fórmula 60/40: {pred_manual:.2f}")
