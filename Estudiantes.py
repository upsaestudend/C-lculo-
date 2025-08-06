import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(page_title="Predicción Nota Cálculo", layout="centered")
st.title("📘 Predicción de Calificación en Cálculo")
st.markdown("Basado en notas previas y diagnóstico inicial")

# Cargar dataset desde archivo local
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

# Validación del dataset
if df is None or df.empty:
    st.stop()

# Renombrar columnas para consistencia
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Diagnostico': 'diagnostico',
    'Calificacion_Calculo': 'calculo'
})

# Seleccionar variables para el modelo
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones en conjunto de prueba
y_pred = modelo.predict(X_test)
# Limitar predicciones entre 0 y 100
y_pred = np.clip(y_pred, 0, 100)

# Métricas del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Métricas del Modelo")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("R²", f"{r2:.2f}")

# Mostrar coeficientes
st.subheader("📌 Coeficientes del Modelo")
st.dataframe(pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
}))

# Gráfico Real vs Predicho
st.subheader("📈 Real vs Predicho")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal', ax=ax)
ax.set_xlabel("Valor Real")
ax.set_ylabel("Valor Predicho")
ax.legend()
st.pyplot(fig)

# Vista previa del dataset
if st.checkbox("👀 Mostrar dataset completo"):
    st.dataframe(df)

# Tabla de predicciones
if st.checkbox("📋 Mostrar tabla de predicciones"):
    st.dataframe(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}))

# Formulario de predicción personalizada
st.subheader("🔍 Predicción Personalizada")
with st.form("formulario_prediccion"):
    aritmetica = st.number_input("Aritmética", 0.0, 100.0)
    algebra = st.number_input("Álgebra", 0.0, 100.0)
    geometria = st.number_input("Geometría Plana", 0.0, 100.0)
    trigonometria = st.number_input("Trigonometría", 0.0, 100.0)
    progresiones = st.number_input("Progresiones", 0.0, 100.0)
    diagnostico = st.number_input("Diagnóstico", 0.0, 100.0)
    submit = st.form_submit_button("Predecir Nota Final")

    if submit:
        entrada = [[aritmetica, algebra, geometria, trigonometria, progresiones, diagnostico]]
        prediccion = modelo.predict(entrada)[0]
        # Limitar predicción entre 0 y 100
        prediccion = np.clip(prediccion, 0, 100)
        st.success(f"📈 Nota predicha en Cálculo: {prediccion:.2f}")


