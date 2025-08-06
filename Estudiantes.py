import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración
st.set_page_config(page_title="Predicción Cálculo", layout="centered")
st.title("📘 Modelo de Regresión para Cálculo")
st.markdown("Predicción basada en materias anteriores y diagnóstico")

# URL del dataset (reemplaza si tienes otro link o archivo local)
URL_CSV = "https://raw.githubusercontent.com/openai-streamlit/data/main/dataset_estudiantes_final.csv"

# Función segura para cargar datos
@st.cache_data
def cargar_dataset(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error cargando dataset: {e}")
        print(f"Error cargando dataset: {e}")
        return None

# Cargar dataset desde URL
df = cargar_dataset(URL_CSV)

# Validar carga
if df is None or df.empty:
    st.error("❌ El dataset no se cargó correctamente o está vacío.")
    st.stop()

# Renombrar columnas a minúsculas consistentes
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Calculo': 'calculo',
    'Calificacion_Diagnostico': 'diagnostico'
})

# Validar columnas requeridas
columnas_requeridas = ['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico', 'calculo']
faltan = [col for col in columnas_requeridas if col not in df.columns]
if faltan:
    st.error(f"❌ Faltan columnas necesarias: {faltan}")
    st.stop()

# Vista previa
st.subheader("👀 Vista previa del Dataset")
st.dataframe(df.head())

# Separar variables
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Métricas del Modelo")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("R²", f"{r2:.2f}")

# Coeficientes
st.subheader("📌 Coeficientes del Modelo")
st.dataframe(pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
}))

# Gráfico Real vs Predicho
st.subheader("📈 Real vs Predicho")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
sns.lineplot(x=y_test, y=y_test, color='red', label='Línea Ideal', ax=ax)
ax.set_xlabel("Valor Real")
ax.set_ylabel("Valor Predicho")
ax.legend()
st.pyplot(fig)

# Tabla de predicciones opcional
if st.checkbox("📋 Mostrar tabla de predicciones"):
    st.dataframe(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}))

# Predicción interactiva
st.subheader("🔍 Predicción Personalizada")
with st.form("formulario_prediccion"):
    aritmetica = st.number_input("Aritmética", 0.0, 100.0)
    algebra = st.number_input("Álgebra", 0.0, 100.0)
    geometria = st.number_input("Geometría Plana", 0.0, 100.0)
    trigonometria = st.number_input("Trigonometría", 0.0, 100.0)
    progresiones = st.number_input("Progresiones", 0.0, 100.0)
    diagnostico = st.number_input("Calificación Diagnóstico", 0.0, 100.0)
    submit = st.form_submit_button("Predecir Nota Final")
    
    if submit:
        entrada = [[aritmetica, algebra, geometria, trigonometria, progresiones, diagnostico]]
        prediccion = modelo.predict(entrada)[0]
        st.success(f"📈 Nota predicha en Cálculo: {prediccion:.2f}")
