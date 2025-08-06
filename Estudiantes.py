import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset local
df = pd.read_csv("dataset_estudiantes_final.csv")

# 2. Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# 3. Renombrar columnas para consistencia (opcional pero recomendado)
df = df.rename(columns={
    'Nota_Aritmetica': 'aritmetica',
    'Nota_Algebra': 'algebra',
    'Nota_Geometria_Plana': 'geometria_plana',
    'Nota_Trigonometria': 'trigonometria',
    'Nota_Progresiones': 'progresiones',
    'Calificacion_Calculo': 'calculo',
    'Calificacion_Diagnostico': 'diagnostico'
})

# 4. Selección de variables
X = df[['aritmetica', 'algebra', 'geometria_plana', 'trigonometria', 'progresiones', 'diagnostico']]
y = df['calculo']

# 5. División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 7. Predicciones
y_pred = modelo.predict(X_test)

# 8. Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 9. Resultados
print("Coeficientes del modelo:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("Error cuadrático medio (MSE):", round(mse, 2))
print("Coeficiente de determinación (R²):", round(r2, 2))

# Mostrar coeficientes bien organizados
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
})
print("\nCoeficientes detallados:")
print(coef_df)

