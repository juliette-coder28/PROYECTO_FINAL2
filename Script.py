import pandas as pd
from etl import extraer_datos, transformar_datos  # Importando funciones desde el módulo de ETL
from ml import split_data, entrenar_modelo, evaluar_modelo  # Importando funciones desde el módulo de ML
from visualizacion import grafico_distribucion  # Importando funciones desde el módulo de visualización
from sklearn.preprocessing import OneHotEncoder

# ETL: Extraer y transformar datos
file_path = '/content/Vegetable-fruits-Prices-2022.csv'
data = extraer_datos(file_path)  # Usando función del módulo ETL
data = transformar_datos(data)  # Transformación usando el módulo ETL

# One-Hot Encoding para las columnas categóricas
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(data[[col]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(col, axis=1)

# Guardar datos limpiados
cleaned_file_path = '/content/Vegetable-fruits-Prices-2022-cleaned.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Datos limpiados y guardados en '{cleaned_file_path}'")

# Visualización opcional
grafico_distribucion(data, 'RetailPrice', 'Distribución de Precios de Vegetales')

# Preparar datos para machine learning
target_column = 'RetailPrice'
X_train, X_test, y_train, y_test = split_data(data, target_column)

# Entrenar el modelo
modelo = entrenar_modelo(X_train, y_train)

# Evaluar el modelo
mse = evaluar_modelo(modelo, X_test, y_test)
print(f"Error cuadrático medio: {mse}")
