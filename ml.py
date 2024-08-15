# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DSkglDkK15FRRxMhcckvANpBXN4G9Ld0
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def split_data(datos, target_col):
    """Divide los datos en entrenamiento y prueba"""
    X = datos.drop(columns=[target_col])
    y = datos[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train):
    """Entrena un modelo de regresión lineal"""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    """Evalúa el modelo utilizando MSE"""
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    return mse

def predecir(modelo, X_new):
    """Realiza predicciones con un nuevo conjunto de datos"""
    predicciones = modelo.predict(X_new)
    return predicciones