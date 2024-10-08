# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DSkglDkK15FRRxMhcckvANpBXN4G9Ld0
"""

import matplotlib.pyplot as plt

def grafico_distribucion(datos, columna, titulo):
    """Crea un histograma de la columna dada"""
    plt.hist(datos[columna], bins=20, color='skyblue', edgecolor='black')
    plt.title(titulo)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.show()