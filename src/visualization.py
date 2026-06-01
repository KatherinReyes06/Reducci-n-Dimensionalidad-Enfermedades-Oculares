from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter
import tensorflow_hub as hub
from tensorflow.keras.applications import EfficientNetB3
from google.colab import files
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.manifold import trustworthiness, _utils
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import cupy as cp
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from sklearn.utils import class_weight
from sklearn.naive_bayes import GaussianNB, ComplementNB
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import os
import seaborn as sns
import warnings
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks, utils, initializers, optimizers
from tensorflow.keras.applications import DenseNet121
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pandas as pd
import time
from sklearn.feature_selection import SelectFromModel
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, Flatten, MaxPooling2D
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from imblearn.combine import SMOTEENN
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import nbformat, os
from sklearn.exceptions import UndefinedMetricWarning
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import shutil
from imblearn.over_sampling import SMOTE
from factor_analyzer import FactorAnalyzer
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import umap
from tensorflow.keras import regularizers
from google.colab import drive
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier, plot_tree

def graficar_horn(resultados, metodo='pca'):
    """
    Grafica el análisis paralelo de Horn mostrando los autovalores reales
    versus el umbral de ruido aleatorio. El punto de cruce indica el número
    óptimo de componentes o factores a retener.

    Parámetros:
    -----------
    resultados : salida de horn_parallel_analysis()
    metodo     : 'pca' o 'af' para ajustar etiquetas del gráfico
    """
    col_x = 'Componente' if metodo == 'pca' else 'Factor'
    titulo = 'Componentes (PCA)' if metodo == 'pca' else 'Factores (AF)'
    df_graf = resultados[-1]['tabla']
    plt.figure(figsize=(12, 6))
    plt.plot(df_graf[col_x], df_graf['Autovalor real'], 'o-', color='blue', label='Autovalores Reales', markersize=4, alpha=0.8)
    plt.plot(df_graf[col_x], df_graf['Autovalor aleatorio medio'], '--', color='red', label='Ruido Aleatorio (Umbral de Horn)', linewidth=2)
    plt.title(f'Análisis Paralelo de Horn — Selección de {titulo}', fontsize=14)
    plt.xlabel(f'Número de {col_x}', fontsize=12)
    plt.ylabel('Autovalor (Eigenvalue)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_varianza_acumulada(resultados, metodo='pca'):
    """
    Grafica la curva de varianza acumulada y marca el punto de corte
    donde se alcanza el umbral de Yeomans-Golder.
    """
    col_x = 'Componente' if metodo == 'pca' else 'Factor'
    titulo = 'PCA' if metodo == 'pca' else 'Análisis Factorial'
    color = 'blue' if metodo == 'pca' else 'purple'
    res = resultados[-1]
    df_yg = res['tabla_completa']
    corte = res[f'num_{col_x.lower()}s']
    var_corte = res['varianza_alcanzada']
    plt.figure(figsize=(10, 6))
    plt.plot(df_yg[col_x], df_yg['Varianza_acumulada_%'], label='Varianza Acumulada', color=color, linewidth=2)
    plt.axhline(y=94.0, color='red', linestyle='--', label='Umbral 94%')
    plt.plot(corte, var_corte, 'ro', markersize=8)
    plt.annotate(f'Corte: {corte} {col_x}s\n({var_corte:.2f}%)', xy=(corte, var_corte), xytext=(corte + 5, var_corte - 10), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold')
    plt.title(f'{titulo} — Varianza Acumulada (Yeomans-Golder)\n{col_x}s necesarios: {corte}', fontsize=14)
    plt.xlabel(f'Número de {col_x}s')
    plt.ylabel('Varianza Acumulada (%)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, corte + 50)
    plt.ylim(35, 105)
    plt.tight_layout()
    plt.show()

def graficar_kaiser(resultados, metodo='pca'):
    """
    Grafica el criterio de Kaiser (λ ≥ 1) mostrando qué componentes
    o factores superan el umbral de autovalor igual a 1.
    """
    col_x = 'Componente' if metodo == 'pca' else 'Factor'
    titulo = 'PCA' if metodo == 'pca' else 'AF'
    res_ultimo = resultados[-1]
    df_k = res_ultimo['tabla_completa']
    autovalores = df_k['Autovalor']
    num_kaiser = sum(autovalores >= 1.0)
    mask_aceptados = autovalores >= 1.0
    mask_descartados = autovalores < 1.0
    plt.figure(figsize=(12, 7))
    plt.plot(df_k[col_x], autovalores, '-', color='gray', alpha=0.4, zorder=1)
    plt.scatter(df_k[col_x][mask_aceptados], autovalores[mask_aceptados], color='blue', s=60, label=f'Retenidos (λ ≥ 1)', zorder=2)
    plt.scatter(df_k[col_x][mask_descartados], autovalores[mask_descartados], color='darkorange', s=50, alpha=0.7, label='Descartados (λ < 1)', zorder=2)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Criterio Kaiser (λ = 1)')
    plt.title(f'Gráfico de Sedimentación ({titulo})\n{col_x}s con λ ≥ 1: {num_kaiser}', fontsize=16)
    plt.xlabel(f'Número de {col_x}', fontsize=12)
    plt.ylabel('Autovalor (Eigenvalue)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-50, autovalores.max() + 50)
    plt.xlim(0, num_kaiser + 30)
    plt.tight_layout()
    plt.show()
    print(f'\n📋 Últimos 5 {col_x.lower()}s que cumplen Kaiser (λ ≥ 1):')
    display(df_k[mask_aceptados].tail(5))

def interpretar_kmo(kmo_valor):
    """Retorna la interpretación textual del valor KMO."""
    if kmo_valor >= 0.9:
        return 'Excelente ✅'
    elif kmo_valor >= 0.8:
        return 'Bueno ✅'
    elif kmo_valor >= 0.7:
        return 'Aceptable ✅'
    elif kmo_valor >= 0.6:
        return 'Mediocre ⚠️'
    elif kmo_valor >= 0.5:
        return 'Malo ⚠️'
    else:
        return 'Inaceptable ❌ — AF no recomendado'

def graficar_kmo(resultados_kmo):
    """
    Visualiza la distribución del KMO por variable y la clasificación
    del KMO global según la escala estándar de interpretación.
    """
    kmo_vars = resultados_kmo['kmo_por_variable']
    kmo_global = resultados_kmo['kmo_global']
    interpretacion = interpretar_kmo(kmo_global)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(kmo_vars, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Umbral mínimo (0.50)')
    axes[0].axvline(x=kmo_global, color='green', linestyle='-', linewidth=2, label=f'KMO Global: {kmo_global:.4f}')
    axes[0].set_title('Distribución del KMO por Variable', fontsize=13)
    axes[0].set_xlabel('Valor KMO')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    categorias = ['Inaceptable\n(<0.50)', 'Malo\n(0.50-0.59)', 'Mediocre\n(0.60-0.69)', 'Aceptable\n(0.70-0.79)', 'Bueno\n(0.80-0.89)', 'Excelente\n(0.90-1.00)']
    colores = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    umbrales = [0.5, 0.59, 0.69, 0.79, 0.89, 1.0]
    bars = axes[1].barh(categorias, [1] * 6, color=colores, alpha=0.7, edgecolor='white')
    for i, val in enumerate(umbrales):
        if kmo_global <= val:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(3)
            axes[1].text(1.05, i, f'◄ Tu KMO: {kmo_global:.4f}', va='center', fontsize=11, fontweight='bold')
            break
    axes[1].set_xlim(0, 1.3)
    axes[1].set_title(f'Clasificación KMO Global\nValor: {kmo_global:.4f} — {interpretacion}', fontsize=13)
    axes[1].set_xlabel('Escala de referencia')
    axes[1].grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

def reportar_logreg(resultados_por_dataset, y_test, encoder, condicion='SIN BALANCE', color_mapa='Blues'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    representación reducida evaluada.

    Parámetros:
    -----------
    resultados_por_dataset : diccionario con resultados por reducción
    y_test                 : etiquetas reales de prueba
    encoder                : LabelEncoder con nombres de clases
    condicion              : texto descriptivo de la condición de balanceo
    color_mapa             : color del mapa de la matriz de confusión
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  REGRESIÓN LOGÍSTICA [{condicion}]')
    print(f'{'=' * 70}')
    for nombre, res in resultados_por_dataset.items():
        print(f'\n>>> {nombre}')
        print(f'    Accuracy : {res['accuracy']:.4f}')
        print(f'    Tiempo   : {res['tiempo']:.4f}s')
        print(f'\n--- Reporte de clasificación ---')
        print(classification_report(y_test, res['y_pred'], target_names=clases_nombres, zero_division=0))
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, res['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_nombres)
        disp.plot(cmap=color_mapa, ax=ax, values_format='d', colorbar=True)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        plt.title(f'Regresión Logística [{condicion}]\nDatos: {nombre}')
        plt.tight_layout()
        plt.show()

def graficar_arbol(modelo, etiqueta, encoder, n_componentes, profundidad=3):
    """
    Visualiza la estructura del árbol de decisión hasta una profundidad dada.

    Parámetros:
    -----------
    modelo       : modelo DecisionTreeClassifier entrenado
    etiqueta     : nombre descriptivo para el título
    encoder      : LabelEncoder con nombres de clases
    n_componentes: número de componentes del espacio reducido
    profundidad  : niveles del árbol a mostrar (default 3)
    """
    clases_nombres = encoder.classes_
    nombres_features = [f'Comp {i + 1}' for i in range(n_componentes)]
    plt.figure(figsize=(34, 14), dpi=100)
    anotaciones = plot_tree(modelo, max_depth=profundidad, feature_names=nombres_features, class_names=clases_nombres, filled=True, rounded=True, fontsize=11, precision=2, impurity=False)
    for texto in anotaciones:
        lineas = texto.get_text().split('\n')
        lineas_limpias = [l for l in lineas if not l.startswith('value') and '[' not in l and (']' not in l)]
        texto.set_text('\n'.join(lineas_limpias))
    plt.title(f'Estructura del Árbol — {etiqueta} (Profundidad {profundidad})', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def graficar_curvas(historial, titulo):
    """
    Grafica las curvas de exactitud y pérdida de entrenamiento y validación.
    """
    exactitud = historial.history['accuracy']
    val_exactitud = historial.history['val_accuracy']
    perdida = historial.history['loss']
    val_perdida = historial.history['val_loss']
    epocas = range(len(exactitud))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14)
    ax1.plot(epocas, exactitud, label='Entrenamiento')
    ax1.plot(epocas, val_exactitud, label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epocas, perdida, label='Entrenamiento')
    ax2.plot(epocas, val_perdida, label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_curvas_vit(historial, titulo):
    epocas = range(len(historial.history['accuracy']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14)
    ax1.plot(epocas, historial.history['accuracy'], label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epocas, historial.history['loss'], label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_curvas_mlp(historial, titulo):
    """
    Grafica las curvas de exactitud y pérdida de entrenamiento y validación.
    """
    epocas = range(len(historial.history['accuracy']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(titulo, fontsize=14)
    ax1.plot(epocas, historial.history['accuracy'], label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epocas, historial.history['loss'], label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

