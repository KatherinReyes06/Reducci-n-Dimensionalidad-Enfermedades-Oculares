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

def reportar_svm(resultados_por_dataset, y_test, encoder, condicion='SIN BALANCE', color_mapa='Blues'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    combinación de reducción dimensional y kernel evaluada.

    Parámetros:
    -----------
    resultados_por_dataset : diccionario {dataset: {kernel: resultado}}
    y_test                 : etiquetas reales de prueba
    encoder                : LabelEncoder con nombres de clases
    condicion              : texto descriptivo de la condición de balanceo
    color_mapa             : color del mapa de la matriz de confusión
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  SVM [{condicion}]')
    print(f'{'=' * 70}')
    for nombre_dataset, resultados_kernels in resultados_por_dataset.items():
        print(f'\n{'─' * 50}')
        print(f'  Datos: {nombre_dataset}')
        print(f'{'─' * 50}')
        for kernel, res in resultados_kernels.items():
            print(f'\n  Kernel: {kernel.upper()}')
            print(f'  Accuracy : {res['accuracy']:.4f}')
            print(f'  Tiempo   : {res['tiempo']:.4f}s')
            print(f'\n--- Reporte de clasificación ---')
            print(classification_report(y_test, res['y_pred'], target_names=clases_nombres, zero_division=0))
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, res['y_pred'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_nombres)
            disp.plot(cmap=color_mapa, ax=ax, values_format='d', colorbar=True)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            plt.title(f'SVM Kernel {kernel.upper()} [{condicion}]\nDatos: {nombre_dataset}')
            plt.tight_layout()
            plt.show()

def reportar_lda(resultados_por_dataset, y_test, encoder, condicion='SIN BALANCE', color_mapa='Oranges'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    representación reducida evaluada.
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  LDA [{condicion}]')
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
        plt.title(f'LDA [{condicion}]\nDatos: {nombre}')
        plt.tight_layout()
        plt.show()

def reportar_resultados(resultados_por_dataset, y_test, encoder, titulo, color_mapa='Blues'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    representación reducida evaluada.
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  {titulo}')
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
        plt.title(f'{titulo}\nDatos: {nombre}')
        plt.tight_layout()
        plt.show()

def reportar_naive_bayes(resultados_por_dataset, y_test, encoder, titulo, color_mapa='Oranges'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    representación reducida evaluada en los modelos Naive Bayes.

    Parámetros:
    -----------
    resultados_por_dataset : diccionario con resultados por reducción
    y_test                 : etiquetas reales de prueba
    encoder                : LabelEncoder con nombres de clases
    titulo                 : título descriptivo del modelo
    color_mapa             : color para la matriz de confusión
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  {titulo}')
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
        plt.title(f'{titulo}\nDatos: {nombre}')
        plt.tight_layout()
        plt.show()

def reportar_qda(resultados_por_dataset, y_test, encoder, condicion='SIN BALANCE', color_mapa='Reds'):
    """
    Genera reporte de clasificación y matriz de confusión para cada
    representación reducida evaluada en QDA.
    """
    clases_nombres = encoder.classes_
    print(f'\n{'=' * 70}')
    print(f'  QDA [{condicion}]')
    print(f'{'=' * 70}')
    for nombre, res in resultados_por_dataset.items():
        print(f'\n>>> {nombre}')
        print(f'    Accuracy : {res['accuracy']:.4f}')
        print(f'    Tiempo   : {res['tiempo']:.4f}s')
        print(f'\n--- Reporte de clasificación ---')
        print(classification_report(y_test, res['y_pred'], target_names=clases_nombres, zero_division=0))
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, res['y_pred'], display_labels=clases_nombres, cmap=color_mapa, ax=ax, colorbar=True)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        plt.title(f'QDA [{condicion}]\nDatos: {nombre}')
        plt.tight_layout()
        plt.show()

def reportar_logreg_pixeles(resultado, y_test, etiqueta, nombres_clases, color_mapa='Blues'):
    """
    Genera reporte de clasificación y matriz de confusión.
    """
    y_pred = resultado['y_pred']
    print(f'\n🏆 Accuracy: {resultado['accuracy']:.4f}')
    print(f'⏱️  Tiempo  : {resultado['tiempo']:.2f}s')
    print(f'\n--- Reporte de clasificación [{etiqueta}] ---')
    print(classification_report(y_test, y_pred, target_names=nombres_clases, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres_clases)
    disp.plot(cmap=color_mapa, ax=ax, values_format='d', colorbar=True)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'Regresión Logística (Píxeles) [{etiqueta}]')
    plt.tight_layout()
    plt.show()

def reportar_svm_pixeles(resultado, y_test, etiqueta, nombres_clases, color_mapa='Blues'):
    """
    Genera reporte de clasificación y matriz de confusión para SVM.
    """
    y_pred = resultado['y_pred']
    print(f'\n🏆 Accuracy: {resultado['accuracy']:.4f}')
    print(f'⏱️  Tiempo  : {resultado['tiempo']:.2f}s')
    print(f'\n--- Reporte de clasificación [{etiqueta}] ---')
    print(classification_report(y_test, y_pred, target_names=nombres_clases, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres_clases)
    disp.plot(cmap=color_mapa, ax=ax, values_format='d', colorbar=True)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'SVM-RBF (Píxeles) [{etiqueta}]')
    plt.tight_layout()
    plt.show()


def generar_reporte_cnn(modelo, generador_val, titulo):
    """
    Genera reporte de clasificación y matriz de confusión para CNN.
    """
    generador_val.reset()
    predicciones  = modelo.predict(generador_val,
                                   steps=len(generador_val), verbose=1)
    y_pred        = np.argmax(predicciones, axis=1)
    y_real        = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    acc = accuracy_score(y_real, y_pred)
    print(f"\n🏆 Accuracy final: {acc:.4f}")

    print(f"\n--- Reporte de clasificación: {titulo} ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — {titulo}')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return acc


def generar_reporte_mlp_pixeles(modelo, X_test, y_test_onehot,
                                 nombres_clases, titulo):
    """
    Genera reporte de clasificación y matriz de confusión para MLP.
    """
    predicciones   = modelo.predict(X_test, verbose=0)
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = np.argmax(y_test_onehot, axis=1)

    acc = accuracy_score(y_real, y_pred)
    print(f"\n🏆 Accuracy final: {acc:.4f}")

    print(f"\n--- Reporte de clasificación [{titulo}] ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — MLP (Píxeles) [{titulo}]')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return acc


def generar_reporte_vit(modelo, generador_val, titulo):
    """
    Genera reporte de clasificación y matriz de confusión para ViT.
    Retorna accuracy Y predicciones (probabilidades softmax) para checkpoint.
    """
    generador_val.reset()
    predicciones   = modelo.predict(generador_val, verbose=1)
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    acc = accuracy_score(y_real, y_pred)
    print(f"\n🏆 Accuracy final: {acc:.4f}")

    print(f"\n--- Reporte de clasificación: {titulo} ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — {titulo}')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    # ── Retorna acc Y predicciones para checkpoint ─────────────────────────
    return acc, predicciones

