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

def horn_parallel_analysis(datos, metodo='pca', n_vars=2000, num_simulations=50, num_repeats=3, random_state=42):
    """
    Análisis paralelo (Horn) unificado para PCA y AF usando GPU (CuPy).

    Parámetros:
    -----------
    datos          : array estandarizado de entrenamiento (X_train_scaled)
    metodo         : 'pca' usa matriz de covarianza | 'af' usa matriz de correlación
    n_vars         : número máximo de variables a muestrear por repetición
    num_simulations: simulaciones de Monte Carlo para estimar el umbral de ruido
    num_repeats    : repeticiones para verificar estabilidad del resultado
    random_state   : semilla para reproducibilidad

    Retorna:
    --------
    lista de diccionarios con resultados por repetición
    """
    nombre_col = 'Componente' if metodo == 'pca' else 'Factor'
    print(f'🚀 Iniciando Análisis Paralelo de Horn [{metodo.upper()}] en GPU: {cp.cuda.runtime.getDeviceCount()} dispositivo(s).')
    rng = np.random.default_rng(random_state)
    n_samples, n_features = datos.shape
    resultados = []
    for rep in range(num_repeats):
        print(f'\n🔄 Repetición {rep + 1}/{num_repeats}')
        selected_vars = rng.choice(n_features, size=min(n_vars, n_features), replace=False)
        subset = datos[:, selected_vars]
        subset = StandardScaler().fit_transform(subset)
        subset_gpu = cp.asarray(subset)
        if metodo == 'pca':
            matrix = cp.cov(subset_gpu, rowvar=False)
        else:
            matrix = cp.corrcoef(subset_gpu, rowvar=False)
        eigvals_real_gpu = cp.linalg.eigvalsh(matrix)[::-1]
        eigvals_real = cp.asnumpy(eigvals_real_gpu)
        del subset_gpu, matrix, eigvals_real_gpu
        cp.get_default_memory_pool().free_all_blocks()
        rand_eigs = np.zeros((num_simulations, len(selected_vars)))
        for i in range(num_simulations):
            semilla = random_state + i + rep * 1000
            rnd_gpu = cp.random.default_rng(semilla)
            random_data_gpu = rnd_gpu.standard_normal(size=(n_samples, len(selected_vars)), dtype=cp.float32)
            if metodo == 'pca':
                cov_rnd_gpu = cp.cov(random_data_gpu, rowvar=False)
            else:
                cov_rnd_gpu = cp.corrcoef(random_data_gpu, rowvar=False)
            eigs_rnd_gpu = cp.linalg.eigvalsh(cov_rnd_gpu)[::-1]
            rand_eigs[i, :] = cp.asnumpy(eigs_rnd_gpu)
            del random_data_gpu, cov_rnd_gpu, eigs_rnd_gpu
            if i % 10 == 0:
                cp.get_default_memory_pool().free_all_blocks()
        mean_rand_eigs = rand_eigs.mean(axis=0)
        selected_mask = eigvals_real > mean_rand_eigs
        num_selected = selected_mask.sum()
        selected_indices = [idx + 1 for idx, v in enumerate(selected_mask) if v]
        if metodo == 'pca':
            total_var = eigvals_real.sum()
            var_individual = eigvals_real / total_var * 100
        else:
            var_individual = eigvals_real / len(selected_vars) * 100
        var_acumulada = np.cumsum(var_individual)
        tabla = pd.DataFrame({nombre_col: np.arange(1, len(selected_vars) + 1), 'Autovalor real': eigvals_real, 'Autovalor aleatorio medio': mean_rand_eigs, 'Varianza individual (%)': var_individual, 'Varianza acumulada (%)': var_acumulada, 'Seleccionado': selected_mask}).round(2)
        tabla_filtrada = tabla[tabla['Seleccionado']].reset_index(drop=True)
        resultados.append({'repeticion': rep + 1, f'num_{nombre_col.lower()}s': num_selected, f'{nombre_col.lower()}s': selected_indices, 'tabla': tabla_filtrada})
        var_final = tabla_filtrada['Varianza acumulada (%)'].iloc[-1] if not tabla_filtrada.empty else 0
        print(f'📌 {nombre_col}s retenidos: {num_selected} | Varianza acumulada: {var_final:.2f}%')
    return resultados

def criterio_varianza(datos, metodo='pca', n_vars=2000, umbral_varianza=94.0, num_repeats=3, random_state=42):
    """
    Selecciona componentes (PCA) o factores (AF) hasta que la varianza
    acumulada supere el umbral especificado (por defecto 94%).

    Parámetros:
    -----------
    datos           : array estandarizado de entrenamiento (X_train_scaled)
    metodo          : 'pca' usa matriz de covarianza | 'af' usa matriz de correlación
    n_vars          : número máximo de variables a muestrear por repetición
    umbral_varianza : porcentaje de varianza acumulada deseado (ej. 94.0)
    num_repeats     : repeticiones para verificar estabilidad del resultado
    random_state    : semilla para reproducibilidad

    Retorna:
    --------
    lista de diccionarios con resultados por repetición
    """
    nombre_col = 'Componente' if metodo == 'pca' else 'Factor'
    rng = np.random.default_rng(random_state)
    n_samples, n_features = datos.shape
    resultados = []
    print(f'🎯 Iniciando criterio de varianza [{metodo.upper()}]: buscando retener el {umbral_varianza}% de la varianza total...')
    for rep in range(num_repeats):
        print(f'🔄 Repetición {rep + 1}/{num_repeats}')
        sel = rng.choice(n_features, size=min(n_vars, n_features), replace=False)
        subset = datos[:, sel]
        subset_gpu = cp.asarray(subset)
        if metodo == 'pca':
            matrix = cp.cov(subset_gpu, rowvar=False)
        else:
            matrix = cp.corrcoef(subset_gpu, rowvar=False)
        eigvals_gpu = cp.linalg.eigvalsh(matrix)
        eigvals_real = cp.asnumpy(cp.flip(eigvals_gpu))
        del subset_gpu, matrix, eigvals_gpu
        cp.get_default_memory_pool().free_all_blocks()
        if metodo == 'pca':
            total_varianza = eigvals_real.sum()
        else:
            total_varianza = len(sel)
        var_indiv_pct = eigvals_real / total_varianza * 100.0
        var_acum_pct = np.cumsum(var_indiv_pct)
        indices_superan = np.where(var_acum_pct >= umbral_varianza)[0]
        if len(indices_superan) > 0:
            num_retenidos = indices_superan[0] + 1
        else:
            num_retenidos = len(sel)
        mask_selected = np.zeros(len(sel), dtype=bool)
        mask_selected[:num_retenidos] = True
        tabla = pd.DataFrame({nombre_col: np.arange(1, len(sel) + 1), 'Autovalor': eigvals_real, 'Varianza_individual_%': var_indiv_pct, 'Varianza_acumulada_%': var_acum_pct, 'Seleccionado': mask_selected}).round(3)
        tabla_filtrada = tabla[tabla['Seleccionado']].copy()
        var_final = tabla_filtrada['Varianza_acumulada_%'].iloc[-1]
        resultados.append({'repeticion': rep + 1, f'num_{nombre_col.lower()}s': num_retenidos, 'varianza_alcanzada': var_final, 'tabla_completa': tabla, 'tabla_filtrada': tabla_filtrada})
        print(f'✅ Repetición {rep + 1}: Se necesitan {num_retenidos} {nombre_col.lower()}s para explicar el {var_final:.2f}%')
    return resultados

def calcular_kmo(datos, n_vars=500, random_state=42):
    """
    Calcula la medida KMO (Kaiser-Meyer-Olkin) para evaluar la adecuación
    del Análisis Factorial sobre los datos de entrenamiento.

    Parámetros:
    -----------
    datos        : array estandarizado de entrenamiento (X_train_scaled)
    n_vars       : número de variables a muestrear para eficiencia computacional
    random_state : semilla para reproducibilidad

    Retorna:
    --------
    diccionario con KMO global y KMO por variable
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = datos.shape
    sel = rng.choice(n_features, size=min(n_vars, n_features), replace=False)
    subset = datos[:, sel]
    print(f'📐 Calculando KMO sobre {subset.shape[1]} variables y {subset.shape[0]} muestras...')
    subset_gpu = cp.asarray(subset)
    corr_gpu = cp.corrcoef(subset_gpu, rowvar=False)
    corr = cp.asnumpy(corr_gpu)
    del subset_gpu, corr_gpu
    cp.get_default_memory_pool().free_all_blocks()
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        print('⚠️ Matriz singular detectada, usando pseudoinversa.')
        inv_corr = np.linalg.pinv(corr)
    diag = np.sqrt(np.diag(inv_corr))
    partial_corr = -inv_corr / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 1.0)
    corr_sq = corr ** 2
    partial_corr_sq = partial_corr ** 2
    np.fill_diagonal(corr_sq, 0)
    np.fill_diagonal(partial_corr_sq, 0)
    kmo_global = corr_sq.sum() / (corr_sq.sum() + partial_corr_sq.sum())
    kmo_por_var = corr_sq.sum(axis=1) / (corr_sq.sum(axis=1) + partial_corr_sq.sum(axis=1))
    return {'kmo_global': kmo_global, 'kmo_por_variable': kmo_por_var}

def entropia_shannon(vec):
    """
    Calcula la entropía de Shannon de un vector de valores.
    Normaliza los valores absolutos para obtener una distribución
    de probabilidad y calcula H = -sum(p * log(p)).

    Parámetros:
    -----------
    vec : vector de valores (componentes, factores o proyecciones)

    Retorna:
    --------
    valor de entropía (float)
    """
    p = np.abs(vec) / np.sum(np.abs(vec))
    p = p[p > 0]
    return -np.sum(p * np.log(p))

