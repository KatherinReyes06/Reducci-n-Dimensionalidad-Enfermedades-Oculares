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

def entrenar_logreg(X_train, y_train, X_test, y_test, balanceado=False):
    """
    Entrena y evalúa un modelo de Regresión Logística.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    balanceado : si True usa class_weight='balanced', si False no pondera clases

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = LogisticRegression(class_weight='balanced' if balanceado else None, C=0.1, max_iter=5000, random_state=42, solver='saga', multi_class='multinomial', n_jobs=-1)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_svm(X_train, y_train, X_test, y_test, kernel='rbf', balanceado=False):
    """
    Entrena y evalúa un modelo SVM con el kernel especificado.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    kernel     : tipo de kernel ('linear', 'rbf', 'poly', 'sigmoid')
    balanceado : si True usa class_weight='balanced'

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo, predicciones e iteraciones
    """
    limite_iter = 10000
    modelo = SVC(kernel=kernel, class_weight='balanced' if balanceado else None, C=10, gamma='scale', degree=3, coef0=1.0, cache_size=1000, max_iter=limite_iter, random_state=42)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if np.max(modelo.n_iter_) >= limite_iter:
        print(f'    ⚠️ Kernel {kernel.upper()}: se alcanzó el límite de iteraciones ({limite_iter}). Considera aumentar max_iter.')
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred, 'n_iter': modelo.n_iter_}

def entrenar_cnn1d(X_train, y_train, X_test, y_test, encoder, semilla=42, epocas=80, tam_lote=32, pesos_clases=None):
    """
    Entrena y evalúa una CNN 1D con una semilla específica.

    Parámetros:
    -----------
    X_train       : datos de entrenamiento reducidos
    y_train       : etiquetas de entrenamiento
    X_test        : datos de prueba reducidos
    y_test        : etiquetas de prueba
    encoder       : LabelEncoder con nombres de clases
    semilla       : semilla para reproducibilidad (varía en Monte Carlo)
    epocas        : número máximo de épocas (EarlyStopping detiene antes)
    tam_lote      : tamaño del lote de entrenamiento
    pesos_clases  : diccionario de pesos por clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, pérdida, historial y predicciones
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    n_componentes = X_train.shape[1]
    num_clases = len(encoder.classes_)
    X_train_c = X_train.reshape((-1, n_componentes, 1))
    X_test_c = X_test.reshape((-1, n_componentes, 1))
    modelo = construir_cnn1d(n_componentes, num_clases)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    detener_pronto = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reducir_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    historial = modelo.fit(X_train_c, y_train, validation_split=0.15, epochs=epocas, batch_size=tam_lote, class_weight=pesos_clases, callbacks=[detener_pronto, reducir_lr], verbose=0)
    perdida, acc = modelo.evaluate(X_test_c, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(X_test_c, verbose=0), axis=1)
    return {'modelo': modelo, 'accuracy': acc, 'perdida': perdida, 'historial': historial, 'y_pred': y_pred}

def entrenar_transformer(X_train, y_train, X_test, y_test, encoder, semilla=42, epocas=50, tam_lote=32, pesos_clases=None):
    """
    Parámetros:
    -----------
    X_train      : datos de entrenamiento reducidos
    y_train      : etiquetas de entrenamiento
    X_test       : datos de prueba reducidos
    y_test       : etiquetas de prueba
    encoder      : LabelEncoder con nombres de clases
    semilla      : semilla para Monte Carlo
    epocas       : número máximo de épocas
    tam_lote     : tamaño del lote
    pesos_clases : diccionario de pesos por clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, pérdida, historial y predicciones
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    n_caracteristicas = X_train.shape[1]
    n_clases = len(encoder.classes_)
    modelo = construir_transformer(n_caracteristicas, n_clases)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-05), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    detener_pronto = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reducir_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06)
    inicio = time.time()
    historial = modelo.fit(X_train, y_train, validation_split=0.15, epochs=epocas, batch_size=tam_lote, class_weight=pesos_clases, callbacks=[detener_pronto, reducir_lr], verbose=0)
    duracion = time.time() - inicio
    perdida, acc = modelo.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    return {'modelo': modelo, 'accuracy': acc, 'perdida': perdida, 'historial': historial, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_mlp(X_train, y_train, X_test, y_test, encoder, semilla=42, epocas=50, tam_lote=32, pesos_clases=None):
    """
    Entrena y evalúa un MLP profundo con una semilla específica.

    Parámetros:
    -----------
    X_train      : datos de entrenamiento reducidos
    y_train      : etiquetas de entrenamiento
    X_test       : datos de prueba reducidos
    y_test       : etiquetas de prueba
    encoder      : LabelEncoder con nombres de clases
    semilla      : semilla para Monte Carlo
    epocas       : número máximo de épocas
    tam_lote     : tamaño del lote
    pesos_clases : diccionario de pesos por clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, pérdida, historial y predicciones
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    n_caracteristicas = X_train.shape[1]
    n_clases = len(encoder.classes_)
    modelo = construir_mlp(n_caracteristicas, n_clases)
    detener_pronto = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reducir_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-06)
    inicio = time.time()
    historial = modelo.fit(X_train, y_train, validation_split=0.15, epochs=epocas, batch_size=tam_lote, class_weight=pesos_clases, callbacks=[detener_pronto, reducir_lr], verbose=0)
    duracion = time.time() - inicio
    perdida, acc = modelo.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    return {'modelo': modelo, 'accuracy': acc, 'perdida': perdida, 'historial': historial, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_mlp_sklearn(X_train, y_train, X_test, y_test, semilla=42):
    """
    Entrena y evalúa un MLPClassifier de Scikit-learn con una semilla
    específica.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba
    semilla : semilla para Monte Carlo

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.1, batch_size=128, learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, random_state=semilla, verbose=False)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_lda(X_train, y_train, X_test, y_test, balanceado=False, n_clases=8):
    """
    Entrena y evalúa un modelo LDA.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    balanceado : si True usa priors iguales, si False usa frecuencias reales
    n_clases   : número de clases (para calcular priors iguales)

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    priors = [1 / n_clases] * n_clases if balanceado else None
    modelo = LinearDiscriminantAnalysis(solver='svd', priors=priors)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_knn(X_train, y_train, X_test, y_test, pesos='uniform'):
    """
    Entrena y evalúa un modelo KNN.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba
    pesos   : 'uniform' (voto igual) o 'distance' (peso por cercanía)

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = KNeighborsClassifier(n_neighbors=5, weights=pesos, n_jobs=-1)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_centroide(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un Clasificador de Centroide Más Cercano.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = NearestCentroid(metric='euclidean')
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_gaussian_nb(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo Gaussian Naive Bayes.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = GaussianNB()
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_complement_nb(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo Complement Naive Bayes.
    Aplica MinMaxScaler internamente porque ComplementNB requiere
    valores no negativos. El scaler se ajusta solo sobre entrenamiento
    para evitar fuga de información hacia los datos de prueba.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    scaler = MinMaxScaler()
    X_train_pos = scaler.fit_transform(X_train)
    X_test_pos = scaler.transform(X_test)
    modelo = ComplementNB()
    inicio = time.time()
    modelo.fit(X_train_pos, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test_pos)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_arbol(X_train, y_train, X_test, y_test, semilla=42, balanceado=False):
    """
    Entrena y evalúa un Árbol de Decisión con una semilla específica.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    semilla    : semilla para Monte Carlo
    balanceado : si True usa class_weight='balanced'

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = DecisionTreeClassifier(criterion='gini', max_depth=10, class_weight='balanced' if balanceado else None, random_state=semilla)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_rf(X_train, y_train, X_test, y_test, semilla=42, balanceado=False, n_arboles=100):
    """
    Entrena y evalúa un modelo Random Forest con una semilla específica.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    semilla    : semilla para Monte Carlo
    balanceado : si True usa class_weight='balanced'
    n_arboles  : número de árboles en el bosque

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = RandomForestClassifier(n_estimators=n_arboles, criterion='gini', max_depth=15, class_weight='balanced' if balanceado else None, random_state=semilla, n_jobs=-1)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_gb(X_train, y_train, X_test, y_test, semilla=42, balanceado=False):
    """
    Entrena y evalúa un modelo HistGradientBoostingClassifier.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    semilla    : semilla para Monte Carlo
    balanceado : si True usa class_weight='balanced'

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, class_weight='balanced' if balanceado else None, random_state=semilla)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_qda(X_train, y_train, X_test, y_test, balanceado=False, n_clases=8):
    """
    Entrena y evalúa un modelo QDA.

    Parámetros:
    -----------
    X_train    : datos de entrenamiento reducidos
    y_train    : etiquetas de entrenamiento
    X_test     : datos de prueba reducidos
    y_test     : etiquetas de prueba
    balanceado : si True usa priors iguales entre clases
    n_clases   : número de clases para calcular priors iguales

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    priors = [1 / n_clases] * n_clases if balanceado else None
    modelo = QuadraticDiscriminantAnalysis(reg_param=0.1, priors=priors)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_adaboost(X_train, y_train, X_test, y_test, semilla=42):
    """
    Entrena y evalúa un modelo AdaBoost con árbol de decisión como
    estimador base.

    Parámetros:
    -----------
    X_train : datos de entrenamiento reducidos
    y_train : etiquetas de entrenamiento
    X_test  : datos de prueba reducidos
    y_test  : etiquetas de prueba
    semilla : semilla para Monte Carlo

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    estimador_base = DecisionTreeClassifier(max_depth=1)
    modelo = AdaBoostClassifier(estimator=estimador_base, n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=semilla)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def crear_generadores(preprocesamiento=None):
    """
    Crea generadores de entrenamiento y validación.

    Parámetros:
    -----------
    preprocesamiento : función de preprocesamiento del modelo
                       si None usa rescale=1./255

    Retorna:
    --------
    train_generator, val_generator
    """
    if preprocesamiento is not None:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocesamiento, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
        val_datagen = ImageDataGenerator(preprocessing_function=preprocesamiento)
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
    val_generator = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    return (train_generator, val_generator)

def crear_generadores_cnn():
    """
    Crea generadores específicos para CNN con data augmentation extendido.
    El augmentation incluye cambios de brillo para simular variaciones en
    las condiciones de iluminación de los exámenes oculares.
    Solo se aplica augmentation al entrenamiento, nunca a validación.
    """
    tren_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.2, shear_range=0.1, brightness_range=[0.8, 1.2], fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    generador_tren = tren_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
    generador_val = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    return (generador_tren, generador_val)

def crear_generadores_effnet():
    tren_datagen = ImageDataGenerator(preprocessing_function=effnet_preprocess, brightness_range=[0.6, 1.4], channel_shift_range=20.0, zoom_range=0.25, rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=effnet_preprocess)
    generador_tren = tren_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE_EFF, batch_size=BATCH_SIZE_EFF, class_mode='categorical', shuffle=True, seed=SEED)
    generador_val = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE_EFF, batch_size=BATCH_SIZE_EFF, class_mode='categorical', shuffle=False)
    return (generador_tren, generador_val)

def entrenar_effnet_una_vez(generador_tren, generador_val, n_clases, semilla=42, pesos_dict=None):
    print(f'\n   🏗️  Construyendo modelo (semilla {semilla})...')
    modelo, base = construir_effnet(n_clases, semilla)
    callbacks_fase = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-07, verbose=0)]
    print(f'   📌 Fase 1: entrenando capas de clasificación (base congelada)...')
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(generador_tren, epochs=5, validation_data=generador_val, class_weight=pesos_dict, verbose=0)
    print(f'   🔓 Fase 2: fine-tuning completo (lr=1e-5)...')
    base.trainable = True
    modelo.compile(optimizer=Adam(learning_rate=1e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    inicio = time.time()
    historial = modelo.fit(generador_tren, epochs=30, validation_data=generador_val, class_weight=pesos_dict, callbacks=callbacks_fase, verbose=0)
    duracion = time.time() - inicio
    _, acc = modelo.evaluate(generador_val, verbose=0)
    return {'modelo': modelo, 'accuracy': acc, 'historial': historial, 'tiempo': duracion}

def entrenar_logreg_pixeles(X_train, y_train, X_test, y_test, balanceado=False):
    """
    Entrena y evalúa Regresión Logística sobre píxeles directos.

    Parámetros:
    -----------
    X_train    : array de imágenes aplanadas de entrenamiento
    y_train    : etiquetas de entrenamiento
    X_test     : array de imágenes aplanadas de prueba
    y_test     : etiquetas de prueba
    balanceado : si True usa class_weight='balanced'

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo y predicciones
    """
    modelo = LogisticRegression(class_weight='balanced' if balanceado else None, max_iter=1000, random_state=SEED, solver='lbfgs', n_jobs=-1)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred}

def entrenar_svm_rbf(X_train, y_train, X_test, y_test, balanceado=False):
    """
    Entrena y evalúa SVM-RBF sobre píxeles directos.

    Parámetros:
    -----------
    X_train    : array de imágenes aplanadas de entrenamiento
    y_train    : etiquetas de entrenamiento
    X_test     : array de imágenes aplanadas de prueba
    y_test     : etiquetas de prueba
    balanceado : si True usa class_weight='balanced'

    Retorna:
    --------
    diccionario con modelo, accuracy, tiempo, predicciones y probabilidades
    """
    modelo = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced' if balanceado else None, cache_size=2000, probability=True, random_state=SEED)
    print('⏳ Entrenando SVM-RBF (puede tardar varias horas)...')
    inicio = time.time()
    modelo.fit(X_train, y_train)
    duracion = time.time() - inicio
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'modelo': modelo, 'accuracy': acc, 'tiempo': duracion, 'y_pred': y_pred, 'y_prob': y_prob}

def crear_generadores_resnet():
    """
    Crea generadores específicos para ResNet50 con preprocess_input
    propio de ResNet y data augmentation extendido.
    """
    tren_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
    generador_tren = tren_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
    generador_val = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    return (generador_tren, generador_val)

def entrenar_resnet_una_vez(generador_tren, generador_val, n_clases, semilla=42, pesos_dict=None):
    """
    Entrena ResNet50 en dos fases con una semilla específica.

    Fase 1: base congelada, lr=1e-4, 5 épocas
    Fase 2: fine-tuning completo, lr=1e-5, hasta EarlyStopping

    Parámetros:
    -----------
    generador_tren : generador de entrenamiento
    generador_val  : generador de validación
    n_clases       : número de clases
    semilla        : semilla para Monte Carlo
    pesos_dict     : pesos de clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, historial y tiempo
    """
    print(f'\n   🏗️  Construyendo modelo (semilla {semilla})...')
    modelo, base = construir_resnet(n_clases, semilla)
    callbacks_fase = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-07, verbose=0)]
    print(f'   📌 Fase 1: entrenando capas de clasificación (base congelada)...')
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(generador_tren, epochs=5, validation_data=generador_val, class_weight=pesos_dict, verbose=0)
    print(f'   🔓 Fase 2: fine-tuning completo (lr=1e-5)...')
    base.trainable = True
    modelo.compile(optimizer=Adam(learning_rate=1e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    inicio = time.time()
    historial = modelo.fit(generador_tren, epochs=30, validation_data=generador_val, class_weight=pesos_dict, callbacks=callbacks_fase, verbose=0)
    duracion = time.time() - inicio
    _, acc = modelo.evaluate(generador_val, verbose=0)
    return {'modelo': modelo, 'accuracy': acc, 'historial': historial, 'tiempo': duracion}

def crear_generadores_densenet():
    """
    Crea generadores específicos para DenseNet121 con preprocess_input
    propio de DenseNet y data augmentation extendido.
    """
    tren_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)
    generador_tren = tren_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
    generador_val = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    return (generador_tren, generador_val)

def entrenar_densenet_una_vez(generador_tren, generador_val, n_clases, semilla=42, pesos_dict=None):
    """
    Entrena DenseNet121 en dos fases con una semilla específica.

    Fase 1: base congelada, lr=1e-4, 5 épocas
    Fase 2: fine-tuning completo, lr=1e-5, hasta EarlyStopping

    Parámetros:
    -----------
    generador_tren : generador de entrenamiento
    generador_val  : generador de validación
    n_clases       : número de clases
    semilla        : semilla para Monte Carlo
    pesos_dict     : pesos de clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, historial y tiempo
    """
    print(f'\n   🏗️  Construyendo modelo (semilla {semilla})...')
    modelo, base = construir_densenet(n_clases, semilla)
    callbacks_fase = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-07, verbose=0)]
    print(f'   📌 Fase 1: entrenando capas de clasificación (base congelada)...')
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(generador_tren, epochs=5, validation_data=generador_val, class_weight=pesos_dict, verbose=0)
    print(f'   🔓 Fase 2: fine-tuning completo (lr=1e-5)...')
    base.trainable = True
    modelo.compile(optimizer=Adam(learning_rate=1e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    inicio = time.time()
    historial = modelo.fit(generador_tren, epochs=30, validation_data=generador_val, class_weight=pesos_dict, callbacks=callbacks_fase, verbose=0)
    duracion = time.time() - inicio
    _, acc = modelo.evaluate(generador_val, verbose=0)
    return {'modelo': modelo, 'accuracy': acc, 'historial': historial, 'tiempo': duracion}

def crear_generadores_inception():
    """
    Crea generadores específicos para InceptionV3 con preprocess_input
    propio de Inception y data augmentation extendido.
    Usa tamaño 299×299 nativo de InceptionV3.
    """
    tren_datagen = ImageDataGenerator(preprocessing_function=inception_preprocess, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=inception_preprocess)
    generador_tren = tren_datagen.flow_from_dataframe(dataframe=train_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE_INC, batch_size=BATCH_SIZE_INC, class_mode='categorical', shuffle=True, seed=SEED)
    generador_val = val_datagen.flow_from_dataframe(dataframe=test_df, directory=FOLDER_PATH, x_col='filename', y_col='labels', target_size=IMG_SHAPE_INC, batch_size=BATCH_SIZE_INC, class_mode='categorical', shuffle=False)
    return (generador_tren, generador_val)

def entrenar_inception_una_vez(generador_tren, generador_val, n_clases, semilla=42, pesos_dict=None):
    """
    Entrena InceptionV3 en dos fases con una semilla específica.

    Fase 1: base congelada, lr=1e-4, 5 épocas
    Fase 2: fine-tuning completo, lr=1e-5, hasta EarlyStopping

    Parámetros:
    -----------
    generador_tren : generador de entrenamiento
    generador_val  : generador de validación
    n_clases       : número de clases
    semilla        : semilla para Monte Carlo
    pesos_dict     : pesos de clase o None

    Retorna:
    --------
    diccionario con modelo, accuracy, historial y tiempo
    """
    print(f'\n   🏗️  Construyendo modelo (semilla {semilla})...')
    modelo, base = construir_inception(n_clases, semilla)
    callbacks_fase = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-07, verbose=0)]
    print(f'   📌 Fase 1: entrenando capas de clasificación (base congelada)...')
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(generador_tren, epochs=5, validation_data=generador_val, class_weight=pesos_dict, verbose=0)
    print(f'   🔓 Fase 2: fine-tuning completo (lr=1e-5)...')
    base.trainable = True
    modelo.compile(optimizer=Adam(learning_rate=1e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    inicio = time.time()
    historial = modelo.fit(generador_tren, epochs=30, validation_data=generador_val, class_weight=pesos_dict, callbacks=callbacks_fase, verbose=0)
    duracion = time.time() - inicio
    _, acc = modelo.evaluate(generador_val, verbose=0)
    return {'modelo': modelo, 'accuracy': acc, 'historial': historial, 'tiempo': duracion}


def construir_cnn(n_clases):
    """
    Construye la arquitectura CNN con 5 bloques convolucionales.
    Usa GlobalAveragePooling2D en lugar de Flatten para reducir
    parámetros y mejorar la generalización.

    Parámetros:
    -----------
    n_clases : número de clases a clasificar (8 enfermedades)

    Retorna:
    --------
    modelo Keras sin compilar
    """
    modelo = Sequential([
        # Bloque 1 — detección de bordes y texturas simples
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Bloque 5 — patrones complejos específicos por enfermedad
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Clasificación — GlobalAveragePooling reduce sobreajuste vs Flatten
        GlobalAveragePooling2D(),
        Dense(512, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.5),
        Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(n_clases, activation='softmax')
    ])

    return modelo


def construir_cnn1d(longitud_entrada, num_clases):
    """
    Construye una CNN 1D para clasificación de representaciones reducidas.
    Trata el vector de componentes como una secuencia y aplica filtros
    convolucionales para detectar patrones locales.

    Parámetros:
    -----------
    longitud_entrada : número de componentes/factores (30 en este caso)
    num_clases       : número de clases a clasificar (8)

    Retorna:
    --------
    modelo de Keras compilable
    """
    # Entrada: vector de componentes con canal adicional para Conv1D
    entrada = layers.Input(shape=(longitud_entrada, 1))

    # ── Bloque convolucional 1: patrones simples ───────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(entrada)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # ── Bloque convolucional 2: patrones más complejos ─────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # ── Bloque convolucional 3: patrones de alto nivel ─────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Reduce el tensor a un vector promediando por canal
    x = layers.GlobalAveragePooling1D()(x)

    # ── Capa densa con regularización ─────────────────────────────────────
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    # ── Capa de salida ─────────────────────────────────────────────────────
    salida = layers.Dense(num_clases, activation='softmax')(x)

    return models.Model(inputs=entrada, outputs=salida)


def construir_densenet(n_clases, semilla=42):
    """
    Construye DenseNet121 con capas de clasificación.
    Retorna el modelo con la base congelada listo para la Fase 1.

    Parámetros:
    -----------
    n_clases : número de clases (8 enfermedades)
    semilla  : semilla para reproducibilidad

    Retorna:
    --------
    modelo Keras, base DenseNet121
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    base = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)
    )
    base.trainable = False  # Fase 1: base congelada

    modelo = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation='softmax')
    ])

    return modelo, base


def construir_effnet(n_clases, semilla=42):
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    base = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SHAPE_EFF[0], IMG_SHAPE_EFF[1], 3)
    )
    base.trainable = False

    modelo = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(n_clases, activation='softmax')
    ])

    return modelo, base


def construir_inception(n_clases, semilla=42):
    """
    Construye InceptionV3 con capas de clasificación.
    Retorna el modelo con la base congelada listo para la Fase 1.

    Parámetros:
    -----------
    n_clases : número de clases (8 enfermedades)
    semilla  : semilla para reproducibilidad

    Retorna:
    --------
    modelo Keras, base InceptionV3
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    base = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SHAPE_INC[0], IMG_SHAPE_INC[1], 3)
    )
    base.trainable = False  # Fase 1: base congelada

    modelo = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation='softmax')
    ])

    return modelo, base


def construir_mlp(n_caracteristicas, n_clases):
    """
    Construye un MLP profundo con activación GELU e inicialización HeNormal.

    Parámetros:
    -----------
    n_caracteristicas : número de componentes/factores de entrada
    n_clases          : número de clases a clasificar (8)

    Retorna:
    --------
    modelo de Keras compilado
    """
    modelo = models.Sequential([
        layers.Input(shape=(n_caracteristicas,)),

        # ── Capa 1 ────────────────────────────────────────────────────────
        layers.Dense(256, activation='gelu',
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ── Capa 2 ────────────────────────────────────────────────────────
        layers.Dense(128, activation='gelu',
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # ── Capa 3 ────────────────────────────────────────────────────────
        layers.Dense(64, activation='gelu',
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # ── Capa 4 ────────────────────────────────────────────────────────
        layers.Dense(32, activation='gelu',
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),

        # ── Capa de salida ────────────────────────────────────────────────
        layers.Dense(n_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


def construir_mlp_pixeles(n_features, n_clases):
    """
    Construye MLP para clasificación sobre píxeles directos.
    Arquitectura: 4 capas densas con BatchNormalization y Dropout progresivo.

    Parámetros:
    -----------
    n_features : número de features de entrada (49,152 para 128×128×3)
    n_clases   : número de clases (8 enfermedades)

    Retorna:
    --------
    modelo Keras sin compilar
    """
    modelo = models.Sequential([
        layers.InputLayer(input_shape=(n_features,)),

        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),

        layers.Dense(n_clases, activation='softmax')
    ])

    return modelo


def construir_resnet(n_clases, semilla=42):
    """
    Construye ResNet50 con capas de clasificación.
    Retorna el modelo con la base congelada listo para la Fase 1.

    Parámetros:
    -----------
    n_clases : número de clases (8 enfermedades)
    semilla  : semilla para reproducibilidad

    Retorna:
    --------
    modelo Keras, base ResNet50
    """
    tf.keras.backend.clear_session()
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)
    )
    base.trainable = False  # Fase 1: base congelada

    modelo = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation='softmax')
    ])

    return modelo, base


def construir_transformer(n_caracteristicas, n_clases,
                           dim_proyeccion=128, num_cabezas=4,
                           dim_clave=32, dim_ffn=256,
                           n_bloques=2, tasa_dropout=0.1):
    """
    Construye un Transformer tabular para clasificación de representaciones
    reducidas. Trata cada componente como un token de una secuencia.

    Parámetros:
    -----------
    n_caracteristicas : número de componentes/factores de entrada (30)
    n_clases          : número de clases a clasificar (8)
    dim_proyeccion    : dimensión de la proyección inicial de cada token
    num_cabezas       : número de cabezas de atención multi-cabeza
    dim_clave         : dimensión de cada cabeza de atención
    dim_ffn           : dimensión interna de la red feed-forward
    n_bloques         : número de bloques Transformer apilados
    tasa_dropout      : tasa de dropout para regularización

    Retorna:
    --------
    modelo de Keras sin compilar
    """
    entrada = layers.Input(shape=(n_caracteristicas,))

    # ── Proyección inicial ─────────────────────────────────────────────────
    # Cada componente se trata como un token. Se expande a una secuencia
    # y se proyecta a una dimensión mayor para enriquecer la representación.
    x = layers.Reshape((n_caracteristicas, 1))(entrada)
    x = layers.Dense(dim_proyeccion,
                     kernel_regularizer=regularizers.l2(1e-4))(x)

    # ── Bloques Transformer apilados ──────────────────────────────────────
    # Múltiples bloques permiten capturar relaciones de mayor complejidad
    # entre los componentes en capas sucesivas de abstracción.
    for _ in range(n_bloques):
        x = bloque_transformer(
            x,
            num_cabezas=num_cabezas,
            dim_clave=dim_clave,
            dim_ffn=dim_ffn,
            tasa_dropout=tasa_dropout
        )

    # ── Agregación y clasificación ─────────────────────────────────────────
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    salida = layers.Dense(n_clases, activation='softmax')(x)

    return models.Model(entrada, salida)


def construir_vit(n_clases):
    modelo = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        hub.KerasLayer(VIT_URL, trainable=False),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation='softmax')
    ])
    return modelo


def bloque_transformer(x, num_cabezas, dim_clave, dim_ffn, tasa_dropout=0.1):
    """
    Parámetros:
    -----------
    x            : tensor de entrada
    num_cabezas  : número de cabezas de atención
    dim_clave    : dimensión de cada cabeza de atención
    dim_ffn      : dimensión de la red feed-forward interna
    tasa_dropout : tasa de dropout para regularización

    Retorna:
    --------
    tensor de salida con la misma forma que la entrada
    """
    # ── Atención multi-cabeza con conexión residual ────────────────────────
    atencion = layers.MultiHeadAttention(
        num_heads=num_cabezas,
        key_dim=dim_clave,
        dropout=tasa_dropout
    )(x, x)
    atencion = layers.Dropout(tasa_dropout)(atencion)
    x = layers.Add()([x, atencion])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # ── Red feed-forward con conexión residual ─────────────────────────────
    ffn = layers.Dense(dim_ffn, activation='relu',
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    ffn = layers.Dropout(tasa_dropout)(ffn)
    ffn = layers.Dense(x.shape[-1],
                       kernel_regularizer=regularizers.l2(1e-4))(ffn)
    ffn = layers.Dropout(tasa_dropout)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x

