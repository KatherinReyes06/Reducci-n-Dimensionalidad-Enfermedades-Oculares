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

def resize_with_padding(image, target_size):
    """
    Redimensiona una imagen manteniendo su proporción (aspect ratio)
    y agrega bordes negros (padding) para llegar al tamaño objetivo.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def load_all_images(folder, img_size):
    images_list = []
    filenames_list = []
    if not os.path.exists(folder):
        print(f"Error: La carpeta '{folder}' no fue encontrada.")
        return ([], [])
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('jpg', 'png', 'jpeg', 'JPG', 'PNG')):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path)
            if img is not None:
                img_processed = resize_with_padding(img, img_size)
                images_list.append(img_processed)
                filenames_list.append(filename)
            else:
                print(f'Advertencia: No se pudo cargar la imagen {filename}. Se omitirá.')
    return (images_list, filenames_list)


def cargar_imagenes_como_array(df, folder_path, img_shape=IMG_SHAPE_SK):
    """
    Carga imágenes desde disco y las convierte en array 2D aplanado
    listo para modelos sklearn.

    Parámetros:
    -----------
    df          : DataFrame con columnas 'filename' y 'labels'
    folder_path : ruta a la carpeta de imágenes
    img_shape   : tamaño de redimensionado (default: 128×128)

    Retorna:
    --------
    X : array (n_muestras, 49152) normalizado [0, 1]
    y : array de etiquetas codificadas
    """
    X, y  = [], []
    total = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):
        ruta = os.path.join(folder_path, fila['filename'])
        img  = cv2.imread(ruta)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, img_shape)
        img  = img / 255.0      # Normalizar a [0, 1]
        X.append(img.flatten()) # 128×128×3 → 49,152 features
        y.append(fila['labels'])

        if (i + 1) % 500 == 0:
            print(f"   Cargadas {i+1}/{total} imágenes...", end="\r")

    print(f"   ✅ {total} imágenes cargadas como array             ")
    return np.array(X, dtype=np.float32), np.array(y)

