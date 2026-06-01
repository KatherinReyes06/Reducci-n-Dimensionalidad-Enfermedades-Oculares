# Comparación de Métodos de Reducción de Dimensionalidad en la Predicción de Enfermedades Oculares mediante Redes Neuronales Convolucionales

[![Abrir en nbviewer](https://img.shields.io/badge/Ver%20notebook-nbviewer-orange)](https://nbviewer.org/github/KatherinReyes06/Reducci-n-Dimensionalidad-Enfermedades-Oculares/blob/main/clasificacion_enfermedades_oculares_ODIR5K.ipynb)

Trabajo de grado — Escuela de Matemáticas, Universidad Industrial de Santander  
Autora: Katherin Liceth Reyes Enciso  
Director: PhD. Andrés Sebastián Ríos Gutiérrez

## Descripción

Este repositorio contiene el código fuente desarrollado para el trabajo de grado que evalúa y compara el impacto de distintos métodos de reducción de dimensionalidad (PCA, Análisis Factorial y UMAP) en el desempeño de modelos de clasificación de enfermedades oculares a partir de imágenes de retina del conjunto de datos ODIR-5K.

## Requisitos de software

El código fue desarrollado y ejecutado en **Google Colaboratory**, por lo que no requiere instalación local. Se recomienda ejecutarlo en el mismo entorno para garantizar compatibilidad.

Las principales librerías utilizadas son:

- TensorFlow / Keras
- scikit-learn
- UMAP (`umap-learn`)
- factor-analyzer
- OpenCV (`cv2`)
- NumPy, Pandas, Matplotlib, Seaborn
- imbalanced-learn (`imblearn`)
- CuPy (`cupy-cuda12x`)
- tqdm

## Versiones de librerías utilizadas

| Librería | Versión |
|---|---|
| TensorFlow / Keras | 2.20.0 |
| scikit-learn | 1.6.1 |
| CuPy | 14.0.1 |
| UMAP | 0.5.12 |
| factor-analyzer | 0.5.1 |
| OpenCV | 4.13.0.92 |
| NumPy | 2.0.2 |
| Pandas | 2.2.2 |

## Requisitos de hardware

- **GPU obligatoria**: el código utiliza CuPy y operaciones de aprendizaje profundo que requieren GPU.
- En Google Colab: activar GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4 o superior)*.
- RAM recomendada: mínimo 12 GB (disponible en Colab estándar).

## Nota importante sobre recursos computacionales

Los experimentos con modelos entrenados directamente sobre imágenes completas 
(CNN, ViT, EfficientNetB3, ResNet50, DenseNet121, InceptionV3) requieren 
recursos computacionales significativamente mayores. Para estos experimentos 
se utilizó **Google Colab Pro** con entorno de ejecución **GPU T4** y mayor 
capacidad de RAM.

Se recomienda:

- **Google Colab Pro** con GPU T4 para modelos con imágenes completas.
- **Google Colab gratuito** con GPU T4 es suficiente para los experimentos 
  con reducción de dimensionalidad (PCA, AF, UMAP).

## Datos

El conjunto de datos utilizado es **ODIR-5K** (Ocular Disease Intelligent Recognition), disponible en:  
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

### Opción A: Descargar y subir a Google Drive
Descarga el dataset desde Kaggle y súbelo a tu Google Drive. Luego monta el Drive en Colab y ajusta la ruta en el notebook.

### Opción B: Conectar Kaggle directamente desde Colab (Recomendada)
En Google Colab, haz clic en el ícono de base de datos en el panel izquierdo, busca "ODIR" en el buscador y selecciona el dataset. Colab generará automáticamente el código de importación:

```python
import kagglehub
path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
```

Esta opción no requiere subir los datos a tu Google Drive personal.

## Estructura del repositorio
El repositorio se ha organizado de forma modular para facilitar el mantenimiento y la escalabilidad:

```text
Reducci-n-Dimensionalidad-Enfermedades-Oculares/
├── README.md
├── main.py
├── clasificacion_enfermedades_oculares_ODIR5K.ipynb  # Notebook original con resultados
└── src/
    ├── __init__.py
    ├── data_loader.py               # Funciones para carga y preprocesamiento de imágenes
    ├── dimensionality_reduction.py  # Algoritmos de PCA, UMAP, AF y estadísticos
    ├── visualization.py             # Gráficas y reportes visuales
    ├── models.py                    # Lógica de entrenamiento de modelos
    ├── evaluation.py                # Métricas, matrices de confusión y reportes
    ├── experiments.py               # Configuración y ejecución de experimentos
    └── utils.py                     # Funciones auxiliares
```


## Forma de ejecución recomendada

### ✅ Opción 1: Notebook en Google Colab (Recomendada)

Esta es la forma en que fue desarrollado y ejecutado el proyecto originalmente. Se recomienda usar esta opción para garantizar compatibilidad y acceso a GPU.

1. Abre Google Colab: https://colab.research.google.com
2. Sube o abre el notebook `notebooks/clasificacion_enfermedades_oculares_ODIR5K.ipynb`.
3. Activa el entorno de GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
4. Ejecuta las celdas de configuración del entorno en el orden indicado.
5. Verifica que el dataset ODIR-5K esté en tu Google Drive en la ruta indicada en el notebook.
6. Ejecuta las celdas en orden secuencial.

### ⚠️ Opción 2: Ejecución local (No recomendada)

Es posible ejecutar el código de forma local usando los módulos de `src/` y el archivo `main.py`. Sin embargo, en esta opción  pueden presentarse errores dependiendo del sistema operativo, la versión de Python y la disponibilidad de GPU en el equipo.

Requisitos mínimos para ejecución local:
- Python 3.10+
- GPU compatible con CUDA (obligatoria para CuPy y modelos de aprendizaje profundo)
- Instalación manual de todas las librerías listadas en la sección de requisitos


## Configuración del entorno en Google Colab

Ejecuta estas celdas al inicio antes de correr el código:

**1. Verificar versión de CUDA:**
```python
!nvcc --version
```

**2. Configurar compatibilidad con Keras 2:**
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

**3. Instalar librerías adicionales:**
```python
!pip install cupy-cuda12x factor-analyzer
```

**4. Montar Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```
Colab te pedirá autorización para acceder a tu Drive. Acepta y continúa.

**5. Verificar GPU:**
```python
import cupy as cp
print(f"Versión de CuPy: {cp.__version__}")
print(f"GPUs disponibles: {cp.cuda.runtime.getDeviceCount()}")
```



## Contacto

Katherin Liceth Reyes Enciso  
Universidad Industrial de Santander  
Escuela de Matemáticas
