# Comparación de Métodos de Reducción de Dimensionalidad en la Predicción de Enfermedades Oculares mediante Redes Neuronales Convolucionales

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

Si intentas ejecutar estos experimentos en Google Colab gratuito, es probable 
que encuentres errores de memoria (OOM - Out of Memory). Se recomienda:

- **Google Colab Pro** con GPU T4 para modelos con imágenes completas.
- **Google Colab gratuito** con GPU T4 es suficiente para los experimentos 
  con reducción de dimensionalidad (PCA, AF, UMAP).

## Datos

El conjunto de datos utilizado es **ODIR-5K** (Ocular Disease Intelligent Recognition), disponible en:  
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Descarga el dataset y súbelo a tu Google Drive antes de ejecutar el notebook.

## Estructura del repositorio
El repositorio contiene actualmente el siguiente archivo:

- `clasificacion_enfermedades_oculares_ODIR5K.ipynb` — Notebook principal con todo el código del proyecto.


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

## Pasos de ejecución

1. Abre Google Colab: https://colab.research.google.com
2. Sube o abre el notebook del repositorio.
3. Activa el entorno de GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
4. Ejecuta las celdas de configuración del entorno en el orden indicado.
5. Verifica que el dataset ODIR-5K esté en tu Google Drive en la ruta indicada en el notebook.
6. Ejecuta las celdas en orden secuencial.

## Contacto

Katherin Liceth Reyes Enciso  
Universidad Industrial de Santander  
Escuela de Matemáticas
