import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Custom modules
from src.data_loader import load_all_images, cargar_imagenes_como_array
from src.dimensionality_reduction import horn_parallel_analysis, criterio_varianza, calcular_kmo
from src.models import *
from src.experiments import *
from src.utils import guardar_checkpoint, cargar_checkpoint, agregar_fila_global, mostrar_pendientes
from src.evaluation import *
from src.visualization import graficar_horn, graficar_varianza_acumulada, graficar_kaiser, graficar_kmo

def parse_args():
    parser = argparse.ArgumentParser(description="Orquestador del Pipeline de Clasificación de Enfermedades Oculares")
    parser.add_argument("--data-dir", type=str, default="./data/preprocessed_images", help="Ruta a las imágenes preprocesadas")
    parser.add_argument("--csv-path", type=str, default="./data/full_df.csv", help="Ruta al CSV con etiquetas")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/resultados.pkl", help="Ruta para guardar/cargar resultados intermedios")
    
    # Fases de ejecución
    parser.add_argument("--run-data-prep", action="store_true", help="Ejecutar carga y preparación de datos")
    parser.add_argument("--run-dim-reduction", action="store_true", help="Ejecutar análisis de reducción dimensional")
    parser.add_argument("--run-ml-classic", action="store_true", help="Entrenar modelos clásicos (SVM, RF, etc.)")
    parser.add_argument("--run-dl-custom", action="store_true", help="Entrenar modelos DL propios (CNN1D, MLP, Transformer)")
    parser.add_argument("--run-transfer-learning", action="store_true", help="Entrenar modelos pre-entrenados (ResNet, EfficientNet, etc.)")
    parser.add_argument("--run-all", action="store_true", help="Ejecutar todo el pipeline")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Activar todas las banderas si --run-all está presente
    if args.run-all:
        args.run_data_prep = True
        args.run_dim_reduction = True
        args.run_ml_classic = True
        args.run_dl_custom = True
        args.run_transfer_learning = True

    # Si no se provee ninguna bandera, imprimir ayuda y salir
    if not any([args.run_data_prep, args.run_dim_reduction, args.run_ml_classic, args.run_dl_custom, args.run_transfer_learning]):
        print("⚠️ No se ha seleccionado ninguna fase para ejecutar.")
        print("Ejecuta con --help para ver las opciones disponibles (ej: --run-all).")
        sys.exit(1)

    print("🚀 Iniciando Pipeline ODIR-5K...")
    
    # ---------------------------------------------------------
    # 1. Carga de Datos
    # ---------------------------------------------------------
    if args.run_data_prep or args.run_dim_reduction or args.run_ml_classic or args.run_dl_custom:
        print("\n" + "="*50)
        print(" 📂 FASE 1: CARGA DE DATOS")
        print("="*50)
        
        IMG_SIZE = (128, 128)
        all_images, image_filenames = load_all_images(args.data_dir, IMG_SIZE)
        
        if len(all_images) == 0:
            print(f"❌ No se encontraron imágenes en {args.data_dir}. Abortando.")
            sys.exit(1)
            
        images_array = np.array(all_images)
        n_samples, h, w, c = images_array.shape
        datos_aplanados = images_array.reshape((n_samples, h * w * c))
        
        try:
            df = pd.read_csv(args.csv_path)
            df["labels"] = df["labels"].str.replace(r"[\[\]' ]", "", regex=True)
            dicc_labels = dict(zip(df['filename'], df['labels']))
            
            labels_final = [dicc_labels[n] for n in image_filenames if n in dicc_labels]
            encoder = LabelEncoder()
            y = encoder.fit_transform(labels_final)
            
            # Split
            X_train_ap, X_test_ap, y_train, y_test = train_test_split(
                datos_aplanados, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_ap)
            X_test_scaled  = scaler.transform(X_test_ap)
            print("✅ Datos cargados y estandarizados con éxito.")
            
        except Exception as e:
            print(f"❌ Error al procesar CSV: {e}")
            sys.exit(1)

    # ---------------------------------------------------------
    # 2. Reducción de Dimensionalidad
    # ---------------------------------------------------------
    n_componentes = 239  # Valores hardcodeados como fallback
    n_factores = 239
    
    if args.run_dim_reduction:
        print("\n" + "="*50)
        print(" 📉 FASE 2: REDUCCIÓN DE DIMENSIONALIDAD")
        print("="*50)
        
        kmo_res = calcular_kmo(X_train_scaled, n_vars=500)
        graficar_kmo(kmo_res)
        
        # Horn Parallel Analysis
        res_pca_horn = horn_parallel_analysis(X_train_scaled, metodo='pca')
        graficar_horn(res_pca_horn, metodo='pca')
        
        # Varianza
        res_pca_var = criterio_varianza(X_train_scaled, metodo='pca')
        graficar_varianza_acumulada(res_pca_var, metodo='pca')
        
        n_componentes = res_pca_var[-1]['num_componentes']
        
        # PCA Transform
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_componentes, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
        
        # AF
        from sklearn.decomposition import FactorAnalysis
        n_factores = res_pca_var[-1]['num_componentes'] # Usualmente se alinea con PCA o requiere AF Horn
        af = FactorAnalysis(n_components=n_factores, random_state=42)
        X_train_af = af.fit_transform(X_train_scaled)
        X_test_af  = af.transform(X_test_scaled)
        
        print(f"✅ Reducción terminada: PCA={n_componentes}, AF={n_factores}")
        
    else:
        # Dummy si no se ejecuta
        if 'X_train_scaled' in locals():
            from sklearn.decomposition import PCA, FactorAnalysis
            pca = PCA(n_components=n_componentes, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca  = pca.transform(X_test_scaled)
            
            af = FactorAnalysis(n_components=n_factores, random_state=42)
            X_train_af = af.fit_transform(X_train_scaled)
            X_test_af  = af.transform(X_test_scaled)

    # ---------------------------------------------------------
    # 3. Modelos Clásicos (SVM, RF, etc.)
    # ---------------------------------------------------------
    if args.run_ml_classic:
        print("\n" + "="*50)
        print(" 🤖 FASE 3: MODELOS MACHINE LEARNING CLÁSICOS")
        print("="*50)
        
        # Ejemplo de ejecución: Random Forest con PCA
        buscar_mejor_rf(
            X_train_pca, y_train, X_test_pca, y_test,
            encoder=encoder, etiqueta="PCA", n_intentos=3, balanceado=False
        )
        # Aquí se añadirían el resto (SVM, AdaBoost, GB, LDA, etc.)
        # según el tiempo y los requerimientos computacionales.

    # ---------------------------------------------------------
    # 4. Deep Learning Custom (CNN1D, MLP, Transformer)
    # ---------------------------------------------------------
    if args.run_dl_custom:
        print("\n" + "="*50)
        print(" 🧠 FASE 4: DEEP LEARNING (CUSTOM)")
        print("="*50)
        
        # MLP
        buscar_mejor_mlp(
            X_train_pca, y_train, X_test_pca, y_test,
            encoder=encoder, etiqueta="PCA", n_intentos=3
        )
        
        # CNN1D
        buscar_mejor_cnn1d(
            X_train_pca, y_train, X_test_pca, y_test,
            encoder=encoder, etiqueta="PCA", n_intentos=3
        )

    # ---------------------------------------------------------
    # 5. Transfer Learning (Pre-entrenados)
    # ---------------------------------------------------------
    if args.run_transfer_learning:
        print("\n" + "="*50)
        print(" 🖼️ FASE 5: TRANSFER LEARNING (CNNs)")
        print("="*50)
        print("Nota: Requiere DataFrame de imágenes y configuración de generadores.")
        # Aquí iría crear_generadores_effnet(), entrenar_effnet_una_vez(), etc.
        # buscar_mejor_effnet(generador_tren, generador_val, n_clases)

    print("\n✅ Ejecución finalizada con éxito.")

if __name__ == "__main__":
    main()
