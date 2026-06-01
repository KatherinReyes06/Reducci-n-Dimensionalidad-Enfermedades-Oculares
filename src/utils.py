import os
import pandas as pd
import pickle
from datetime import datetime


def agregar_fila(modelo, condicion, resultados, filas, n_componentes, n_factores, n_componentes_umap, col_pca, col_af, col_umap):
    acc_pca  = resultados.get(f"PCA  ({n_componentes} comp.)", {}).get('accuracy', 0)
    acc_af   = resultados.get(f"AF   ({n_factores} fact.)", {}).get('accuracy', 0)
    acc_umap = resultados.get(f"UMAP ({n_componentes_umap} comp.)", {}).get('accuracy', 0)
    filas.append({
        'Modelo'    : modelo,
        'Condición' : condicion,
        col_pca     : acc_pca,
        col_af      : acc_af,
        col_umap    : acc_umap,
    })


def agregar_fila_global(modelo, condicion, accuracy, filas):
    """Agrega una fila a la tabla comparativa global."""
    filas.append({
        'Modelo'    : modelo,
        'Condición' : condicion,
        'Accuracy'  : accuracy
    })


def guardar_checkpoint(nuevo_resultado, checkpoint_path="./checkpoints/resultados.pkl"):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"📂 Checkpoint existente cargado con {len(checkpoint)} entradas")
    else:
        checkpoint = {}
        print("📂 Creando checkpoint nuevo...")

    checkpoint.update(nuevo_resultado)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"✅ Checkpoint actualizado — entradas: {list(checkpoint.keys())}")


def cargar_checkpoint(checkpoint_path="./checkpoints/resultados.pkl"):
    if not os.path.exists(checkpoint_path):
        print("⚠️ No existe checkpoint guardado")
        return {}

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"✅ Checkpoint cargado con {len(checkpoint)} entradas:")
    for clave, valor in checkpoint.items():
        if isinstance(valor, float):
            print(f"   {clave:30s} : {valor:.4f}")
        else:
            print(f"   {clave:30s} : array {type(valor).__name__}")

    return checkpoint


def mostrar_pendientes(checkpoint):
    todos = [
        'acc_cnn_sin',  'acc_cnn_cw',
        'acc_vit_sin',  'acc_vit_cw',
        'acc_eff_sin',  'acc_eff_cw',
        'acc_rn_sin',   'acc_rn_cw',
        'acc_dn_sin',   'acc_dn_cw',
        'acc_inc_sin',  'acc_inc_cw',
        'acc_mlp_sin',  'acc_mlp_cw',
        'acc_lr_sin',   'acc_lr_cw',
        'acc_svm_sin',  'acc_svm_cw'
    ]

    print("\n" + "═"*50)
    print("  ESTADO DE MODELOS")
    print("═"*50)
    for clave in todos:
        estado = "✅" if clave in checkpoint else "⏳"
        print(f"  {estado} {clave}")
    print("═"*50)
    completados = sum(1 for c in todos if c in checkpoint)
    print(f"  Completados: {completados}/{len(todos)}")
    print("═"*50)

