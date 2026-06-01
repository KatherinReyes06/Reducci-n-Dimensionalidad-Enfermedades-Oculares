import time
import numpy as np
from src.models import *
from src.utils import *
from src.evaluation import *


def buscar_mejor_adaboost(X_train, y_train, X_test, y_test,
                           encoder, etiqueta="PCA",
                           n_intentos=10, color_mapa='Oranges'):
    """
    Búsqueda Monte Carlo del mejor AdaBoost variando la semilla.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  ADABOOST [SIN BALANCE] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_adaboost(
            X_train, y_train, X_test, y_test,
            semilla=semilla_actual
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte ────────────────────────────────────────────────────────────
    clases_nombres = encoder.classes_
    y_pred         = mejor_resultado['y_pred']

    print(f"\n--- Reporte de clasificación [{etiqueta}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # ── Matriz de confusión ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=clases_nombres,
        cmap=color_mapa,
        ax=ax,
        colorbar=True
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'AdaBoost [SIN BALANCE]\nDatos: {etiqueta}')
    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_arbol(X_train, y_train, X_test, y_test,
                       encoder, etiqueta="PCA",
                       n_intentos=10, balanceado=False,
                       condicion="SIN BALANCE", color_mapa='Oranges'):
    """
    Búsqueda Monte Carlo del mejor Árbol de Decisión variando la semilla.
    """
    print(f"\n{'='*70}")
    print(f"  ÁRBOL DE DECISIÓN [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_arbol(
            X_train, y_train, X_test, y_test,
            semilla=semilla_actual, balanceado=balanceado
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte ────────────────────────────────────────────────────────────
    clases_nombres = encoder.classes_
    y_pred         = mejor_resultado['y_pred']

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # ── Matriz de confusión ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=clases_nombres,
        cmap=color_mapa,
        ax=ax,
        colorbar=True
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'Árbol de Decisión [{condicion}]\nDatos: {etiqueta}')
    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_cnn1d(X_train, y_train, X_test, y_test,
                        encoder, etiqueta="PCA",
                        n_intentos=20, pesos_clases=None,
                        condicion="SIN BALANCE", color_mapa='Blues'):
    """
    Búsqueda Monte Carlo del mejor modelo CNN 1D variando la semilla
    de inicialización. Reporta el modelo con mayor accuracy.

    Parámetros:
    -----------
    X_train      : datos de entrenamiento
    y_train      : etiquetas de entrenamiento
    X_test       : datos de prueba
    y_test       : etiquetas de prueba
    encoder      : LabelEncoder con nombres de clases
    etiqueta     : nombre del dataset para los títulos
    n_intentos   : número de semillas a probar
    pesos_clases : diccionario de pesos o None
    condicion    : descripción de la condición de balanceo
    color_mapa   : color para la matriz de confusión

    Retorna:
    --------
    mejor modelo encontrado
    """
    print(f"\n{'='*70}")
    print(f"  CNN 1D [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc      = 0.0
    mejor_resultado = None
    inicio_global  = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_cnn1d(
            X_train, y_train, X_test, y_test,
            encoder, semilla=semilla_actual,
            pesos_clases=pesos_clases
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")
            del res

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte y visualización del mejor modelo ───────────────────────────
    historial      = mejor_resultado['historial']
    y_pred         = mejor_resultado['y_pred']
    clases_nombres = encoder.classes_

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # Gráficas: pérdida, accuracy y matriz de confusión
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pérdida
    axes[0].plot(historial.history['loss'],     label='Entrenamiento')
    axes[0].plot(historial.history['val_loss'], label='Validación')
    axes[0].set_title(f'Pérdida — {etiqueta}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(historial.history['accuracy'],     label='Entrenamiento')
    axes[1].plot(historial.history['val_accuracy'], label='Validación')
    axes[1].set_title(f'Exactitud — {etiqueta}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Exactitud')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=color_mapa,
        xticklabels=clases_nombres,
        yticklabels=clases_nombres,
        ax=axes[2]
    )
    axes[2].set_title(f'Matriz de Confusión\n{etiqueta} [{condicion}]')
    axes[2].set_xlabel('Predicción')
    axes[2].set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_densenet(generador_tren, generador_val,
                           n_clases, condicion="SIN BALANCE",
                           pesos_dict=None, n_intentos=3):
    """
    Búsqueda Monte Carlo del mejor DenseNet121 variando la semilla.
    """
    print(f"\n{'='*70}")
    print(f"  DENSENET121 [{condicion}]")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        print(f"\n🔄 Intento {i+1}/{n_intentos}...")

        try:
            res = entrenar_densenet_una_vez(
                generador_tren, generador_val,
                n_clases, semilla=semilla_actual,
                pesos_dict=pesos_dict
            )
            print(f"   ✅ Accuracy: {res['accuracy']:.4f} | "
                  f"Tiempo: {res['tiempo']/60:.1f} min", end="")

            if res['accuracy'] > mejor_acc:
                mejor_acc       = res['accuracy']
                mejor_resultado = res
                print("  🌟 ¡Nuevo récord!")
            else:
                print("")
                del res

        except Exception as e:
            print(f"\n   ❌ Error en intento {i+1}: {e}")

    tiempo_total = (time.time() - inicio_global) / 60
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f} minutos")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Curvas de aprendizaje ──────────────────────────────────────────────
    historial = mejor_resultado['historial']
    epocas    = range(len(historial.history['accuracy']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'DenseNet121 [{condicion}] — Mejor intento', fontsize=14)

    ax1.plot(epocas, historial.history['accuracy'],     label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epocas, historial.history['loss'],     label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Reporte y matriz ───────────────────────────────────────────────────
    generador_val.reset()
    predicciones   = mejor_resultado['modelo'].predict(
        generador_val, verbose=1
    )
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    acc_final = accuracy_score(y_real, y_pred)
    print(f"\n--- Reporte de clasificación [{condicion}] ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — DenseNet121 [{condicion}]')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return mejor_resultado, acc_final


def buscar_mejor_effnet(generador_tren, generador_val,
                         n_clases, condicion="SIN BALANCE",
                         pesos_dict=None, n_intentos=3):
    print(f"\n{'='*70}")
    print(f"  EFFICIENTNETB3 [{condicion}]")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        print(f"\n🔄 Intento {i+1}/{n_intentos}...")

        try:
            res = entrenar_effnet_una_vez(
                generador_tren, generador_val,
                n_clases, semilla=semilla_actual,
                pesos_dict=pesos_dict
            )
            print(f"   ✅ Accuracy: {res['accuracy']:.4f} | "
                  f"Tiempo: {res['tiempo']/60:.1f} min", end="")

            if res['accuracy'] > mejor_acc:
                mejor_acc       = res['accuracy']
                mejor_resultado = res
                print("  🌟 ¡Nuevo récord!")
            else:
                print("")
                del res

        except Exception as e:
            print(f"\n   ❌ Error en intento {i+1}: {e}")

    tiempo_total = (time.time() - inicio_global) / 60
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f} minutos")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Curvas de aprendizaje ──────────────────────────────────────────────
    historial = mejor_resultado['historial']
    epocas    = range(len(historial.history['accuracy']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'EfficientNetB3 [{condicion}] — Mejor intento', fontsize=14)

    ax1.plot(epocas, historial.history['accuracy'],     label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epocas, historial.history['loss'],     label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Reporte y matriz ───────────────────────────────────────────────────
    generador_val.reset()
    predicciones   = mejor_resultado['modelo'].predict(
        generador_val, verbose=1
    )
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    print(f"\n--- Reporte de clasificación [{condicion}] ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — EfficientNetB3 [{condicion}]')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    # ── Retorna resultado y predicciones para checkpoint ───────────────────
    return mejor_resultado, predicciones


def buscar_mejor_gb(X_train, y_train, X_test, y_test,
                    encoder, etiqueta="PCA",
                    n_intentos=10, balanceado=False,
                    condicion="SIN BALANCE", color_mapa='Oranges'):
    """
    Búsqueda Monte Carlo del mejor Gradient Boosting variando la semilla.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  GRADIENT BOOSTING [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_gb(
            X_train, y_train, X_test, y_test,
            semilla=semilla_actual,
            balanceado=balanceado
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte ────────────────────────────────────────────────────────────
    clases_nombres = encoder.classes_
    y_pred         = mejor_resultado['y_pred']

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # ── Matriz de confusión ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=clases_nombres,
        cmap=color_mapa,
        ax=ax,
        colorbar=True
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'Gradient Boosting [{condicion}]\nDatos: {etiqueta}')
    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_inception(generador_tren, generador_val,
                            n_clases, condicion="SIN BALANCE",
                            pesos_dict=None, n_intentos=3):
    """
    Búsqueda Monte Carlo del mejor InceptionV3 variando la semilla.
    """
    print(f"\n{'='*70}")
    print(f"  INCEPTIONV3 [{condicion}]")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        print(f"\n🔄 Intento {i+1}/{n_intentos}...")

        try:
            res = entrenar_inception_una_vez(
                generador_tren, generador_val,
                n_clases, semilla=semilla_actual,
                pesos_dict=pesos_dict
            )
            print(f"   ✅ Accuracy: {res['accuracy']:.4f} | "
                  f"Tiempo: {res['tiempo']/60:.1f} min", end="")

            if res['accuracy'] > mejor_acc:
                mejor_acc       = res['accuracy']
                mejor_resultado = res
                print("  🌟 ¡Nuevo récord!")
            else:
                print("")
                del res

        except Exception as e:
            print(f"\n   ❌ Error en intento {i+1}: {e}")

    tiempo_total = (time.time() - inicio_global) / 60
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f} minutos")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Curvas de aprendizaje ──────────────────────────────────────────────
    historial = mejor_resultado['historial']
    epocas    = range(len(historial.history['accuracy']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'InceptionV3 [{condicion}] — Mejor intento', fontsize=14)

    ax1.plot(epocas, historial.history['accuracy'],     label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epocas, historial.history['loss'],     label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Reporte y matriz ───────────────────────────────────────────────────
    generador_val.reset()
    predicciones   = mejor_resultado['modelo'].predict(
        generador_val, verbose=1
    )
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    acc_final = accuracy_score(y_real, y_pred)
    print(f"\n--- Reporte de clasificación [{condicion}] ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — InceptionV3 [{condicion}]')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return mejor_resultado, acc_final


def buscar_mejor_mlp(X_train, y_train, X_test, y_test,
                     encoder, etiqueta="PCA",
                     n_intentos=10, pesos_clases=None,
                     condicion="SIN BALANCE", color_mapa='Blues'):
    """
    Búsqueda Monte Carlo del mejor MLP variando la semilla de inicialización.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  MLP [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_mlp(
            X_train, y_train, X_test, y_test,
            encoder, semilla=semilla_actual,
            pesos_clases=pesos_clases
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")
            del res

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte y visualización ────────────────────────────────────────────
    historial      = mejor_resultado['historial']
    y_pred         = mejor_resultado['y_pred']
    clases_nombres = encoder.classes_

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(historial.history['loss'],     label='Entrenamiento')
    axes[0].plot(historial.history['val_loss'], label='Validación')
    axes[0].set_title(f'Pérdida — {etiqueta}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(historial.history['accuracy'],     label='Entrenamiento')
    axes[1].plot(historial.history['val_accuracy'], label='Validación')
    axes[1].set_title(f'Exactitud — {etiqueta}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Exactitud')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=color_mapa,
        xticklabels=clases_nombres,
        yticklabels=clases_nombres,
        ax=axes[2]
    )
    axes[2].set_title(f'Matriz de Confusión\n{etiqueta} [{condicion}]')
    axes[2].set_xlabel('Predicción')
    axes[2].set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_mlp_sklearn(X_train, y_train, X_test, y_test,
                              encoder, etiqueta="PCA",
                              n_intentos=10, color_mapa='Purples'):
    """
    Búsqueda Monte Carlo del mejor MLPClassifier variando la semilla.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  MLP SKLEARN [SIN BALANCE] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"  Nota: MLPClassifier no soporta class_weight")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_mlp_sklearn(
            X_train, y_train, X_test, y_test,
            semilla=semilla_actual
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")
    print(f"🔄 Épocas entrenadas: {mejor_resultado['modelo'].n_iter_}")

    # ── Reporte ────────────────────────────────────────────────────────────
    y_pred         = mejor_resultado['y_pred']
    clases_nombres = encoder.classes_
    modelo_ganador = mejor_resultado['modelo']

    print(f"\n--- Reporte de clasificación [{etiqueta}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # ── Gráfica de curvas de aprendizaje ───────────────────────────────────
    # Se usa doble eje porque pérdida y accuracy tienen escalas diferentes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Pérdida', color='steelblue')
    ax1.plot(modelo_ganador.loss_curve_,
             color='steelblue', label='Pérdida entrenamiento')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Exactitud validación', color='darkorange')
    ax2.plot(modelo_ganador.validation_scores_,
             color='darkorange', linestyle='--',
             label='Exactitud validación')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    plt.title(f'Curva de Aprendizaje — {etiqueta}\n(Mejor intento)')
    fig.tight_layout()
    plt.show()

    # ── Matriz de confusión ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=color_mapa,
        xticklabels=clases_nombres,
        yticklabels=clases_nombres,
        ax=ax
    )
    ax.set_title(f'Matriz de Confusión — {etiqueta}\n'
                 f'(Mejor de {n_intentos} intentos)')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_resnet(generador_tren, generador_val,
                         n_clases, condicion="SIN BALANCE",
                         pesos_dict=None, n_intentos=3):
    """
    Búsqueda Monte Carlo del mejor ResNet50 variando la semilla.
    """
    print(f"\n{'='*70}")
    print(f"  RESNET50 [{condicion}]")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        print(f"\n🔄 Intento {i+1}/{n_intentos}...")

        try:
            res = entrenar_resnet_una_vez(
                generador_tren, generador_val,
                n_clases, semilla=semilla_actual,
                pesos_dict=pesos_dict
            )
            print(f"   ✅ Accuracy: {res['accuracy']:.4f} | "
                  f"Tiempo: {res['tiempo']/60:.1f} min", end="")

            if res['accuracy'] > mejor_acc:
                mejor_acc       = res['accuracy']
                mejor_resultado = res
                print("  🌟 ¡Nuevo récord!")
            else:
                print("")
                del res

        except Exception as e:
            print(f"\n   ❌ Error en intento {i+1}: {e}")

    tiempo_total = (time.time() - inicio_global) / 60
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f} minutos")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Curvas de aprendizaje ──────────────────────────────────────────────
    historial = mejor_resultado['historial']
    epocas    = range(len(historial.history['accuracy']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'ResNet50 [{condicion}] — Mejor intento', fontsize=14)

    ax1.plot(epocas, historial.history['accuracy'],     label='Entrenamiento')
    ax1.plot(epocas, historial.history['val_accuracy'], label='Validación')
    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epocas, historial.history['loss'],     label='Entrenamiento')
    ax2.plot(epocas, historial.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida')
    ax2.set_xlabel('Épocas')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Reporte y matriz ───────────────────────────────────────────────────
    generador_val.reset()
    predicciones   = mejor_resultado['modelo'].predict(
        generador_val, verbose=1
    )
    y_pred         = np.argmax(predicciones, axis=1)
    y_real         = generador_val.classes
    nombres_clases = list(generador_val.class_indices.keys())

    acc_final = accuracy_score(y_real, y_pred)
    print(f"\n--- Reporte de clasificación [{condicion}] ---")
    print(classification_report(y_real, y_pred,
                                 target_names=nombres_clases,
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases, ax=ax)
    ax.set_title(f'Matriz de Confusión — ResNet50 [{condicion}]')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return mejor_resultado, acc_final


def buscar_mejor_rf(X_train, y_train, X_test, y_test,
                    encoder, etiqueta="PCA",
                    n_intentos=10, balanceado=False,
                    condicion="SIN BALANCE", color_mapa='Blues'):
    """
    Búsqueda Monte Carlo del mejor Random Forest variando la semilla.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  RANDOM FOREST [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_rf(
            X_train, y_train, X_test, y_test,
            semilla=semilla_actual,
            balanceado=balanceado
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte ────────────────────────────────────────────────────────────
    clases_nombres = encoder.classes_
    y_pred         = mejor_resultado['y_pred']

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    # ── Matriz de confusión ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=clases_nombres,
        cmap=color_mapa,
        ax=ax,
        colorbar=True
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.title(f'Random Forest [{condicion}]\nDatos: {etiqueta}')
    plt.tight_layout()
    plt.show()

    return mejor_resultado


def buscar_mejor_transformer(X_train, y_train, X_test, y_test,
                              encoder, etiqueta="PCA",
                              n_intentos=10, pesos_clases=None,
                              condicion="SIN BALANCE", color_mapa='Blues'):
    """
    Búsqueda Monte Carlo del mejor Transformer variando la semilla.
    Reporta el modelo con mayor accuracy entre todos los intentos.
    """
    print(f"\n{'='*70}")
    print(f"  TRANSFORMER [{condicion}] — {etiqueta}")
    print(f"  Buscando mejor modelo en {n_intentos} intentos...")
    print(f"{'='*70}")

    mejor_acc       = 0.0
    mejor_resultado = None
    inicio_global   = time.time()

    for i in range(n_intentos):
        semilla_actual = 42 + i
        res = entrenar_transformer(
            X_train, y_train, X_test, y_test,
            encoder, semilla=semilla_actual,
            pesos_clases=pesos_clases
        )

        print(f"  Intento {i+1:02d}/{n_intentos} | "
              f"Accuracy: {res['accuracy']:.4f}", end="")

        if res['accuracy'] > mejor_acc:
            mejor_acc       = res['accuracy']
            mejor_resultado = res
            print("  🌟 ¡Nuevo récord!")
        else:
            print("")
            del res

    tiempo_total = time.time() - inicio_global
    print(f"\n✅ Búsqueda finalizada en {tiempo_total:.1f}s")
    print(f"🏆 Mejor accuracy: {mejor_acc:.4f}")

    # ── Reporte y visualización ────────────────────────────────────────────
    historial      = mejor_resultado['historial']
    y_pred         = mejor_resultado['y_pred']
    clases_nombres = encoder.classes_

    print(f"\n--- Reporte de clasificación [{etiqueta} | {condicion}] ---")
    print(classification_report(
        y_test, y_pred,
        target_names=clases_nombres,
        zero_division=0
    ))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pérdida
    axes[0].plot(historial.history['loss'],     label='Entrenamiento')
    axes[0].plot(historial.history['val_loss'], label='Validación')
    axes[0].set_title(f'Pérdida — {etiqueta}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(historial.history['accuracy'],     label='Entrenamiento')
    axes[1].plot(historial.history['val_accuracy'], label='Validación')
    axes[1].set_title(f'Exactitud — {etiqueta}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Exactitud')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=color_mapa,
        xticklabels=clases_nombres,
        yticklabels=clases_nombres,
        ax=axes[2]
    )
    axes[2].set_title(f'Matriz de Confusión\n{etiqueta} [{condicion}]')
    axes[2].set_xlabel('Predicción')
    axes[2].set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    return mejor_resultado


def correr_svm_por_condicion(datasets, kernels, y_test, encoder,
                              condicion, color_mapa, balanceado=False):
    """
    Corre SVM para todos los datasets y kernels bajo una condición
    de balanceo específica.
    """
    print(f"\n⏳ Entrenando SVM [{condicion}]...")
    resultados = {}

    for nombre, (X_tr, X_te, y_tr) in datasets.items():
        resultados[nombre] = {}
        for kernel in kernels:
            print(f"  → {nombre} | Kernel: {kernel.upper()}...", end="")
            resultados[nombre][kernel] = entrenar_svm(
                X_tr, y_tr, X_te, y_test,
                kernel=kernel,
                balanceado=balanceado
            )
            print(f" Accuracy: {resultados[nombre][kernel]['accuracy']:.4f}")

    reportar_svm(resultados, y_test, encoder,
                 condicion=condicion, color_mapa=color_mapa)
    return resultados

