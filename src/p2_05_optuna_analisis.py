"""Sesión 5: análisis de los resultados de Optuna."""
import json
from pathlib import Path

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)
from optuna.importance import FanovaImportanceEvaluator

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def main():
    """Carga el estudio guardado y grafica su historia."""
    raiz = Path(__file__).resolve().parent.parent
    ruta_estudio = raiz / "reports" / "p2_04_optuna.db"

    # Comprobar que exista el estudio de la sesión 4.
    if not ruta_estudio.is_file():
        raise FileNotFoundError(
            f"No se encontró el estudio: {ruta_estudio}"
        )

    almacenamiento = f"sqlite:///{ruta_estudio.as_posix()}"

    estudio = optuna.load_study(
        study_name="p2_random_forest_f1",
        storage=almacenamiento,
    )

    completadas = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in estudio.trials
    )

    if completadas == 0:
        raise ValueError("El estudio no tiene pruebas completadas.")

    print(f"Estudio cargado: {estudio.study_name}")
    print(f"Pruebas completadas: {completadas}")
    print(f"Mejor trial: {estudio.best_trial.number}")
    print(f"Mejor F1 en validación: {estudio.best_value:.4f}")
    print(f"Mejores hiperparámetros: {estudio.best_params}")

    # Crear el gráfico solicitado por la guía.
    figura = plot_optimization_history(
        estudio,
        target_name="F1 en validación",
    )

    figura.update_layout(
        title="Historia de optimización — Random Forest",
        xaxis_title="Número de trial",
        yaxis_title="F1 en validación",
    )

    carpeta_figuras = raiz / "reports" / "figures"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    ruta_grafico = carpeta_figuras / "optuna_historia.html"

    # Incluir Plotly para abrir el gráfico sin conexión a Internet.
    figura.write_html(
        str(ruta_grafico),
        include_plotlyjs=True,
        full_html=True,
    )

    print(f"\nGráfico guardado en: {ruta_grafico}")


        # Estimar la importancia de los hiperparámetros.
    evaluador = FanovaImportanceEvaluator(seed=42)

    figura_importancias = plot_param_importances(
        estudio,
        evaluator=evaluador,
        target_name="F1 en validación",
    )

    figura_importancias.update_layout(
        title="Importancia de hiperparámetros — Random Forest",
        xaxis_title="Importancia relativa estimada",
        yaxis_title="Hiperparámetro",
    )

    ruta_importancias = carpeta_figuras / "optuna_importancias.html"

    figura_importancias.write_html(
        str(ruta_importancias),
        include_plotlyjs=True,
        full_html=True,
    )

    print(f"Gráfico de importancias guardado en: {ruta_importancias}")



        # Exportar la configuración seleccionada para el modelo final.
    configuracion = {
        "modelo": "RandomForestClassifier",
        "clase_positiva": "Slight-Left-Turn",
        "variables": [f"US{i}" for i in range(1, 25)],
        "hiperparametros": {
            **estudio.best_params,
            "random_state": 42,
            "n_jobs": -1,
        },
        "umbral": 0.50,
        "regla_decision": "probabilidad >= umbral",
        "seleccion": {
            "estudio": estudio.study_name,
            "mejor_trial": estudio.best_trial.number,
            "metrica": "F1 de la clase positiva en validacion",
            "valor": estudio.best_value,
        },
    }

    ruta_configuracion = raiz / "config.json"

    with ruta_configuracion.open("w", encoding="utf-8") as archivo:
        json.dump(
            configuracion,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Configuración guardada en: {ruta_configuracion}")

        # Leer la configuración exportada.
    with ruta_configuracion.open("r", encoding="utf-8") as archivo:
        config_final = json.load(archivo)

    # Cargar los datos y conservar el orden de las variables.
    ruta_datos = raiz / "data" / "processed" / "dataset.parquet"
    df = pd.read_parquet(ruta_datos)

    X = df[config_final["variables"]]
    y = (
        df["Movimiento"] == config_final["clase_positiva"]
    ).astype(int)

    # Entrenamiento final con toda la partición original train.
    es_train = df["Conjunto"] == "train"
    X_train_final = X.loc[es_train]
    y_train_final = y.loc[es_train]

    modelo_final = RandomForestClassifier(
        **config_final["hiperparametros"]
    )
    modelo_final.fit(X_train_final, y_train_final)

    # Guardar el modelo junto con la configuración de inferencia.
    carpeta_modelos = raiz / "models"
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    ruta_modelo = carpeta_modelos / "p2_random_forest_final.joblib"

    paquete_modelo = {
        "modelo": modelo_final,
        "variables": config_final["variables"],
        "clase_positiva": config_final["clase_positiva"],
        "umbral": config_final["umbral"],
        "regla_decision": config_final["regla_decision"],
        "registros_entrenamiento": len(X_train_final),
    }

    joblib.dump(paquete_modelo, ruta_modelo)

    print("\nENTRENAMIENTO FINAL")
    print(f"Registros utilizados: {len(X_train_final)}")
    print(f"Positivos: {int(y_train_final.sum())}")
    print(f"Clases del modelo: {modelo_final.classes_}")
    print(f"Umbral guardado: {config_final['umbral']:.2f}")
    print(f"Modelo guardado en: {ruta_modelo}")

        # Reproducir el Random Forest original de la sesión 2.
    modelo_original = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    modelo_original.fit(X_train_final, y_train_final)

    # Evaluar ambos modelos sobre los mismos datos de prueba.
    es_test = df["Conjunto"] == "test"
    X_test = X.loc[es_test]
    y_test = y.loc[es_test]

    costo_fp = 50
    costo_fn = 500
    umbral = config_final["umbral"]

    resultados = []
    predicciones_modelos = {}

    for nombre, modelo in [
        ("Original", modelo_original),
        ("Seleccionado por Optuna", modelo_final),
    ]:
        columna_positiva = list(modelo.classes_).index(1)
        probabilidades = modelo.predict_proba(X_test)[:, columna_positiva]
        predicciones = (probabilidades >= umbral).astype(int)

        predicciones_modelos[nombre] = predicciones

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predicciones,
            labels=[0, 1],
        ).ravel()

        resultados.append({
            "Modelo": nombre,
            "Umbral": umbral,
            "Accuracy": accuracy_score(y_test, predicciones),
            "Precision": precision_score(
                y_test, predicciones, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predicciones, zero_division=0
            ),
            "F1": f1_score(
                y_test, predicciones, zero_division=0
            ),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Costo ($)": int(fp * costo_fp + fn * costo_fn),
        })

    tabla_comparacion = pd.DataFrame(resultados)

    print(f"\nCOMPARACIÓN EN PRUEBA: {len(y_test)} registros")
    print("Regla común: probabilidad >= umbral")
    print(
        tabla_comparacion.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.4f}",
        )
    )

    diferencias = int((
        predicciones_modelos["Original"]
        != predicciones_modelos["Seleccionado por Optuna"]
    ).sum())

    ahorro = resultados[0]["Costo ($)"] - resultados[1]["Costo ($)"]

    print(f"\nPredicciones diferentes entre modelos: {diferencias}")

    if ahorro > 0:
        print(f"El modelo seleccionado reduce el costo en ${ahorro:,.2f}.")
    elif ahorro == 0:
        print("Ambos modelos tienen el mismo costo en prueba.")
    else:
        print(
            f"El modelo seleccionado aumenta el costo "
            f"en ${-ahorro:,.2f}."
        )

    # Guardar los resultados sin redondear las métricas.
    ruta_comparacion = raiz / "reports" / "p2_05_comparacion.csv"
    tabla_comparacion.to_csv(
        ruta_comparacion,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Comparación guardada en: {ruta_comparacion}")


if __name__ == "__main__":
    main()