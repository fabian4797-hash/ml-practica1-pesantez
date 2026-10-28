"""Sesión 4: optimización de Random Forest con Optuna."""

import json
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def main():
    """Ejecuta el estudio y guarda el mejor resultado."""
    raiz = Path(__file__).resolve().parent.parent
    ruta_datos = raiz / "data" / "processed" / "dataset.parquet"

    df = pd.read_parquet(ruta_datos)

    # Mismas variables y clase positiva de las sesiones anteriores.
    sensores = [f"US{i}" for i in range(1, 25)]
    X = df[sensores]
    y = (df["Movimiento"] == "Slight-Left-Turn").astype(int)

    # Usamos únicamente la partición original de entrenamiento.
    es_desarrollo = df["Conjunto"] == "train"
    X_desarrollo = X.loc[es_desarrollo]
    y_desarrollo = y.loc[es_desarrollo]

    # Reproducimos la división de la sesión 3.
    X_train, X_val, y_train, y_val = train_test_split(
        X_desarrollo,
        y_desarrollo,
        test_size=0.20,
        stratify=y_desarrollo,
        random_state=42,
    )

    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de validación: {len(X_val)}")
    print(f"Positivos en validación: {int(y_val.sum())}")

    def objetivo(trial):
        """Entrena un modelo y devuelve su F1 en validación."""
        parametros = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 200, step=50
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 5, 10, 15, 20]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 4
            ),
        }

        modelo = RandomForestClassifier(
            **parametros,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)

        columna_positiva = list(modelo.classes_).index(1)
        probabilidades = modelo.predict_proba(X_val)[:, columna_positiva]
        predicciones = (probabilidades >= 0.50).astype(int)

        return f1_score(
            y_val,
            predicciones,
            pos_label=1,
            zero_division=0,
        )

    carpeta_resultados = raiz / "reports"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    ruta_estudio = carpeta_resultados / "p2_04_optuna.db"
    almacenamiento = f"sqlite:///{ruta_estudio.as_posix()}"

    estudio = optuna.create_study(
        study_name="p2_random_forest_f1",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=almacenamiento,
        load_if_exists=True,
    )

    # Al volver a ejecutar, completamos lo que falta hasta llegar a 20.
    completadas = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in estudio.trials
    )
    pendientes = max(0, 20 - completadas)

    if pendientes > 0:
        print(f"\nPruebas nuevas por ejecutar: {pendientes}")
        estudio.optimize(objetivo, n_trials=pendientes)
    else:
        print("\nEl estudio ya tiene al menos 20 pruebas completadas.")

    total_completadas = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in estudio.trials
    )

    resultado = {
        "estudio": estudio.study_name,
        "metrica": "F1 de la clase positiva en validacion",
        "umbral": 0.50,
        "pruebas_completadas": total_completadas,
        "mejor_trial": estudio.best_trial.number,
        "mejor_f1_validacion": estudio.best_value,
        "mejores_hiperparametros": estudio.best_params,
        "random_state_modelo": 42,
        "registros_entrenamiento": len(X_train),
        "registros_validacion": len(X_val),
    }

    ruta_resultado = carpeta_resultados / "p2_04_mejor_resultado.json"
    with ruta_resultado.open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=4, ensure_ascii=False)

    print("\nRESULTADO DEL ESTUDIO")
    print(f"Pruebas completadas: {total_completadas}")
    print(f"Mejor trial: {estudio.best_trial.number}")
    print(f"Mejor F1 en validación: {estudio.best_value:.4f}")
    print("Mejores hiperparámetros:")
    for nombre, valor in estudio.best_params.items():
        print(f"  {nombre}: {valor}")

    print(f"\nEstudio guardado en: {ruta_estudio}")
    print(f"Mejor resultado guardado en: {ruta_resultado}")


if __name__ == "__main__":
    main()