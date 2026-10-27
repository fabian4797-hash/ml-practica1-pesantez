from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def calcular_costos(y_real, y_predicha, costo_fp, costo_fn):
    """Calcula los errores y su costo económico."""
    tn, fp, fn, tp = confusion_matrix(
        y_real, y_predicha, labels=[0, 1]
    ).ravel()

    return {
        "FP": int(fp),
        "FN": int(fn),
        "Costo total ($)": int(fp * costo_fp + fn * costo_fn)
    }


def main():
    """Compara el costo de Random Forest con el de predecir siempre 0."""
    # Supuestos económicos establecidos por la guía.
    costo_fp = 50
    costo_fn = 500

    # Localizar el dataset desde la ubicación de este script.
    raiz_proyecto = Path(__file__).resolve().parent.parent
    ruta_dataset = raiz_proyecto / "data" / "processed" / "dataset.parquet"
    df = pd.read_parquet(ruta_dataset)

    # Mantener la misma clasificación y partición de la sesión 1.
    columnas_sensores = [f"US{i}" for i in range(1, 25)]
    X = df[columnas_sensores]
    y = (df["Movimiento"] == "Slight-Left-Turn").astype(int)

    es_train = df["Conjunto"] == "train"
    es_test = df["Conjunto"] == "test"

    X_train = X.loc[es_train]
    X_test = X.loc[es_test]
    y_train = y.loc[es_train]
    y_test = y.loc[es_test]

    # Reproducir el modelo de la sesión 1.
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    modelo_rf.fit(X_train, y_train)

    y_pred_rf = modelo_rf.predict(X_test)
    y_pred_baseline = np.zeros(len(y_test), dtype=int)

    resultado_rf = calcular_costos(
        y_test, y_pred_rf, costo_fp, costo_fn
    )
    resultado_baseline = calcular_costos(
        y_test, y_pred_baseline, costo_fp, costo_fn
    )

    comparacion = pd.DataFrame(
        [resultado_rf, resultado_baseline],
        index=["Random Forest", "Siempre 0"]
    )

    print(f"Costo por falso positivo: ${costo_fp}")
    print(f"Costo por falso negativo: ${costo_fn}")
    print(f"Registros evaluados: {len(y_test)}")
    print("\nComparación de costos:")
    print(comparacion.to_string())

    ahorro = (
        resultado_baseline["Costo total ($)"]
        - resultado_rf["Costo total ($)"]
    )

    print("\nConclusión:")
    if ahorro > 0:
        print(
            f"Random Forest reduce el costo en ${ahorro:,.2f} "
            "respecto al clasificador Siempre 0."
        )
    elif ahorro == 0:
        print("Ambos clasificadores tienen el mismo costo.")
    else:
        print(
            f"Random Forest aumenta el costo en ${-ahorro:,.2f} "
            "respecto al clasificador Siempre 0."
        )

    print(
        "Esta comparación corresponde al conjunto de prueba "
        "y a los costos supuestos por la guía."
    )


if __name__ == "__main__":
    main()