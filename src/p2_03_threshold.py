from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    """Entrena el modelo y muestra probabilidades en validación."""
    raiz_proyecto = Path(__file__).resolve().parent.parent
    ruta_dataset = raiz_proyecto / "data" / "processed" / "dataset.parquet"
    df = pd.read_parquet(ruta_dataset)

    columnas_sensores = [f"US{i}" for i in range(1, 25)]
    X = df[columnas_sensores]
    y = (df["Movimiento"] == "Slight-Left-Turn").astype(int)

    # Utilizar únicamente la partición de entrenamiento de la práctica.
    es_train = df["Conjunto"] == "train"
    X_desarrollo = X.loc[es_train]
    y_desarrollo = y.loc[es_train]

    # Reservar el 20 % de esta partición para elegir el umbral.
    X_train, X_val, y_train, y_val = train_test_split(
        X_desarrollo,
        y_desarrollo,
        test_size=0.20,
        stratify=y_desarrollo,
        random_state=42
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    modelo_rf.fit(X_train, y_train)

    # Las columnas siguen el orden indicado por modelo_rf.classes_.
    probabilidades = modelo_rf.predict_proba(X_val)
    columna_positiva = list(modelo_rf.classes_).index(1)
    probabilidad_positiva = probabilidades[:, columna_positiva]

    ejemplos = pd.DataFrame({
        "Clase real": y_val.to_numpy(),
        "Probabilidad clase 1": probabilidad_positiva,
        "Predicción umbral 0.50": (
            probabilidad_positiva >= 0.50
        ).astype(int),
        "Predicción umbral 0.30": (
            probabilidad_positiva >= 0.30
        ).astype(int)
    })

    print("Registros para entrenar:", len(y_train))
    print("Registros para validación:", len(y_val))
    print("Positivos reales en validación:", int(y_val.sum()))
    print("Orden de las clases:", modelo_rf.classes_)

    print("\nPrimeros 10 ejemplos:")
    print(ejemplos.head(10).to_string(index=False))

    cambios = (
        ejemplos["Predicción umbral 0.50"]
        != ejemplos["Predicción umbral 0.30"]
    )

    print("\nCantidad de decisiones que cambian:", int(cambios.sum()))
    print("\nEjemplos de decisiones que cambian:")
    print(ejemplos.loc[cambios].head(10).to_string(index=False))


    # Costos establecidos por la guía.
    costo_fp = 50
    costo_fn = 500

    # Evaluar 17 umbrales: 0.10, 0.15, ..., 0.90.
    resultados = []

    for porcentaje in range(10, 91, 5):
        umbral = porcentaje / 100
        predicciones = (probabilidad_positiva >= umbral).astype(int)

        fp = int(((y_val == 0) & (predicciones == 1)).sum())
        fn = int(((y_val == 1) & (predicciones == 0)).sum())
        costo = fp * costo_fp + fn * costo_fn

        resultados.append({
            "Umbral": umbral,
            "FP": fp,
            "FN": fn,
            "Costo ($)": costo
        })

    tabla_costos = pd.DataFrame(resultados)

    print("\nCostos por umbral en VALIDACIÓN:")
    print(tabla_costos.to_string(index=False))

    # Mostrar todos los umbrales que alcanzan el costo mínimo.
    costo_minimo = tabla_costos["Costo ($)"].min()
    mejores = tabla_costos.loc[
        tabla_costos["Costo ($)"] == costo_minimo
    ]

    # Decisión de apoyo: en caso de empate, elegir el menor umbral.
    umbral_optimo = float(mejores["Umbral"].min())

    costo_050 = int(
        tabla_costos.loc[
            tabla_costos["Umbral"] == 0.50, "Costo ($)"
        ].iloc[0]
    )
    ahorro = costo_050 - costo_minimo

    print("\nUmbrales con el menor costo:")
    print(mejores.to_string(index=False))

    print(f"\nUmbral seleccionado: {umbral_optimo:.2f}")
    print(f"Costo con umbral 0.50: ${costo_050:,.2f}")
    print(f"Costo mínimo en validación: ${costo_minimo:,.2f}")
    print(f"Reducción del costo en validación: ${ahorro:,.2f}")


    # Crear la carpeta de salida indicada por la guía.
    carpeta_figuras = raiz_proyecto / "reports" / "figures"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    ruta_grafico = carpeta_figuras / "threshold_tuning.png"

    # Graficar los costos calculados en validación.
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        tabla_costos["Umbral"],
        tabla_costos["Costo ($)"],
        marker="o",
        label="Costo en validación"
    )

    # Marcar todos los umbrales que alcanzan el mínimo.
    ax.scatter(
        mejores["Umbral"],
        mejores["Costo ($)"],
        color="green",
        s=100,
        zorder=3,
        label=f"Costo mínimo: ${costo_minimo:,.0f}"
    )

    ax.scatter(
        [0.50],
        [costo_050],
        color="red",
        s=100,
        zorder=3,
        label=f"Referencia 0.50: ${costo_050:,.0f}"
    )

    ax.axvline(
        umbral_optimo,
        color="green",
        linestyle="--",
        alpha=0.6,
        label=f"Umbral seleccionado: {umbral_optimo:.2f}"
    )

    ax.set_title("Costo monetario vs. umbral de decisión — Validación")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Costo total ($)")
    ax.set_xticks(tabla_costos["Umbral"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(ruta_grafico, dpi=150)
    plt.close(fig)

    print(f"\nGráfico guardado en: {ruta_grafico}")

    # Evaluación en prueba: el umbral ya fue elegido en validación.
    es_test = df["Conjunto"] == "test"
    X_test = X.loc[es_test]
    y_test = y.loc[es_test]

    # Usar el mismo modelo, sin volver a entrenarlo.
    probabilidades_test = modelo_rf.predict_proba(X_test)
    probabilidad_positiva_test = probabilidades_test[:, columna_positiva]

    resultados_test = []

    for nombre, umbral in [
        ("Referencia", 0.50),
        ("Seleccionado en validación", umbral_optimo)
    ]:
        predicciones_test = (
            probabilidad_positiva_test >= umbral
        ).astype(int)

        fp_test = int(
            ((y_test == 0) & (predicciones_test == 1)).sum()
        )
        fn_test = int(
            ((y_test == 1) & (predicciones_test == 0)).sum()
        )
        costo_test = fp_test * costo_fp + fn_test * costo_fn

        resultados_test.append({
            "Configuración": nombre,
            "Umbral": umbral,
            "FP": fp_test,
            "FN": fn_test,
            "Costo ($)": costo_test
        })

    tabla_test = pd.DataFrame(resultados_test)

    print(f"\nEvaluación en PRUEBA: {len(y_test)} registros")
    print(tabla_test.to_string(index=False))

    diferencia_test = (
        resultados_test[0]["Costo ($)"]
        - resultados_test[1]["Costo ($)"]
    )

    if diferencia_test > 0:
        print(
            f"\nEl umbral seleccionado reduce el costo en prueba "
            f"en ${diferencia_test:,.2f}."
        )
    elif diferencia_test == 0:
        print("\nAmbos umbrales tienen el mismo costo en prueba.")
    else:
        print(
            f"\nEl umbral seleccionado aumenta el costo en prueba "
            f"en ${-diferencia_test:,.2f}."
        )

    print(
        "El umbral se mantiene como fue seleccionado en validación; "
        "no se reajusta con los resultados de prueba."
    )

if __name__ == "__main__":
    main()

