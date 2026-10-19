import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def load_data():
    """
    Carga los datos procesados y selecciona únicamente
    las 24 lecturas de los sensores ultrasónicos.
    """

    path = "data/processed/dataset.parquet"

    df = pd.read_parquet(path)

    sensor_columns = [f"US{i}" for i in range(1, 25)]

    X = df[sensor_columns]

    return X


def find_optimal_k(X):
    """
    Determina posibles valores de k mediante
    el método del codo.
    """

    inertias = []
    valores_k = range(2, 11)

    for k in valores_k:

        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        modelo.fit(X)

        inertias.append(modelo.inertia_)

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores_k,
        inertias,
        marker="o"
    )

    plt.title("Método del codo - K-Means")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.xticks(list(valores_k))
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return inertias

def train_kmeans(X, k):
    """
    Entrena un modelo K-Means con el número
    de clusters seleccionado.
    """

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(X)

    return modelo

def train_dbscan(X, eps, min_samples):
    """
    Entrena un modelo DBSCAN con los parámetros indicados.
    """

    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    modelo.fit(X)

    return modelo

if __name__ == "__main__":

    print("Cargando datos...")

    X = load_data()

    print("Datos cargados correctamente.")
    print("Tamaño de X:", X.shape)

    print("\nBuscando k óptimo...")

    find_optimal_k(X)

    k_optimo = 4

    print("\nEntrenando K-Means...")
    print("k seleccionado:", k_optimo)

    modelo_kmeans = train_kmeans(X, k_optimo)

    print("K-Means entrenado correctamente.")
    print("Clusters encontrados:", modelo_kmeans.n_clusters)

    score_kmeans = silhouette_score(
        X,
        modelo_kmeans.labels_
    )

    print("Silhouette Score K-Means:", round(score_kmeans, 4))

    print("\nEntrenando DBSCAN...")

    eps = 3.0
    min_samples = 5

    modelo_dbscan = train_dbscan(
        X,
        eps,
        min_samples
    )

    print("DBSCAN entrenado correctamente.")
    print("eps:", eps)
    print("min_samples:", min_samples)

    etiquetas_dbscan = modelo_dbscan.labels_

    numero_clusters = len(
        set(etiquetas_dbscan)
    ) - (1 if -1 in etiquetas_dbscan else 0)

    numero_ruido = list(
        etiquetas_dbscan
    ).count(-1)

    print("Clusters encontrados:", numero_clusters)
    print("Puntos considerados ruido:", numero_ruido)

    print("\nProbando diferentes valores de eps...")

    valores_eps = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    for eps_prueba in valores_eps:

        modelo = train_dbscan(
            X,
            eps_prueba,
            min_samples=5
        )

        etiquetas = modelo.labels_

        numero_clusters = len(
            set(etiquetas)
        ) - (1 if -1 in etiquetas else 0)

        numero_ruido = list(
            etiquetas
        ).count(-1)

        mascara = etiquetas != -1

        if numero_clusters >= 2:

            score = silhouette_score(
                X[mascara],
                etiquetas[mascara]
            )

            print(
                f"eps={eps_prueba:.1f} | "
                f"clusters={numero_clusters} | "
                f"ruido={numero_ruido} | "
                f"silhouette={score:.4f}"
            )

        else:

            print(
                f"eps={eps_prueba:.1f} | "
                f"clusters={numero_clusters} | "
                f"ruido={numero_ruido} | "
                f"silhouette=No disponible"
            )


    print("\nVisualizando DBSCAN con eps=5.0...")

    modelo_dbscan_final = train_dbscan(
        X,
        eps=5.0,
        min_samples=5
    )

    etiquetas_finales = modelo_dbscan_final.labels_

    # Reducir las 24 variables a 2 dimensiones
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=etiquetas_finales,
        s=20
    )

    plt.title("Clusters encontrados por DBSCAN")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nVisualizando DBSCAN con eps=5.0...")

    modelo_dbscan_final = train_dbscan(
        X,
        eps=5.0,
        min_samples=5
    )

    # Guardar modelo DBSCAN
    import joblib
    from pathlib import Path

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        modelo_dbscan_final,
        models_dir / "dbscan.pkl"
    )

    print("DBSCAN guardado en:")
    print(models_dir / "dbscan.pkl")

    etiquetas_finales = modelo_dbscan_final.labels_

    # Reducir las 24 variables a 2 dimensiones
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    print("\n==============================")
    print("COMPARACIÓN DE MODELOS")
    print("==============================")

    # Resultados de K-Means
    clusters_kmeans = modelo_kmeans.n_clusters
    ruido_kmeans = 0
    silhouette_kmeans = score_kmeans

    # Resultados de DBSCAN
    etiquetas_dbscan = modelo_dbscan_final.labels_

    clusters_dbscan = len(
        set(etiquetas_dbscan)
    ) - (1 if -1 in etiquetas_dbscan else 0)

    ruido_dbscan = list(
        etiquetas_dbscan
    ).count(-1)

    mascara_dbscan = etiquetas_dbscan != -1

    silhouette_dbscan = silhouette_score(
        X[mascara_dbscan],
        etiquetas_dbscan[mascara_dbscan]
    )

    print("\nModelo       Clusters    Ruido    Silhouette")
    print(
        f"K-Means      {clusters_kmeans:<10} "
        f"{ruido_kmeans:<8} "
        f"{silhouette_kmeans:.4f}"
    )

    print(
        f"DBSCAN       {clusters_dbscan:<10} "
        f"{ruido_dbscan:<8} "
        f"{silhouette_dbscan:.4f}"
    )

