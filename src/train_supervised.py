from pathlib import Path
import sys
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Permite importar data_loader.py
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

from data_loader import load_raw_data, preprocess, split_data


def train_random_forest(X_train, y_train):
    """
    Entrena un modelo Random Forest
    utilizando los hiperparámetros por defecto.
    """
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model

def train_svm(X_train, y_train):
    """
    Entrena un clasificador SVM.
    """

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo utilizando:
    - Accuracy
    - F1-score
    - Matriz de confusión
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    matrix = confusion_matrix(y_test, y_pred)

    print("\n===== EVALUACIÓN DEL MODELO =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nMatriz de confusión:")
    print(matrix)

    return accuracy, f1, matrix


def main():

    # Ruta del dataset original
    project_dir = Path(__file__).resolve().parent.parent

    data_path = (
        project_dir
        / "data"
        / "raw"
        / "sensor_readings_24.data"
    )

    # Cargar datos
    df = load_raw_data(data_path)

    print("Dataset cargado:")
    print(df.shape)

    # Separar X e y
    X, y = preprocess(df)

    print("\nDatos preparados:")
    print("X:", X.shape)
    print("y:", y.shape)

    # Dividir y escalar datos
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    print("\nDivisión de datos:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # Crear carpeta models
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    print("\n==============================")
    print("ENTRENANDO RANDOM FOREST")
    print("==============================")

    inicio_rf = time.perf_counter()

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    tiempo_rf = time.perf_counter() - inicio_rf

    print("Modelo entrenado correctamente.")

    accuracy_rf, f1_rf, matrix_rf = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # Guardar Random Forest
    rf_path = models_dir / "random_forest.pkl"

    joblib.dump(
        rf_model,
        rf_path
    )

    print(f"\nRandom Forest guardado en:")
    print(rf_path)

    print(f"Tiempo de entrenamiento: {tiempo_rf:.4f} segundos")


    # =====================================================
    # SVM
    # =====================================================

    print("\n==============================")
    print("ENTRENANDO SVM")
    print("==============================")

    inicio_svm = time.perf_counter()

    svm_model = train_svm(
        X_train,
        y_train
    )

    tiempo_svm = time.perf_counter() - inicio_svm

    print("Modelo entrenado correctamente.")

    accuracy_svm, f1_svm, matrix_svm = evaluate_model(
        svm_model,
        X_test,
        y_test
    )

    # Guardar SVM
    svm_path = models_dir / "svm.pkl"

    joblib.dump(
        svm_model,
        svm_path
    )

    print(f"\nSVM guardado en:")
    print(svm_path)

    print(f"Tiempo de entrenamiento: {tiempo_svm:.4f} segundos")


    # =====================================================
    # COMPARACIÓN
    # =====================================================

    print("\n======================================")
    print("COMPARACIÓN DE MODELOS")
    print("======================================")

    print(f"{'Modelo':<20}{'Accuracy':<15}{'F1':<15}{'Tiempo (s)':<15}")

    print(
        f"{'Random Forest':<20}"
        f"{accuracy_rf:<15.4f}"
        f"{f1_rf:<15.4f}"
        f"{tiempo_rf:<15.4f}"
    )

    print(
        f"{'SVM':<20}"
        f"{accuracy_svm:<15.4f}"
        f"{f1_svm:<15.4f}"
        f"{tiempo_svm:<15.4f}"
    )


if __name__ == "__main__":
    main()
