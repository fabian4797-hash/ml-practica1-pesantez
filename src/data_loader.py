import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SENSOR_COLUMNS = [f"US{i}" for i in range(1, 25)]

TARGET_COLUMN = "Movimiento"


def load_raw_data(path):
    """
    Carga el dataset original.
    """

    df = pd.read_csv(
        path,
        sep=",",
        header=None
    )

    print("Columnas detectadas:", len(df.columns))

    df.columns = SENSOR_COLUMNS + [TARGET_COLUMN]

    return df


def preprocess(df):
    """
    Separa las variables de entrada (X)
    y la variable objetivo (y).
    """

    X = df[SENSOR_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y):
    """
    Divide los datos en entrenamiento y prueba.
    Aplica RobustScaler utilizando únicamente
    los datos de entrenamiento.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )

