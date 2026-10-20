# Práctica 1 - Machine Learning

## Descripción del proyecto

Proyecto de Machine Learning desarrollado como parte de la Práctica 1.

El objetivo es analizar un conjunto de datos de lecturas de sensores,
realizar un proceso de exploración y preprocesamiento, entrenar modelos
de aprendizaje supervisado y no supervisado, comparar su desempeño y
analizar la latencia de inferencia.

Se utilizaron modelos de clasificación supervisada y algoritmos de
clustering para identificar patrones presentes en las lecturas de los
sensores.

## Estructura del repositorio

```text
ml-practica1-pesantez/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── kmeans.pkl
│   └── dbscan.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_supervised.ipynb
│   └── 03_unsupervised.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── train_supervised.py
│   └── train_unsupervised.py
│
├── benchmark_latency.json
├── requirements.txt
└── README.md

## Instalación y ejecución

### 1. Crear y activar el entorno virtual

```bash
python -m venv venv
```

### 2. Activar el entorno virtual

En Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el modelo supervisado

Desde la carpeta raíz del proyecto:

```bash
python src/train_supervised.py
```

Este script:

- Carga el dataset original.
- Separa las variables de entrada y la variable objetivo.
- Divide los datos en entrenamiento y prueba.
- Aplica `RobustScaler`.
- Entrena un modelo Random Forest.
- Entrena un modelo SVM.
- Calcula Accuracy, F1-score y matriz de confusión.
- Compara el tiempo de entrenamiento de ambos modelos.
- Guarda los modelos entrenados en la carpeta `models/`.

### 5. Ejecutar los modelos no supervisados

```bash
python src/train_unsupervised.py
```

Este script:

- Carga los datos procesados.
- Utiliza las 24 variables de los sensores.
- Determina el número de clusters para K-Means mediante el método del codo.
- Entrena K-Means.
- Evalúa el agrupamiento mediante Silhouette Score.
- Entrena DBSCAN.
- Prueba diferentes valores de `eps`.
- Analiza el número de clusters y puntos considerados como ruido.
- Utiliza PCA para visualizar los clusters.
- Guarda los modelos entrenados en la carpeta `models/`.

### 6. Notebooks

La carpeta `notebooks/` contiene los análisis realizados durante la práctica:

- `01_eda.ipynb` — Exploración y análisis inicial de los datos.
- `02_supervised.ipynb` — Modelos de aprendizaje supervisado.
- `03_unsupervised.ipynb` — Modelos de aprendizaje no supervisado.

Los notebooks pueden abrirse directamente desde Visual Studio Code utilizando el entorno virtual del proyecto como kernel de Python.

## Resultados principales

### Aprendizaje supervisado

| Modelo | Accuracy | F1-score |
|---|---:|---:|
| Random Forest | 0.9954 | 0.9954 |
| SVM | 0.8690 | 0.8684 |

Random Forest obtuvo el mejor desempeño en la clasificación del conjunto de prueba.

### Aprendizaje no supervisado

| Modelo | Clusters | Ruido | Silhouette |
|---|---:|---:|---:|
| K-Means | 4 | 0 | 0.4702 |
| DBSCAN | 19 | 107 | 0.4052 |

Para K-Means se seleccionó `k = 4`.

Para DBSCAN se utilizó finalmente:

```text
eps = 5.0
min_samples = 5
```

### Estructura de resultados

Los modelos entrenados se almacenan en:

```text
models/
├── random_forest.pkl
├── svm.pkl
├── kmeans.pkl
└── dbscan.pkl
```

El benchmark de latencia se almacena en:

```text
benchmark_latency.json
```

## Tecnologías utilizadas

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy
- Joblib
- PyArrow
- Jupyter Notebook

## Autor

Práctica 1 - Machine Learning