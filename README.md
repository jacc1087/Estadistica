# 🏀 Estadística para Data Science — Análisis de la NBA

Aplica análisis descriptivo, inferencia estadística, regresión (lineal y logística) y análisis de series temporales sobre un dataset histórico de estadísticas de jugadores de la **NBA** (desde 1947), además de una implementación de regresión lineal **desde cero con NumPy** para demostrar la comprensión matemática detrás de `LinearRegression()`.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jacc1087/Estadistica/blob/main/PracticaEstadistica__Contreras_Ca%C3%B1o_Jose_Angel.ipynb) &nbsp; [![Abrir la demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://estadistica-qjdzinvequiatnzbpnfrrz.streamlit.app/)

---

## 📑 Tabla de contenidos

- [Descripción general](#-descripción-general)
- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Cómo verlo](#-cómo-verlo)
- [Metodología y hallazgos](#-metodología-y-hallazgos)
- [Tecnologías](#-tecnologías)
- [Autor](#-autor)

---

## 📖 Descripción general

Como aficionado a la NBA, el proyecto utiliza un dataset con estadísticas históricas de todos los jugadores de la liga desde 1947 para practicar el ciclo completo de un análisis estadístico aplicado a Data Science, dividido en 4 partes:

1. **Análisis descriptivo** — Exploración, limpieza y clasificación de variables de un dataset propio (NBA).
2. **Inferencia y modelado** — Correlaciones, relaciones bivariantes, regresión lineal y logística sobre el mismo dataset.
3. **Regresión lineal "from scratch"** — Implementación manual con NumPy (sin Scikit-learn) para demostrar el fundamento matemático (OLS, MSE, R²), comparada después con el modelo real de Scikit-learn.
4. **Series temporales** — Análisis de tendencia, estacionalidad y suavizado (medias móviles) sobre una serie simulada.

## 📊 Dataset

**Fuente:** [NBA/ABA/BAA Stats — Kaggle](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats?select=Player+Season+Info.csv)

Del dataset original se utilizó específicamente el archivo `PlayersTotal.csv`, renombrado en este repositorio como **`nba_players.csv`**, que contiene las estadísticas totales por temporada de cada jugador.

**Limpieza aplicada:**
- Filtrado a jugadores de la liga NBA con un mínimo de 500 minutos jugados, para asegurar relevancia estadística.
- Eliminación de columnas poco informativas para el estudio (temporada, liga, id, edad, equipo, partidos titular).
- Agregación por jugador: estadísticas acumuladas (suma) para volumen de juego, y medias para porcentajes de acierto.
- Corrección de tipos de datos (float → int en columnas de conteo).
- Eliminación de registros nulos en `3p_percent` — antes de 1979 no existía el triple en la NBA, por lo que esos jugadores no tienen este dato y se excluyen para no distorsionar el análisis.

## 📂 Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `PracticaEstadistica__Contreras_Caño_Jose_Angel.ipynb` | Notebook principal con todo el análisis y el código |
| `nba_players.csv` | Dataset utilizado (ver [Dataset](#-dataset)) |
| `app_estadistica.py` | Dashboard interactivo desplegado en Streamlit (ver [Demo](#-cómo-verlo)) |
| `requirements.txt` | Dependencias para ejecutar el dashboard |

## 👀 Cómo verlo

Hay dos formas de acceder al proyecto:

1. **🚀 Demo interactiva (recomendada)** — pulsa el botón "Abrir la demo" de arriba. Es un dashboard en Streamlit con filtros (posición, partidos jugados), gráficas interactivas, un modelo de regresión que puedes probar en vivo eligiendo tú las variables, y un buscador de jugadores. No requiere instalar nada ni ver código.
2. **📓 Notebook en Colab** — pulsa "Abrir en Colab" para ejecutar el análisis completo (las 4 partes) paso a paso, con el código a la vista. El CSV se descarga automáticamente, no requiere configuración.

Para ejecutarlo en tu propio ordenador en vez de en Colab:

```bash
git clone https://github.com/jacc1087/Estadistica.git
cd Estadistica
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
jupyter notebook PracticaEstadistica__Contreras_Caño_Jose_Angel.ipynb
```

## 🔍 Metodología y hallazgos

### 1. Análisis descriptivo
Clasificación de variables (nominales, discretas, continuas), estadísticos descriptivos (media vs. mediana) y detección de outliers con boxplots e IQR. La variable `points` muestra un fuerte sesgo hacia la derecha (la mayoría de jugadores no supera los 2.500 puntos en su carrera), mientras que `effective_percent` sigue una distribución mucho más cercana a la normal.

### 2. Correlación y relaciones bivariantes
Matriz de correlación con Heatmap sobre dos variables objetivo: `points` y `effective_percent`.
- **Points** correlaciona fuertemente con el volumen de juego (partidos, minutos) — lógico, a más tiempo en pista, más puntos.
- **Effective_percent** correlaciona sobre todo con el porcentaje de acierto en tiros de 2 puntos, no con el de triples, lo que explica por qué los jugadores más efectivos históricamente son pívots con tiros cercanos al aro.

### 3. Regresión lineal y logística
- **Regresión lineal simple** (minutos jugados → puntos): R² = 0.887, un ajuste muy bueno.
- **Regresión lineal múltiple** (% tiro de 2, 3 y libres → efectividad): R² = 0.831, RMSE = 0.0215.
- **Regresión logística** (¿es un jugador eficiente en triples, >35%?): Accuracy = 0.87.

### 4. Regresión lineal "from scratch"
Implementación manual del algoritmo OLS con NumPy, con cálculo manual de MSE y R² (sin Scikit-learn), comparada frente al modelo real de `sklearn.linear_model.LinearRegression` sobre los mismos datos simulados — los resultados convergen, validando la implementación propia.

### 5. Series temporales
Sobre una serie simulada con tendencia, estacionalidad y ruido: resampleo mensual, medias móviles con distintas ventanas (comparando el efecto de suavizado) y descomposición estacional con `statsmodels.tsa.seasonal_decompose`.

## 🛠️ Tecnologías

- **Lenguaje:** Python (Jupyter Notebook)
- **Análisis de datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn (notebook) · Plotly (demo interactiva)
- **Machine Learning:** Scikit-learn (regresión lineal y logística)
- **Series temporales:** Statsmodels
- **Demo:** Streamlit
