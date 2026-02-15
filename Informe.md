# Informe – Taller 4: Aprendizaje automático escalable con Spark ML  
**Diplomado en Gestión de Datos – Universidad Santo Tomás**

**Presentado por:**  
- Ingrid Melissa Céspedes Díaz  
- María Fernanda Torres Ortiz  

## Descripción
Taller práctico basado en retos para aprender **Machine Learning distribuido con Spark ML** y fundamentos de **MLOps** con **MLflow**, usando datos reales de contratos públicos de Colombia Compra Eficiente (**SECOP II**).

## Objetivos de aprendizaje
- Escalar algoritmos de ML a conjuntos de datos grandes usando Spark MLlib.  
- Construir pipelines reproducibles con Transformers y Estimators.  
- Implementar regresión lineal, logística y regularización.  
- Aplicar Grid Search y validación cruzada para tuning.  
- Gestionar el ciclo de vida de modelos con MLflow (experimentos, métricas, artefactos).

## Dataset – SECOP II (Contratos electrónicos)
**Fuente:** Datos Abiertos Colombia – SECOP II.  
Contratos electrónicos del Sistema Electrónico para la Contratación Pública de Colombia

## Entorno de ejecución (Docker + Spark + MLflow)
Se trabajó con un entorno contenerizado para garantizar reproducibilidad:

- **Spark Master/Worker** (ejecución distribuida)  
- **Jupyter Lab** (ejecución de notebooks)  
- **MLflow Tracking Server** (tracking de experimentos)


## Fase 1: Ingesta y Exploracion

# Notebook 01: Ingesta de Datos
En este primer notebook se realizó el proceso de ingesta y preparación inicial de los datos. Para ello, se configuró una **SparkSession** conectada al clúster (`spark://spark-master:7077`) con el fin de trabajar en un entorno distribuido. Posteriormente, se descargaron **100.000 registros** del dataset de **SECOP II** a través de la API de Datos Abiertos (Socrata), aplicando un filtro por fecha de firma correspondiente al año 2025.

La información obtenida se almacenó en formato **JSON línea por línea** en la ruta `/opt/spark-data/raw/secop_contratos.json`. A continuación, este archivo fue leído con Spark, se exploró su esquema y se normalizaron los nombres de las columnas para facilitar su uso posterior. También se realizó el casteo de tipos de datos, convirtiendo fechas a formato *timestamp*, montos a *double* y variables de duración a tipo *entero*, además de seleccionar únicamente las columnas relevantes para el análisis y el modelado.

Como parte del proceso de enriquecimiento, se generaron las columnas **year** y **month** a partir de la variable *fecha_de_firma*, lo que permitió particionar los datos temporalmente. Finalmente, el resultado se guardó en formato **Parquet**, obteniendo un dataset optimizado en la ruta `/opt/spark-data/processed/secop_contratos_silver.parquet`, con una validación final de **100.000 registros y 13 columnas**, listo para las siguientes fases del proyecto.

# Notebook 02: Analisis Exploratorio (EDA)

En este segundo notebook se cargó el archivo **Parquet** generado en el proceso de ingesta (Notebook 01) y se realizó un análisis exploratorio para entender la estructura y el comportamiento de los datos. En primer lugar, se calcularon estadísticas descriptivas generales utilizando funciones como `describe()` y métricas específicas sobre la variable objetivo **valor_del_contrato**, obteniendo valores mínimos, máximos, promedio y desviación estándar.

Posteriormente, se evaluó la **calidad de los datos**, analizando valores nulos en cada columna. Como resultado, se identificó que la variable **duraci_n_del_contrato** presentaba un **100% de valores nulos**, por lo que se descartó para los análisis posteriores. También se exploró la distribución de la variable **valor_del_contrato**, evidenciando una alta dispersión y presencia de valores extremos.

Adicionalmente, se realizaron análisis agregados por **departamento**, encontrando que Bogotá y Antioquia concentran la mayor cantidad y valor total de contratos. De igual forma, se analizó la distribución por **tipo de contrato**, donde predominan ampliamente los contratos de *prestación de servicios*, así como por **estado del contrato**, siendo los más frecuentes “En ejecución” y “Modificado”. También se identificaron los **principales proveedores** según número de contratos y valor acumulado.

Finalmente, se llevó a cabo una **detección de outliers** utilizando el método del rango intercuartílico (IQR) con cuantiles aproximados, encontrando que aproximadamente el **6.76%** de los contratos pueden considerarse valores atípicos. Como complemento, se realizó un análisis temporal por año y mes, se generaron gráficos de apoyo y se guardó un dataset procesado para ser utilizado en las siguientes etapas del proyecto.

**Qué se obtuvo**
- Gráfico guardado en:
  - `/opt/spark-data/processed/eda_departamentos.png`
- Dataset EDA guardado en:
  - `/opt/spark-data/processed/secop_eda.parquet`
- Hallazgos clave:
  - fuerte concentración en *Prestación de servicios*,
  - outliers relevantes (~6.76% por IQR),
  - columna de duración sin datos útiles (nula).
  - `/opt/spark-data/processed/eda_departamentos.png`
<img width="1202" height="598" alt="image" src="https://github.com/user-attachments/assets/bdf06b55-da88-437f-8dfe-8b626b2cd34c" />


## Fase 2: Feature Engineering (Seccion 13)

# Notebook 03: Pipelines y Feature Engineering

En este notebook se cargó el dataset resultante del análisis exploratorio (EDA) con el objetivo de preparar los datos para el entrenamiento de modelos de Machine Learning. En primer lugar, se definieron las variables a utilizar como **features**, seleccionando como categóricas `tipo_de_contrato`, `departamento` y `ciudad`, y como numéricas `anio` y `mes`. La variable objetivo o **label** se definió como `valor_del_contrato_num`, correspondiente al valor del contrato en formato numérico.

Posteriormente, se aplicó una estrategia de **limpieza de datos** basada en la eliminación de registros con valores nulos mediante `dropna`, asegurando que tanto las features como la variable objetivo estuvieran completas. Esto permitió mantener un dataset consistente para el entrenamiento de los modelos.

A continuación, se construyó un **pipeline de transformación** siguiendo el orden correcto de dependencias. Primero se utilizó `StringIndexer` para convertir las variables categóricas en índices numéricos, luego `OneHotEncoder` para transformar dichos índices en vectores binarios, y finalmente `VectorAssembler` para combinar las variables numéricas y categóricas codificadas en un único vector de entrada llamado `features_raw`. Además, la variable objetivo fue renombrada a `label`, cumpliendo con el formato estándar requerido por Spark ML.

Como parte del análisis adicional, se calculó la **dimensión final del vector de features** después del proceso de codificación, evidenciando el aumento significativo de variables debido al One-Hot Encoding. También se realizó un análisis de **varianza de las features** sobre una muestra del dataset, con el fin de identificar cuáles variables presentan mayor variabilidad y, por tanto, mayor potencial informativo para el modelo.

**Qué se obtuvo**
- Vector final de features:
  - **573 dimensiones** (2 numéricas + 571 one-hot).
- Pipeline entrenado guardado en:
  - `/opt/spark-data/processed/feature_pipeline`
- Dataset listo para modelos guardado en:
  - `/opt/spark-data/processed/secop_features.parquet`

## Fase 3: Modelos de Regresion

# Notebook 06: Regresion Logistica

En este notebook se cargó el dataset previamente transformado con las features ya vectorizadas y se aplicó un proceso de **estandarización** sobre la columna de entrada `features`, con el fin de garantizar que todas las variables tuvieran la misma escala antes del entrenamiento del modelo.

Posteriormente, se definió una variable objetivo binaria llamada **riesgo**, utilizando un enfoque de tipo *proxy*. Para ello, se asignó `riesgo = 1` a los contratos cuyo `valor_del_contrato` se encontraba por encima del **percentil 90**, mientras que se asignó `riesgo = 0` al resto de los registros. Este criterio permitió convertir el problema original en uno de **clasificación binaria** orientado a la detección de contratos de alto valor.

A continuación, se analizó la distribución de la variable objetivo, evidenciando un **fuerte desbalance de clases**, con aproximadamente un 89% de observaciones en la clase 0 y un 11% en la clase 1. Posteriormente, el dataset fue dividido en conjuntos de entrenamiento y prueba utilizando una proporción **70/30**.

Con los datos preparados, se entrenó un modelo de **Regresión Logística** (`LogisticRegression`) utilizando regularización **L2** para evitar sobreajuste. Además, se configuró un **threshold de 0.3** en lugar del valor por defecto, con el objetivo de priorizar la detección de la clase minoritaria (riesgo = 1), aumentando la sensibilidad del modelo frente a contratos de alto riesgo.

Finalmente, el modelo fue evaluado mediante múltiples métricas, incluyendo **AUC-ROC**, accuracy, precision, recall y F1 ponderado, así como métricas específicas para la **clase 1**. También se construyó la **matriz de confusión**, permitiendo analizar de forma detallada los aciertos y errores de clasificación.

Como análisis adicional (*bonus*), se construyó y se **guardó la curva ROC**, lo que permitió visualizar gráficamente el desempeño del modelo y su capacidad de discriminación entre contratos de alto y bajo riesgo.

**Qué se obtuvo**
- AUC-ROC aproximado: **0.7976**
- Matriz de confusión (con threshold 0.3):
  - TP=215, FP=157, FN=3171, TN=26450
- Curva ROC guardada en:
  - `/opt/spark-data/processed/roc_curve.png`
- Modelo guardado en:
  - `/opt/spark-data/processed/logistic_regression_model`
  - `/opt/spark-data/processed/roc_curve.png`
<img width="710" height="544" alt="image" src="https://github.com/user-attachments/assets/3d37dc12-edde-419e-851f-cfe1fdb84661" />

---


# Fase 5: MLOps y Produccion

## Notebook 10: MLflow Tracking

En este notebook se configuró un **MLflow Tracking Server** para gestionar los experimentos de Machine Learning, estableciendo como endpoint `http://mlflow:5000` y creando el experimento llamado `/SECOP_Contratos_Prediccion`. Esto permitió centralizar el registro de métricas, parámetros y modelos entrenados durante el proceso.

Posteriormente, se cargó el dataset de features correspondiente al problema de **regresión** y se entrenó un primer modelo base utilizando **LinearRegression sin regularización**, el cual sirvió como línea base (*baseline*) para comparar el desempeño de los modelos posteriores. En este primer run se registraron los parámetros del modelo, las métricas de evaluación y el modelo como artefacto.

A continuación, se ejecutaron tres experimentos adicionales utilizando modelos regularizados: **Ridge (L2)**, **Lasso (L1)** y **ElasticNet**. Para cada uno de estos modelos se registraron métricas de desempeño como **RMSE**, **MAE** y **R²**, además de almacenar los modelos entrenados como artefactos dentro de MLflow.

Finalmente, se compararon todos los runs directamente desde la interfaz gráfica de MLflow en `localhost:5000`, lo que permitió analizar visualmente las diferencias de desempeño entre los distintos enfoques. Como complemento, se creó un run adicional que incluyó **artefactos personalizados**, entre ellos un archivo de reporte en texto (`model_report.txt`) con un resumen del experimento y sus resultados.
  
  - gráfico real vs predicho (`predictions_vs_real.png`)

**Qué se obtuvo**
- Experimento con runs comparables en MLflow UI.
- Mejor RMSE reportado (en tu UI): **Lasso (l1)** con ~**3,030,748,798.68**
- Artefactos guardados en el run:
  - `model_report.txt`
  - `predictions_vs_real.png`

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/cecbf409-60aa-457f-b5b0-bffcfff0b1ab" />


### 📌 Resultado final del notebook
- Se configuró un tracking server de MLflow y se creó un experimento.
- Se registró un baseline y tres runs con regularización (Ridge/Lasso/ElasticNet).
- Se compararon runs en la UI y se identificó el “mejor” por RMSE.
- Se agregó trazabilidad completa (parámetros, métricas y artefactos) para auditoría y colaboración.

## Conclusiones 

### Conclusiones principales
- **Spark ML permitió escalar el flujo completo de ML** (ingesta → EDA → feature engineering → entrenamiento/evaluación) sobre un volumen grande de datos (100.000 contratos), manteniendo el procesamiento distribuido y reproducible.
- **El Pipeline de Spark ML facilitó la reproducibilidad**: al encadenar `StringIndexer → OneHotEncoder → VectorAssembler`, se garantizó que las mismas transformaciones se apliquen de forma consistente en entrenamiento y en producción.
- **El EDA evidenció alta dispersión y presencia de outliers** en `valor_del_contrato`, lo cual afecta fuertemente métricas como RMSE y sugiere considerar transformaciones (por ejemplo `log(label)`) o modelos más robustos.
- **La regresión logística funcionó como clasificación de “riesgo” usando un proxy (percentil 90)**. Aunque el AUC-ROC fue bueno (~0.80), el **recall de la clase 1** fue bajo, confirmando el impacto del desbalance y la necesidad de ajustes (threshold, pesos de clase o técnicas de muestreo).
- **En los modelos de regresión con regularización (Ridge/Lasso/ElasticNet)** las métricas quedaron prácticamente iguales y el **R² fue muy bajo (~0.0029)**. Esto indica que, con las features actuales, el modelo explica poca variación del valor del contrato (o que la señal predictiva es débil con esas variables).
- **MLflow aportó valor clave de MLOps**: tracking centralizado de runs, comparación en UI, versionado de modelos y almacenamiento de artefactos (reporte + gráfico), lo que mejora trazabilidad y trabajo en equipo frente a guardar resultados “a mano” en CSV.

### Limitaciones identificadas
- La variable objetivo de regresión (`valor_del_contrato`) tiene **outliers fuertes** y distribución sesgada, lo que puede inflar el RMSE.
- Las features usadas (tipo de contrato + ubicación + tiempo) pueden ser **insuficientes** para explicar el valor contractual (R² bajo).
- En clasificación, el criterio de “riesgo” es un **proxy** (valor alto ≠ incumplimiento real). Para un modelo operativo real se requerirían etiquetas de incumplimiento, sanciones, adiciones, terminaciones anómalas, etc.

### Recomendaciones / siguientes pasos
- **Mejorar features**:
  - incluir más variables del SECOP (modalidad, sector, entidad centralizada, orden, proveedor, etc.)
  - crear interacciones (ej. `tipo_contrato × departamento`)
  - engineering temporal (trimestres, fin de año, antigüedad, etc.)
- **Transformar la variable objetivo** en regresión:
  - entrenar con `log1p(valor_del_contrato)` y luego revertir con `expm1()` para reducir el efecto de outliers.
- **Probar modelos alternativos**:
  - árboles (DecisionTree / RandomForest / GBTRegressor) para no linealidad.
- **En clasificación**:
  - probar `weightCol` para penalizar más la clase minoritaria,
  - explorar thresholds para optimizar recall/F1 de la clase 1,
  - evaluar también PR-AUC si el desbalance aumenta.
- **Operacionalizar MLOps**:
  - estandarizar nombres de experimentos/runs,
  - automatizar tuning con loops o GridSearch + MLflow,
  - registrar el mejor modelo en un Model Registry (si aplica).


