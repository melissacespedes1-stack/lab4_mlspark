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
**Qué se hizo**
- Se configuró una `SparkSession` conectada al clúster (`spark://spark-master:7077`).
- Se descargaron **100.000 registros** del dataset SECOP II desde la API de Datos Abiertos (Socrata), filtrando por fecha de firma (2025).
- Se guardó la descarga como **JSON line-by-line** en `/opt/spark-data/raw/secop_contratos.json`.
- Se leyó el JSON con Spark, se exploró el esquema y se normalizaron nombres de columnas.
- Se castearon tipos (fechas a `timestamp`, montos a `double`, duraciones/días a `int`) y se seleccionaron **columnas clave** para ML.
- Se generaron columnas `year` y `month` desde `fecha_de_firma` y se guardó el resultado en **Parquet particionado**.

**Qué se obtuvo**
- Dataset “silver” optimizado en:
  - `/opt/spark-data/processed/secop_contratos_silver.parquet`
- Validación final: **100.000 registros** y **13 columnas** (incluye particiones `year`, `month`).

# Notebook 02: Analisis Exploratorio (EDA)

**Qué se hizo**
- Se cargó el Parquet generado en el Notebook 01.
- Se calcularon estadísticas descriptivas generales (`describe()` + métricas de `valor_del_contrato`).
- Se analizó **calidad / nulos**: se detectó que `duraci_n_del_contrato` estaba **100% nulo**.
- Se exploró la variable objetivo (`valor_del_contrato`) y su dispersión.
- Se analizaron agregaciones por:
  - **departamento** (top por cantidad y por valor total),
  - **tipo de contrato**,
  - **estado del contrato**,
  - **proveedores** (top).
- Se detectaron **outliers** con método IQR usando cuantiles aproximados.
- Bonus: análisis temporal por año/mes.
- Se generó un gráfico y se guardó un dataset de salida para siguientes pasos.

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

**Qué se hizo**
- Se cargó el dataset de EDA.
- Se definieron features:
  - **categóricas**: `tipo_de_contrato`, `departamento`, `ciudad`
  - **numéricas**: `anio`, `mes`
  - **label**: `valor_del_contrato_num`
- Se aplicó estrategia de limpieza con `dropna` (manteniendo dataset completo).
- Se construyó un pipeline con stages en el orden correcto:
  1) `StringIndexer`  
  2) `OneHotEncoder`  
  3) `VectorAssembler` → `features_raw`
- Se transformó el dataset y se renombró label a `label` (formato estándar Spark ML).
- Bonus:
  - se calculó la dimensión final del vector (post-encoding),
  - se estimó varianza de features en una muestra para ver cuáles “varían más”.

**Qué se obtuvo**
- Vector final de features:
  - **573 dimensiones** (2 numéricas + 571 one-hot).
- Pipeline entrenado guardado en:
  - `/opt/spark-data/processed/feature_pipeline`
- Dataset listo para modelos guardado en:
  - `/opt/spark-data/processed/secop_features.parquet`

## Fase 3: Modelos de Regresion

# Notebook 06: Regresion Logistica

**Qué se hizo**
- Se cargó el dataset con features y se estandarizó la columna de entrada (`features`).
- Se definió variable objetivo binaria **“riesgo”** usando un proxy:
  - **riesgo=1** si `valor_del_contrato` > **percentil 90**
  - **riesgo=0** en caso contrario
- Se analizó el **desbalance de clases** (~89% vs ~11%).
- Se dividió en train/test (70/30).
- Se entrenó un modelo de `LogisticRegression` con:
  - regularización L2,
  - `threshold=0.3` para priorizar detección de clase 1.
- Se evaluó con múltiples métricas:
  - AUC-ROC, accuracy, precision/recall/F1 (weighted),
  - métricas específicas de la **clase 1**,
  - matriz de confusión.
- Bonus:
  - se construyó y guardó la curva ROC.

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

**Qué se hizo**
- Se configuró MLflow Tracking Server:
  - `mlflow.set_tracking_uri("http://mlflow:5000")`
  - experimento: `/SECOP_Contratos_Prediccion`
- Se cargó el dataset de features para **regresión** y se entrenó un baseline:
  - `LinearRegression` sin regularización
  - log de parámetros, métrica y modelo
- Se registraron 3 experimentos adicionales:
  - **Ridge (L2)**, **Lasso (L1)**, **ElasticNet**
  - métricas registradas: RMSE, MAE, R²
  - modelos registrados como artefactos
- Se compararon runs en la UI de MLflow (localhost:5000).
- Se creó un run con **artefactos personalizados**:
  - reporte en texto (`model_report.txt`)
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


