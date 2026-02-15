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

## Estructura del Proyecto

lab4_mlspark/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_ingesta_datos.py
│   ├── 02_exploracion_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_transformaciones.py
│   ├── 05_regresion_lineal.py
│   ├── 06_regresion_logistica.py
│   ├── 07_regularizacion.py
│   ├── 08_validacion_cruzada.py
│   ├── 09_optimizacion_hiperparametros.py
│   ├── 10_mlflow_tracking.py
│   ├── 11_model_registry.py
│   └── 12_inferencia_produccion.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_utils.py
│   └── model_trainer.py
│
└── mlruns/

## Fase 1: Ingesta y Exploracion

# Notebook 01: Ingesta de Datos
En el **Notebook 01: Ingesta de Datos**, se completó la fase de ingesta y preparación inicial del dataset **SECOP II** para dejarlo listo en formato optimizado (**Parquet**) para las siguientes etapas (EDA y ML). A continuación, el resumen **por reto**:

---

#### ✅ Reto 1: Configurar SparkSession conectada al cluster
- Se creó una **SparkSession** con nombre de aplicación **`SECOP_Ingesta`** conectada al clúster:
  - `master("spark://spark-master:7077")`
- Se configuró memoria para mejorar estabilidad:
  - `spark.executor.memory = 2g`
  - `spark.driver.memory = 1g`
- Se validó la conexión imprimiendo:
  - **Spark Version: 3.5.0**
  - **Spark Master: spark://spark-master:7077**

---

#### ✅ Reto 2: Descargar datos desde la API de Datos Abiertos Colombia
- Se consumió la API **Socrata** (`www.datos.gov.co`) usando `sodapy`.
- Se descargaron **100,000 registros** del dataset **`jbjy-vk9h`** aplicando filtro:
  - `fecha_de_firma` entre **2025-01-01** y **2026-01-01**
- Se guardó la descarga como **JSON Lines** en:
  - `/opt/spark-data/raw/secop_contratos.json`
- Se verificó la descarga con:
  - conteo de registros descargados
  - listado de columnas recibidas (87 campos)

---

#### ✅ Reto 3: Cargar datos en Spark y explorar el esquema
- Se cargó el archivo JSON con:
  - `spark.read.json(json_path)`
- Se validó el contenido:
  - **100,000 registros**
  - **87 columnas**
- Se exploró el dataset con:
  - `df_raw.printSchema()` para revisar tipos y estructura
- Se normalizaron nombres de columnas (quitar tildes y caracteres especiales, usar `_`, minúsculas) para evitar errores posteriores.
- Se realizó tipificación en una capa “silver”:
  - campos con “fecha” → `timestamp`
  - campos con “valor/saldo” → `double`
  - campos con “dias/duracion/plazo” → `int`
- Se manejó el campo anidado `urlproceso` convirtiéndolo a `url_proceso`.

---

#### ✅ Reto 4: Seleccionar columnas clave para ML
- Se definió un subconjunto de **columnas clave** relevantes para modelado:
  - `referencia_del_contrato`, `nit_entidad`, `nombre_entidad`, `departamento`, `ciudad`,
  - `tipo_de_contrato`, `valor_del_contrato`, `fecha_de_firma`,
  - `duraci_n_del_contrato`, `proveedor_adjudicado`, `estado_contrato`
- Se creó el DataFrame final `df_clean` con **11 columnas**.
- Se generaron variables temporales derivadas:
  - `year` y `month` a partir de `fecha_de_firma`
- Se validó calidad básica del dataset (campos críticos no nulos):
  - **Registros inválidos: 0**
  - **Calidad: 100%**

---

#### ✅ Reto 5: Guardar en formato Parquet optimizado
- Se guardó el dataset en formato **Parquet** (más eficiente para Spark que JSON) en:
  - `/opt/spark-data/processed/secop_contratos_silver.parquet`
- Se aplicó particionado para optimizar lecturas por tiempo:
  - `partitionBy("year", "month")`
- Se verificó la escritura leyendo de nuevo el Parquet:
  - **100,000 registros**
  - **13 columnas** (11 + year + month)
- Se imprimieron schema y muestra final para confirmar integridad.

---

### Resultado final del Notebook 01
Se dejó un dataset “silver” limpio, tipado y optimizado en **Parquet particionado** listo para:
- **Notebook 02 (EDA)**
- **Notebook 03–05 (Feature Engineering + Modelos de regresión)**  
- y el flujo de ML/MLOps posterior con MLflow.

# Notebook 02: Analisis Exploratorio (EDA)

En este notebook se realizó el **Análisis Exploratorio de Datos (EDA)** sobre el dataset **SECOP II** previamente ingestado y guardado en Parquet (capa *silver*). El objetivo fue comprender la estructura, calidad, distribución de variables y detectar outliers, generando además insumos (dataset y gráficos) para los notebooks posteriores.

---

### ✅ Reto 1: Calcular estadísticas descriptivas
- Se cargó el dataset desde:
  - `/opt/spark-data/processed/secop_contratos_silver.parquet`
- Se validó el tamaño del conjunto:
  - **100,000 registros** y **13 columnas**
- Se revisó el esquema con `printSchema()` y se visualizaron filas iniciales con `show()`.
- Se calcularon estadísticas globales con `df.describe()` (conteo, media, desviación estándar, min y max).
  - Hallazgo relevante: `duraci_n_del_contrato` aparece con **count=0** (sin valores no nulos).

---

### ✅ Reto 2: Analizar valores nulos y decidir estrategia
- Se calculó el número y porcentaje de nulos por columna (incluyendo `isnan` en numéricas).
- Resultado:
  - `duraci_n_del_contrato` tuvo **100% nulos** (100,000 nulos).
- **Decisión/estrategia:**  
  - Debido a que la columna no aporta información (100% nula), se recomendó **excluirla del modelado** o imputarla solo si se obtiene de otra fuente (no aplicable aquí).

---

### ✅ Reto 3: Explorar la variable objetivo (valor de contratos)
- Se identificó automáticamente la columna de valor (`valor_del_contrato`) y se creó:
  - `valor_del_contrato_num` (cast a `double`)
- Se calcularon métricas descriptivas:
  - **Min:** 0.0  
  - **Max:** 4.08987300811e11  
  - **Promedio:** 1.0809235143475e8  
  - **Desv. Std:** 2.4064589461934705e9  
- Conclusión: la variable tiene **alta dispersión** y presencia de valores extremos.

---

### ✅ Reto 4: Analizar distribución por departamento
- Se agruparon registros por `departamento` y se calculó:
  - `num_contratos` (conteo)
  - `valor_total` (suma del valor del contrato)
- **Top 10 por número de contratos**:
  1. Distrito Capital de Bogotá (34,342)
  2. Antioquia (15,200)
  3. Valle del Cauca (11,391)
  4. Cundinamarca (6,463)
  5. Boyacá (3,119)
  ...
- Se generó y guardó un gráfico con:
  - Top 10 por número de contratos
  - Top 10 por valor total (en miles de millones COP)
- Archivo generado:
  - `/opt/spark-data/processed/eda_departamentos.png`
<img width="1202" height="598" alt="image" src="https://github.com/user-attachments/assets/bdf06b55-da88-437f-8dfe-8b626b2cd34c" />

---

### ✅ Reto 5: Explorar tipo de contrato y estado
- **Tipo de contrato (Top 10):**
  - `Prestación de servicios` domina ampliamente (**94,836** contratos).
  - Le siguen `Decreto 092 de 2017` (3,089), `Otro` (764), etc.
- **Estado del contrato:**
  - En ejecución (40,896)
  - Modificado (34,492)
  - terminado (11,621)
  - Cerrado (11,068)
  - Otros estados con menor frecuencia.

---

### ✅ Reto 6: Detectar outliers con método IQR
- Se calcularon percentiles aproximados para `valor_del_contrato_num`:
  - Q1 (25%): **$16,800,000**
  - Q2 (50%): **$31,549,600**
  - Q3 (75%): **$56,826,002**
  - P95 (95%): **$124,372,000**
  - P99 (99%): **$408,987,300,811**
- Con el método **IQR**:
  - IQR = Q3 - Q1
  - Límites: **$-43,239,003** a **$116,865,005**
- Outliers detectados:
  - **6,761** contratos (**6.76%**)

---

### ⭐ Bonus: Análisis temporal de contratos
- Se confirmó la variable temporal `fecha_de_firma` y se creó:
  - `fecha_de_firma_parsed`, `anio`, `mes`
- Resultado:
  - Todos los registros pertenecen a **2025** (por el filtro aplicado en la ingesta).

---

### Salidas del Notebook 02
- **Dataset de EDA guardado** para siguientes notebooks:
  - `/opt/spark-data/processed/secop_eda.parquet`
- **Gráfico generado**:
  - `/opt/spark-data/processed/eda_departamentos.png`
- Resumen del análisis:
  - Total registros analizados: **100,000**
  - Valor total contratos: **$10,809,235,143,475.00**
  - Outliers (IQR): **6.76%**
 
## Fase 2: Feature Engineering (Seccion 13)

# Notebook 03: Pipelines y Feature Engineering

### ✅ Reto 1: Seleccionar features categóricas y numéricas
Se definieron y validaron las variables de entrada para predecir el **valor del contrato**:

- **Categóricas (contexto y tipo):**
  - `tipo_de_contrato`
  - `departamento`
  - `ciudad`

- **Numéricas (componente temporal):**
  - `anio`
  - `mes`

- **Variable objetivo (label):**
  - `valor_del_contrato_num`

Se verificó que todas existieran en el dataset (`available_cat` y `available_num`), confirmando disponibilidad completa.

---

### ✅ Reto 2: Implementar estrategia de limpieza de datos
Se aplicó una estrategia conservadora para garantizar consistencia del entrenamiento:

- Se aseguraron tipos correctos en variables clave (cast a `int`/`double` cuando aplica).
- Se eliminaron registros con valores nulos en features o label usando:
  - `df.dropna(subset=available_cat + available_num + [label_col])`

**Resultado:**  
- Registros después de limpieza: **100,000** (no se perdieron filas en las columnas seleccionadas).

**Decisión justificada:**  
Dado el tamaño del dataset y para evitar ruido por imputación (especialmente en regresión), se optó por entrenar solo con datos completos.

---

### ✅ Reto 3: Crear VectorAssembler para combinar features
Como Spark ML requiere una única columna vectorial de entrada, se construyó:

- `StringIndexer` para convertir strings a índices:
  - `tipo_de_contrato_idx`, `departamento_idx`, `ciudad_idx`

- `OneHotEncoder` para codificar categorías:
  - `tipo_de_contrato_vec`, `departamento_vec`, `ciudad_vec`

- `VectorAssembler` para unir todo en un solo vector:
  - Entrada: `['anio', 'mes', 'tipo_de_contrato_vec', 'departamento_vec', 'ciudad_vec']`
  - Salida: `features_raw`

---

### ✅ Reto 4: Construir Pipeline completo (orden correcto de stages)
Se construyó un **Pipeline** con el orden correcto por dependencias:

1. `StringIndexer` (categóricas → índices)
2. `OneHotEncoder` (índices → vectores one-hot)
3. `VectorAssembler` (numéricas + one-hot → vector final)

**Stages confirmados:**
`['StringIndexer', 'StringIndexer', 'StringIndexer', 'OneHotEncoder', 'OneHotEncoder', 'OneHotEncoder', 'VectorAssembler']`

Luego:
- Se entrenó el pipeline: `pipeline.fit(df_clean)`
- Se transformó el dataset: `pipeline_model.transform(df_clean)`
- Se renombró la variable objetivo a `label` (estándar Spark ML).

---

## ⭐ Bonus 1: Dimensión total de features post-encoding
Se verificó la dimensión final del vector generado:

- `tipo_de_contrato`: 16 categorías
- `departamento`: 34 categorías
- `ciudad`: 521 categorías  
Total one-hot: `16 + 34 + 521 = 571`

Sumando numéricas (`anio`, `mes`):
- `571 + 2 = 573`

✅ Dimensión real verificada:
- `features_raw` tiene **573 dimensiones**.

---

## ⭐ Bonus 2: Análisis de varianza de features
Se realizó un análisis simple de “importancia” mediante varianza:

- Se tomó una muestra (~1000 registros).
- Se convirtió `features_raw` a matriz NumPy.
- Se calculó varianza por dimensión.
- Se reportaron las **Top 5 features** con mayor varianza (índices: 18, 52, 53, 20, 55), indicando que son las que más cambian entre contratos y podrían aportar mayor información.

---

## Salidas del Notebook 03
Se guardaron artefactos clave para el flujo completo:

- **Pipeline entrenado:**
  - `/opt/spark-data/processed/feature_pipeline`

- **Dataset transformado listo para ML (con label y features_raw):**
  - `/opt/spark-data/processed/secop_features.parquet`

---

### Resultado final
Se dejó un proceso reproducible de Feature Engineering con Spark ML que:
- codifica variables categóricas,
- incorpora variables temporales,
- crea un vector final de **573 features**,
- y guarda pipeline + dataset transformado para entrenamiento en notebooks posteriores

## Fase 3: Modelos de Regresion

# Notebook 06: Regresion Logistica

### ✅ Reto 1: Crear variable objetivo binaria (definir criterio de riesgo)
1) Se cargó el dataset con features ya transformadas:
- Fuente: `/opt/spark-data/processed/secop_features.parquet`

2) Se estandarizó la columna de features:
- Como el dataset traía `features_raw`, se renombró a `features` (requerido por Spark ML).

3) Se definió el criterio de riesgo con percentil 90 del valor del contrato:
- Se calculó `p90` sobre `label` (valor del contrato):
  - **p90 = 94,640,000 COP**
- Se creó la variable binaria:
  - `riesgo = 1` si `label > p90` (alto riesgo)
  - `riesgo = 0` si `label <= p90` (bajo riesgo)

**Justificación:** El valor del contrato se usa como proxy de riesgo porque contratos de mayor cuantía suelen implicar mayor complejidad, exposición financiera y riesgo operativo/administrativo.

---

### ✅ Reto 2: Analizar balance de clases
Se evaluó la distribución:

- Clase 0 (bajo riesgo): **89,055 (89.1%)**
- Clase 1 (alto riesgo): **10,945 (10.9%)**

**Conclusión:** Dataset **desbalanceado** (esperado por construcción con p90).  
**Acciones sugeridas:** priorizar AUC/F1/Recall de clase 1, ajustar threshold y/o usar pesos de clase.

---

### ✅ Reto 3: Entender diferencia con regresión lineal
Se documentó que:

- **Regresión lineal** predice un valor continuo.
- **Regresión logística** predice **probabilidades** `P(clase=1 | X)` usando función sigmoide y produce clase final según un **threshold**.

---

### ✅ Reto 4: Configurar modelo con threshold apropiado
Se entrenó un clasificador de Regresión Logística con regularización L2:

- `maxIter = 200`
- `regParam = 0.1`
- `elasticNetParam = 0.0` (Ridge / L2)
- `threshold = 0.3` (bajado desde 0.5 para intentar mejorar detección de clase 1)

Se dividió el dataset:
- Train: **70,007**
- Test: **29,993**

---

### ✅ Reto 5: Interpretar probabilidades de predicción
Se inspeccionaron las columnas de salida del modelo:
- `rawPrediction`
- `probability` (vector [P(0), P(1)])
- `prediction`

Se extrajo `p1 = P(clase=1)` y se analizaron “casos dudosos”:
- Casos con `0.4 < p1 < 0.6`: **141 registros**
- Se revisaron ejemplos donde el modelo asigna clase 1 aunque `p1` esté cerca del umbral.

---

### ✅ Reto 6: Evaluar con AUC-ROC, Precision, Recall, F1
Se evaluó el modelo usando:

**AUC-ROC**
- **AUC = 0.7976** (buena capacidad de separación)

**Métricas Weighted (multiclase)**
- Accuracy: **0.8890**
- Weighted Precision: **0.8574**
- Weighted Recall: **0.8890**
- Weighted F1: **0.8475**

**Métricas específicas de la clase 1 (alto riesgo)**
- TP = 215, FP = 157, FN = 3171, TN = 26450
- Precision clase 1: **0.5780**
- Recall clase 1: **0.0635**
- F1 clase 1: **0.1144**

**Interpretación:** AUC es bueno, pero el **recall de clase 1 es muy bajo**, lo que indica dificultad para detectar alto riesgo (muchos FN).

---

### ✅ Reto 7: Construir e interpretar matriz de confusión
Matriz de confusión (test):

| label | prediction | count |
|------:|-----------:|------:|
| 0     | 0          | 26,450 |
| 0     | 1          | 157 |
| 1     | 0          | 3,171 |
| 1     | 1          | 215 |

- **TP=215**, **TN=26450**, **FP=157**, **FN=3171**

**Conclusión de impacto:**
- FP (falsa alarma): costo operativo extra.
- FN (no detectar alto riesgo): potencialmente **más grave**, porque deja pasar contratos riesgosos sin control.

---

## ⭐ Bonus 1: Experimentar con diferentes thresholds
Se justificó que, ante desbalance:
- bajar `threshold` suele aumentar recall de clase 1, a costa de más falsos positivos.
*(En el notebook se implementó threshold=0.3 como primer ajuste.)*

---

## ⭐ Bonus 2: Implementar curva ROC
Se generó curva ROC extrayendo `p1` y evaluando TPR/FPR para múltiples thresholds.
- Se trabajó con una muestra (10%) para evitar consumo excesivo de RAM.
- Archivo generado:
  - `/opt/spark-data/processed/roc_curve.png`
<img width="710" height="544" alt="image" src="https://github.com/user-attachments/assets/3d37dc12-edde-419e-851f-cfe1fdb84661" />

---

## Persistencia del modelo
Se guardó el modelo entrenado en:
- `/opt/spark-data/processed/logistic_regression_model`

---

## Resultado final
Se construyó un baseline de clasificación con Regresión Logística para “riesgo alto” basado en valor del contrato, logrando:
- **AUC-ROC: 0.7976** (buena separación global),
pero con un reto claro:
- **Recall muy bajo de la clase 1**, lo que sugiere que el siguiente paso debe enfocarse en:
  - tuning de threshold,
  - pesos de clase (`weightCol`),
  - features adicionales o modelos no lineales.
 
# Fase 5: MLOps y Produccion

## Notebook 10: MLflow Tracking

### ✅ Reto 1: Configurar MLflow tracking server y experimento
1) Se creó una `SparkSession` conectada al clúster:
- Master: `spark://spark-master:7077`

2) Se configuró el tracking server de MLflow (Docker):
- `mlflow.set_tracking_uri("http://mlflow:5000")`

3) Se creó/seleccionó el experimento:
- `experiment_name = "/SECOP_Contratos_Prediccion"`
- `mlflow.set_experiment(experiment_name)`
- Resultado: MLflow creó el experimento al no existir.

**Importancia (reflexión):** MLflow centraliza parámetros, métricas, modelos y artefactos en un historial comparable y reproducible, facilitando colaboración y control de versiones.

---

### ✅ Preparación de datos para entrenamiento
- Dataset cargado desde:
  - `/opt/spark-data/processed/secop_features.parquet`
- Se renombró `features_raw` → `features` (requisito de Spark ML).
- Se filtraron registros válidos:
  - `label` no nulo
- Split de datos:
  - Train: **79,954**
  - Test: **20,046**

Se configuró el evaluador base:
- `RegressionEvaluator(metricName="rmse")`

---

### ✅ Reto 2: Registrar experimento baseline (log_param / log_metric)
Se registró un baseline sin regularización:

**Run:** `baseline_no_regularization`  
**Parámetros loggeados:**
- `regParam = 0.0`
- `elasticNetParam = 0.0`
- `maxIter = 100`
- `model_type = LinearRegression`

**Entrenamiento:** `LinearRegression(featuresCol="features", labelCol="label")`  
**Métrica loggeada:**
- `rmse = 3,030,748,234.06`  
**Artefacto:**
- `mlflow.spark.log_model(model, "model")`

> Nota: aparecieron warnings esperados (p.ej. `regParam=0` puede generar inestabilidad numérica / overfitting).

---

### ✅ Reto 3: Registrar múltiples modelos (Ridge, Lasso, ElasticNet)
Se implementó un loop con 3 configuraciones:

1) **Ridge (L2)**
- Run: `ridge_l2`
- `regParam=0.1`, `elasticNetParam=0.0`
- RMSE: **3,030,754,693.01**
- MAE: **152,105,378.77**
- R²: **0.0029**

2) **Lasso (L1)**
- Run: `lasso_l1`
- `regParam=0.1`, `elasticNetParam=1.0`
- RMSE: **3,030,748,798.68**
- MAE: **152,110,708.07**
- R²: **0.0029**

3) **ElasticNet**
- Run: `elasticnet`
- `regParam=0.1`, `elasticNetParam=0.5`
- RMSE: **3,030,748,798.69**
- MAE: **152,110,708.62**
- R²: **0.0029**

**¿Por qué registrar múltiples métricas y no solo RMSE?**  
Porque:
- **RMSE** penaliza más errores grandes (sensible a outliers),
- **MAE** es más robusto y fácil de interpretar como error promedio,
- **R²** indica cuánta varianza explica el modelo.
Con una sola métrica puedes elegir un modelo “bueno” en RMSE pero pobre en estabilidad (MAE) o con explicación muy baja (R²).

---

### ✅ Reto 4: Explorar y comparar runs en MLflow UI
Se revisó el experimento en:
- `http://localhost:5000`

**Mejor modelo (menor RMSE):**
- **lasso_l1**
- **RMSE = 3,030,748,798.68**

**Observaciones principales:**
- Diferencias mínimas entre Lasso y ElasticNet; Ridge quedó apenas peor.
- No se observa una relación fuerte “más regularización = mejor” porque los resultados quedaron casi iguales.
- **R² ≈ 0.0029** en todos: el modelo explica muy poca variación → sugiere mejorar features, transformar target (p.ej. log(label)), o probar modelos no lineales.

**Cómo compartir con el equipo:**
- Enviar el link del experimento en MLflow + `Run ID` del mejor run
- Adjuntar reporte en artefactos (txt) y gráficos para discusión técnica.

---

### ✅ Reto 5: Agregar artefactos personalizados (reportes, gráficos)
Se creó el run:

**Run:** `model_with_artifacts` (usando la mejor configuración: Lasso)  
**Parámetros loggeados:**
- `regParam=0.1`
- `elasticNetParam=1.0`
- `maxIter=100`
- `regularization_type=Lasso`

**Métricas loggeadas:**
- RMSE: **3,030,748,798.68**
- MAE:  **152,110,708.07**
- R²:   **0.0029**

**Artefactos guardados:**
1) **Reporte de texto**
- `model_report.txt`
- Se guardó con: `mlflow.log_text(report, "model_report.txt")`

2) **Gráfico Predicho vs Real**
- Se generó con matplotlib usando una muestra de 5000 registros
- Archivo local: `/tmp/predictions_vs_real.png`
- Se guardó con: `mlflow.log_artifact(plot_path)`
- Resultado: `predictions_vs_real.png` visible en la pestaña **Artifacts** del run.

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/cecbf409-60aa-457f-b5b0-bffcfff0b1ab" />


3) **Modelo**
- `mlflow.spark.log_model(model, "model")`

---

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


