# Informe de Análisis Exploratorio de Datos (EDA)

Este informe resume los hallazgos clave del Análisis Exploratorio de Datos (EDA) realizado en el dataset "Crude Suicide Rates," con un enfoque en la preparación de variables para modelos de Regresión y Clasificación.

## 1. Limpieza y Estructura de Datos

Se aplicaron las siguientes transformaciones para estandarizar el conjunto de datos:
* Se cargó el archivo sin encabezados y se eliminaron las primeras tres filas de metadatos.
* Se renombraron las columnas clave (`Country`, `Year`, `Suicide_Rate_Both_Sexes`, etc.).
* Las columnas de las tasas de suicidio se convirtieron a tipo de dato numérico (`float`).
* **Ausencia de Nulos:** Se confirmó que las variables objetivo (`Suicide_Rate_Both_Sexes`, `Suicide_Rate_Male`, `Suicide_Rate_Female`) **no contienen valores nulos** después de la limpieza.

## 2. Exploración para Modelos de Regresión

El objetivo de Regresión es predecir la tasa de suicidio como un valor continuo.

### Estadísticas Descriptivas de la Tasa de Suicidio (Ambos Sexos)

| Estadística | Valor |
| :--- | :--- |
| **count** | 914.00 |
| **mean** | 9.94 |
| **std** | 7.27 |
| **min** | 0.30 |
| **25%** | 4.90 |
| **50% (Mediana)** | 8.10 |
| **75%** | 12.70 |
| **max** | 52.60 |

**Conclusión de Regresión:** La media es de aproximadamente **10 por 100,000 habitantes**. La alta dispersión de datos (std de 7.27 y un rango amplio) indica una **variabilidad significativa** en las tasas de suicidio entre los países, haciendo que el dataset sea adecuado y desafiante para un modelo de regresión.

## 3. Exploración para Modelos de Clasificación

El objetivo de Clasificación es predecir si un país tiene un riesgo de suicidio **Alto (1) o Bajo (0)**.

### Distribución de Riesgo (Umbral: > 10 por 100,000)

| Clase | Conteo | Porcentaje |
| :--- | :--- | :--- |
| **0 (Riesgo Bajo)** | 569 | 62.25% |
| **1 (Riesgo Alto)** | 345 | 37.75% |

**Conclusión de Clasificación:** La distribución de clases muestra un **ligero desequilibrio** (62% vs 38%). El dataset es apto para un modelo de Clasificación, pero se recomienda que la evaluación de los resultados se centre en métricas sensibles al desequilibrio, como el **F1-score** o el **Recall**.