import pandas as pd
import numpy as np

# 1. Cargar el dataset SIN NINGÚN ENCABEZADO (header=None).
df = pd.read_csv('crude suicide rates.csv', header=None)

# 2. Asignar los nombres correctos a las columnas.
df.rename(columns={
    0: 'Country',
    1: 'Year',
    5: 'Suicide_Rate_Both_Sexes',
    6: 'Suicide_Rate_Male',
    7: 'Suicide_Rate_Female'
}, inplace=True)

# 3. Eliminar las primeras tres filas que contienen metadatos y encabezados inútiles
df = df.iloc[3:].copy()

# 4. Convertir las tasas de suicidio a formato numérico (float)
df['Suicide_Rate_Both_Sexes'] = pd.to_numeric(df['Suicide_Rate_Both_Sexes'], errors='coerce')
df['Suicide_Rate_Male'] = pd.to_numeric(df['Suicide_Rate_Male'], errors='coerce')
df['Suicide_Rate_Female'] = pd.to_numeric(df['Suicide_Rate_Female'], errors='coerce')

# 5. Volver a inspeccionar los datos limpios (Parte de la Limpieza)
print("--- DATOS DESPUÉS DE LA LIMPIEZA FINAL ---")
print(df.head())
print("\n--- VALORES NULOS EN TASAS ---")
print(df[['Suicide_Rate_Both_Sexes', 'Suicide_Rate_Male', 'Suicide_Rate_Female']].isnull().sum())


# --- EXPLORACIÓN PARA REGRESIÓN (VARIABLE OBJETIVO CONTINUA) ---
print("\n\n#####################################################")
print("### EXPLORACIÓN PARA REGRESIÓN (TASA DE SUICIDIO) ###")
print("#####################################################")

# 6. Descripción estadística de la variable objetivo de Regresión
print("\nEstadísticas Descriptivas de la Tasa de Suicidio (Ambos Sexos):")
print(df['Suicide_Rate_Both_Sexes'].describe())


# --- EXPLORACIÓN PARA CLASIFICACIÓN (VARIABLE OBJETIVO BINARIA) ---

# 7. Crear la variable objetivo para CLASIFICACIÓN (Riesgo Alto/Bajo).
threshold = 10
df['High_Suicide_Risk'] = np.where(df['Suicide_Rate_Both_Sexes'] > threshold, 1, 0)

# 8. Contar la distribución de la nueva variable de Clasificación
print("\n#####################################################")
print("### EXPLORACIÓN PARA CLASIFICACIÓN (RIESGO ALTO/BAJO) ###")
print("#####################################################")
print(f"\nDistribución de Riesgo de Suicidio (Umbral > {threshold}):")
print(df['High_Suicide_Risk'].value_counts())
print("\nPorcentajes de la Clase (Para revisar balance):")
print(df['High_Suicide_Risk'].value_counts(normalize=True) * 100)