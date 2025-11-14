import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# Asegúrate de que yellowbrick está instalado: pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer 
import joblib 
import matplotlib.pyplot as plt
import warnings

# Suprimir advertencias molestas de KMeans y Matplotlib
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CARGA, LIMPIEZA Y PREPARACIÓN DE DATOS ---

df = pd.read_csv('crude suicide rates.csv', header=None)

# Renombrar columnas clave
df.rename(columns={
    0: 'Country',
    1: 'Year',
    # Columna 2 es la probabilidad de morir por enfermedades no contagiosas (NCDs)
    2: 'Prob_Dying_30_70_Total',
    5: 'Suicide_Rate_Both_Sexes',
    6: 'Suicide_Rate_Male',
    7: 'Suicide_Rate_Female'
}, inplace=True)

df = df.iloc[3:].copy()

# Convertir a numérico (variables de clustering)
df['Suicide_Rate_Both_Sexes'] = pd.to_numeric(df['Suicide_Rate_Both_Sexes'], errors='coerce')
df['Suicide_Rate_Male'] = pd.to_numeric(df['Suicide_Rate_Male'], errors='coerce')
df['Suicide_Rate_Female'] = pd.to_numeric(df['Suicide_Rate_Female'], errors='coerce')
df['Prob_Dying_30_70_Total'] = pd.to_numeric(df['Prob_Dying_30_70_Total'], errors='coerce')

# Rellenar nulos con la media
df.fillna(df.mean(numeric_only=True), inplace=True)

# Seleccionar y Escalar las Características
features = ['Suicide_Rate_Both_Sexes', 'Suicide_Rate_Male', 'Suicide_Rate_Female', 'Prob_Dying_30_70_Total']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. DETERMINAR EL K ÓPTIMO (Visualización) ---

print("--- Determinando K Óptimo (Yellowbrick) ---")
# Usamos el método de la inercia ('distortion') para el Elbow Plot
model = KMeans(random_state=42, n_init='auto') # Usamos n_init='auto' para evitar warnings
visualizer = KElbowVisualizer(model, k=(2, 11), metric='distortion', timings=False) 

visualizer.fit(X_scaled) 

# Guarda la gráfica en un archivo PNG para verla
visualizer.fig.savefig('elbow_plot.png')
print("Gráfica 'elbow_plot.png' guardada en la carpeta. Ábrela para determinar el K óptimo.")
plt.close(visualizer.fig) 

# --- 3. APLICAR CLUSTERING Y EVALUACIÓN ---

# NOTA: Usaremos k=4 como punto de partida para la ejecución
# AJUSTA este valor (optimal_k) después de revisar 'elbow_plot.png'.
optimal_k = 4 

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calcular Métricas de Rendimiento
inertia_value = kmeans.inertia_
# Calcula Silueta, solo si hay más de 1 cluster y más de 1 punto de datos
if len(np.unique(df['Cluster'])) > 1 and len(df) > 1:
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
else:
    silhouette_avg = np.nan # No es posible calcular

print(f"\n--- Métricas del Cluster (k={optimal_k}) ---")
print(f"Mejor Valor de Inercia (Distortion): {inertia_value:.2f}")
print(f"Mejor Valor de Silueta (Silhouette Score): {silhouette_avg:.4f}")

# --- 4. ANÁLISIS FINAL POR CLUSTER (Interpretación Médica) ---

print("\n--- Análisis Estadístico Final por Cluster ---")
cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
print(cluster_summary.to_string()) 

# --- 5. GUARDAR MODELOS PARA INTERFAZ WEB ---

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelos (K-Means y Escalador) guardados correctamente.")
