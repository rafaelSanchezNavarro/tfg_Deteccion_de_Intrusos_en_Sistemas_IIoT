import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, silhouette_score
from sklearn.ensemble import IsolationForest
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize

def cargar_datos():
    """Carga todos los archivos procesados y los devuelve como DataFrames."""
    carpeta = r"datos/preprocesados"
    datos = {}

    # Cargar X_train
    path_X_train = os.path.join(carpeta, "X_train.csv")
    try:
        datos["X_train"] = pd.read_csv(path_X_train, low_memory=False)
        print(f"✅ X_train cargado: {datos['X_train'].shape[0]} filas, {datos['X_train'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_train}.")
        return None

    # Cargar X_val
    path_X_val = os.path.join(carpeta, "X_val.csv")
    try:
        datos["X_val"] = pd.read_csv(path_X_val, low_memory=False)
        print(f"✅ X_val cargado: {datos['X_val'].shape[0]} filas, {datos['X_val'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_val}.")
        return None

    # Cargar y_train_class3
    path_y_train_class3 = os.path.join(carpeta, "y_train_class3.csv")
    try:
        datos["y_train_class3"] = pd.read_csv(path_y_train_class3, low_memory=False)
        print(f"✅ y_train_class3 cargado: {datos['y_train_class3'].shape[0]} filas, {datos['y_train_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class3}.")
        return None

    # Cargar y_val_class3
    path_y_val_class3 = os.path.join(carpeta, "y_val_class3.csv")
    try:
        datos["y_val_class3"] = pd.read_csv(path_y_val_class3, low_memory=False)
        print(f"✅ y_val_class3 cargado: {datos['y_val_class3'].shape[0]} filas, {datos['y_val_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class3}.")
        return None

    # Cargar y_train_class2
    path_y_train_class2 = os.path.join(carpeta, "y_train_class2.csv")
    try:
        datos["y_train_class2"] = pd.read_csv(path_y_train_class2, low_memory=False)
        print(f"✅ y_train_class2 cargado: {datos['y_train_class2'].shape[0]} filas, {datos['y_train_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class2}.")
        return None

    # Cargar y_val_class2
    path_y_val_class2 = os.path.join(carpeta, "y_val_class2.csv")
    try:
        datos["y_val_class2"] = pd.read_csv(path_y_val_class2, low_memory=False)
        print(f"✅ y_val_class2 cargado: {datos['y_val_class2'].shape[0]} filas, {datos['y_val_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class2}.")
        return None

    # Cargar y_train_class1
    path_y_train_class1 = os.path.join(carpeta, "y_train_class1.csv")
    try:
        datos["y_train_class1"] = pd.read_csv(path_y_train_class1, low_memory=False)
        print(f"✅ y_train_class1 cargado: {datos['y_train_class1'].shape[0]} filas, {datos['y_train_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class1}.")
        return None

    # Cargar y_val_class1
    path_y_val_class1 = os.path.join(carpeta, "y_val_class1.csv")
    try:
        datos["y_val_class1"] = pd.read_csv(path_y_val_class1, low_memory=False)
        print(f"✅ y_val_class1 cargado: {datos['y_val_class1'].shape[0]} filas, {datos['y_val_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class1}.")
        return None

    return datos


def main(random_state):  
    print("🚀 Iniciando entrenamiento...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_val = datos["X_val"]

    y_train_class3 = datos["y_train_class3"]
    y_val_class3 = datos["y_val_class3"]
    
    y_train_class2 = datos["y_train_class2"]
    y_val_class2 = datos["y_val_class2"]
    
    y_train_class1 = datos["y_train_class1"]
    y_val_class1 = datos["y_val_class1"]
    
    
    
    
    # 1️⃣ Filtrar el 80% de la clase "1" en entrenamiento
    df_train = pd.DataFrame(X_train)
    df_train["y"] = y_train_class3  # Agregar la etiqueta al dataframe

    # Separar clases
    df_class_1 = df_train[df_train["y"] == 1]
    df_class_0 = df_train[df_train["y"] == 0]

    # Seleccionar solo el 20% de la clase "1"
    df_class_1_sample = df_class_1.sample(frac=0.08, random_state=42)

    # Reunir los datos desbalanceados
    df_train_balanced = pd.concat([df_class_0, df_class_1_sample])

    # Separar nuevamente X_train e y_train
    X_train_balanced = df_train_balanced.drop(columns=["y"])
    y_train_balanced = df_train_balanced["y"]

    print(f"📉 Datos después del balanceo: {y_train_balanced.value_counts()}")
    
    
    
    
    
    
    # # 2️⃣ Entrenar Isolation Forest con el nuevo conjunto de entrenamiento
    # iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # iso_forest.fit(X_train_balanced)

    # # 3️⃣ Predecir en el conjunto original (sin modificar) para ver el impacto
    # y_train_anomaly = iso_forest.predict(X_train)
    # y_val_anomaly = iso_forest.predict(X_val)

    # # Convertir predicciones a etiquetas binarias (1 = normal, 0 = anomalía)
    # y_train_anomaly = (y_train_anomaly == 1).astype(int)
    # y_val_anomaly = (y_val_anomaly == 1).astype(int)

    # # 4️⃣ Evaluar rendimiento
    # print("\n📊 Reporte en X_train:")
    # print(classification_report(y_train_class3, y_train_anomaly))

    # print("\n📊 Reporte en X_val:")
    # print(classification_report(y_val_class3, y_val_anomaly))

    # print("\n🔍 Matriz de confusión en X_train:")
    # print(confusion_matrix(y_train_class3, y_train_anomaly))

    # print("\n🔍 Matriz de confusión en X_val:")
    # print(confusion_matrix(y_val_class3, y_val_anomaly))
    
    
    # 3️⃣ Entrenar DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Ajusta eps y min_samples según los datos
    y_train_cluster = dbscan.fit_predict(X_train_balanced)

    # Convertir etiquetas de DBSCAN (-1 = ruido, 0-N = clusters)
    y_train_dbscan = np.where(y_train_cluster == -1, 0, 1)

    # 4️⃣ Evaluar rendimiento
    print("\n📊 Reporte en X_train:")
    print(classification_report(y_train_balanced, y_train_dbscan))

    print("\n🔍 Matriz de confusión en X_train:")
    print(confusion_matrix(y_train_balanced, y_train_dbscan))

    # 5️⃣ Calcular Silhouette Score (para evaluar calidad del clustering)
    silhouette_train = silhouette_score(X_train_balanced, y_train_cluster)
    print(f"\n📈 Silhouette Score en X_train: {silhouette_train:.4f}")

    # 6️⃣ Visualizar Clusters (si hay 2D)
    plt.scatter(X_train_balanced[:, 0], X_train_balanced[:, 1], c=y_train_cluster, cmap="coolwarm", edgecolors="k")
    plt.title("Clusters detectados por DBSCAN en X_train")
    plt.show()