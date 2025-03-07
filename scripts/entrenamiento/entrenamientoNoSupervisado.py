import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import adjusted_rand_score


def outliers(df):
    df_numeric = df.select_dtypes(include=['float', 'int'])  # Seleccionar solo columnas numéricas

    threshold = 3  # Umbral de Z-score
    z_scores = np.abs(zscore(df_numeric))  # Cálculo del Z-score

    mask = (z_scores < threshold).all(axis=1)  # Filtrar filas sin outliers

    df_clean = df[mask]  # Conservar filas sin outliers, manteniendo todas las columnas

    print(f"Outliers eliminados con Z-score: {len(df) - len(df_clean)}")

    return df_clean  # Retornar el DataFrame limpio

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

    # Cargar X_test
    path_X_test = os.path.join(carpeta, "X_test_preprocesado.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None
    
    
    # Cargar y_train_class3
    path_y_train_class3 = os.path.join(carpeta, "y_train_class3.csv")
    try:
        datos["y_train_class3"] = pd.read_csv(path_y_train_class3, low_memory=False)
        print(f"✅ y_train_class3 cargado: {datos['y_train_class3'].shape[0]} filas, {datos['y_train_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class3}.")
        return None
    
    # Cargar y_train_class2
    path_y_train_class2 = os.path.join(carpeta, "y_train_class2.csv")
    try:
        datos["y_train_class2"] = pd.read_csv(path_y_train_class2, low_memory=False)
        print(f"✅ y_train_class2 cargado: {datos['y_train_class2'].shape[0]} filas, {datos['y_train_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class2}.")
        return None
    
    # Cargar y_val_class3
    path_y_val_class3 = os.path.join(carpeta, "y_val_class3.csv")
    try:
        datos["y_val_class3"] = pd.read_csv(path_y_val_class3, low_memory=False)
        print(f"✅ y_val_class3 cargado: {datos['y_val_class3'].shape[0]} filas, {datos['y_val_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class3}.")
        return None
    
    # Cargar y_test_class3
    path_y_test_class3 = os.path.join(carpeta, "y_test_class3.csv")
    try:
        datos["y_test_class3"] = pd.read_csv(path_y_test_class3, low_memory=False)
        print(f"✅ y_test_class3 cargado: {datos['y_test_class3'].shape[0]} filas, {datos['y_test_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class3}.")
        return None
    

    return datos

def main(random_state):  
    print("🚀 Iniciando entrenamiento...")
    
    # Cargar todos los archivos de datos preprocesados
    datos = cargar_datos()

    X_train = datos["X_train"]
    X_val = datos["X_val"]
    X_test = datos["X_test"]
    y_train_class3 = datos["y_train_class3"].values.ravel()
    y_train_class2 = datos["y_train_class2"].values.ravel()
    y_val = datos["y_val_class3"].values.ravel()
    y_test = datos["y_test_class3"].values.ravel()
    
    X_train_without_0 = X_train[y_train_class3 == 1]
    X_train_without_1 = X_train[y_train_class3 == 0]



    ###################################################################
    # ISOLATION FOREST
    ###################################################################

    # # Entrenar con X_train solo normal
    # iso_forest = IsolationForest(random_state=random_state, n_estimators=100, contamination=0.5)
    # iso_forest.fit(X_train)
    # # Evaluación en X_test
    # pred_test_todo = iso_forest.predict(X_test)
    
    # # Entrenar con X_train solo normal
    # iso_forest = IsolationForest(random_state=random_state, n_estimators=100, contamination='auto')
    # iso_forest.fit(X_train_without_1)
    # # Evaluación en X_test
    # pred_test_normales = iso_forest.predict(X_test)
    # pred_test_normales = (pred_test_normales == -1).astype(int) # Los normales serán 0 y las anomalías 1 PARA ENTRENAR CON SOLO NORMALES
    
    
    # # Entrenar con X_train solo ataques
    # iso_forest = IsolationForest(random_state=random_state, n_estimators=100, contamination='auto')
    # iso_forest.fit(X_train_without_0)
    # # Evaluación en X_test
    # pred_test_ataques = iso_forest.predict(X_test)
    # pred_test_ataques = (pred_test_ataques == 1).astype(int)  # Los normales serán 1 y las anomalías 0 PARA ENTRENAR CON SOLO ATAQUES

    
    # df_pred_test = pd.DataFrame(pred_test_todo)
    # df_pred_test.to_csv("predicciones/isolation_todo.csv", index=False)
    
    # df_pred_test = pd.DataFrame(pred_test_normales)
    # df_pred_test.to_csv("predicciones/isolation_normales.csv", index=False)
    
    # df_pred_test = pd.DataFrame(pred_test_ataques)
    # df_pred_test.to_csv("predicciones/isolation_ataques.csv", index=False)
    
    
    
    
    
    ###################################################################
    # DBSCAN
    ###################################################################
    
    # X_train = outliers(X_train)
    # X_train = X_train[:100000]
    
    # # Parámetros optimizados de ejemplo
    # eps = 1  # O calcular con el método de k-distances
    # min_samples = 5
    # # Aplicar DBSCAN
    # db = DBSCAN(eps=eps, min_samples=min_samples)
    # labels = db.fit_predict(X_train)

    # # Mostrar resultados
    # print(f"Clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)}")
    # print(f"Puntos clasificados como ruido (-1): {list(labels).count(-1)}")
    
    
    
    # # Reducir la dimensionalidad a 2D usando PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_train)

    # # Crear los gráficos
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # # Gráfico 1: Datos originales (sin color)
    # axes[0].scatter(X_pca[:, 0], X_pca[:, 1], marker='o', s=50, alpha=0.6)
    # axes[0].set_title('Datos Originales (Sin Clusters)')
    # axes[0].set_xlabel('Componente Principal 1')
    # axes[0].set_ylabel('Componente Principal 2')

    # # Gráfico 2: Clusters DBSCAN
    # scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.6)
    # axes[1].set_title('Clusters de DBSCAN')
    # axes[1].set_xlabel('Componente Principal 1')
    # axes[1].set_ylabel('Componente Principal 2')
    # fig.colorbar(scatter, ax=axes[1], label='Cluster ID')

    # plt.tight_layout()
    # plt.show()
    
    # # Evaluar qué tan buenos son los clusters comparados con las etiquetas reales (si las tienes)
    # ari = adjusted_rand_score(y_train_class3, clusters)
    # print(f"ARI Score: {ari}")
    
    # ari = adjusted_rand_score(y_train_class3[:100000], labels)
    # print(f"ARI: {ari}")
    
    
    ###################################################################
    # KMEANS
    ###################################################################
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    # Determinamos el número óptimo de clusters con el método del codo y la silhouette score
    # Calcular la inercia (suma de distancias cuadradas a los centroides)
    inertia = []
    silhouette_scores = []
    range_n_clusters = range(2, 21)  # Rango de posibles números de clusters

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=50, random_state=42)
        kmeans.fit(X_train)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))

    # Graficar el método del codo
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, inertia, marker='o', color='b')
    plt.title("Método del Codo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inercia (Suma de Distancias Cuadradas)")

    # Graficar la Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range_n_clusters, silhouette_scores, marker='o', color='g')
    plt.title("Silhouette Score")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()

    # Seleccionar el número de clusters que tiene el mayor silhouette score (o usar el codo)
    optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    print(f"El número óptimo de clusters es: {optimal_n_clusters}")

    # Entrenar el modelo con el número óptimo de clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, init='k-means++', max_iter=500, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(X_train)

    # Reducir a 2D con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    # Calcular ARI
    ari = adjusted_rand_score(y_train_class2, clusters)
    print(f"ARI: {ari}")

    # Graficar los clusters con el número óptimo de clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab20", alpha=0.6, s=100)
    plt.title(f"Clusters de K-Means (ARI: {ari:.2f})")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


    
    
if __name__ == "__main__":
    main(random_state=42)