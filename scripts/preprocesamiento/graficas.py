import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def prueba1(X_train, y_class3):
    """
    Función para graficar la varianza de las variables numéricas.
    :param X_train: DataFrame de características de entrenamiento.
    :param y_class3: Serie de la variable objetivo (no se utiliza en esta función).
    """
    # 1. Filtrar columnas numéricas
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

    # 2. Calcular varianza y ordenarla
    variances = X_train[numerical_columns].var()
    variances_sorted = variances.sort_values().head(15)

    # 3. Definir colores personalizados
    color_arch1 = '#D5E8D4'
    color_arch2 = '#F8CECC'
    color_arch3 = '#DAE8FC'

    # 4. Asignar colores alternos a las barras
    colors = []
    for i in range(len(variances_sorted)):
        if i % 3 == 0:
            colors.append(color_arch1)
        elif i % 3 == 1:
            colors.append(color_arch2)
        else:
            colors.append(color_arch3)

    # 5. Graficar
    plt.figure(figsize=(12, 6))
    plt.bar(
        variances_sorted.index,
        variances_sorted.values,
        color=colors[:len(variances_sorted)],
        edgecolor='grey',
    )
    plt.xticks(rotation=45)
    plt.ylabel("Varianza")
    plt.title("Variables predictoras numéricas con menor varianza")
    plt.tight_layout()
    plt.show()

def prueba2(X_train, y_class3):
    """
    Función para graficar la correlación de las variables numéricas con la variable objetivo.
    """
    import matplotlib.pyplot as plt

    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    correlations_with_target = X_train[numerical_columns].corrwith(y_class3)

    correlations_df = correlations_with_target.reset_index()
    correlations_df.columns = ['Feature', 'Correlation']
    correlations_df = correlations_df.sort_values(by='Correlation', ascending=False)
    print(correlations_df)
    
    # Colores personalizados
    color_arch1 = '#D5E8D4'
    color_arch2 = '#F8CECC'
    color_arch3 = '#DAE8FC'
    colors = [color_arch1, color_arch2, color_arch3] * (len(correlations_df) // 3 + 1)
    colors = colors[:len(correlations_df)]

    # Crear figura con más espacio horizontal
    plt.figure(figsize=(14, 8))
    plt.bar(
        correlations_df['Feature'],
        correlations_df['Correlation'],
        color=colors,
        edgecolor='grey',
    )
    plt.xticks(rotation=45, ha='right')  # Inclinadas y alineadas
    plt.ylabel("Coeficiente de Correlación")
    plt.title("Correlación de las variables predictoras con la variable objetivo class3")

    # Ajustar márgenes inferiores para evitar corte de etiquetas
    plt.subplots_adjust(bottom=0.5, top=0.85)  # Más espacio en la parte inferior
    plt.tight_layout()
    plt.show()

def prueba3(X_train, y_class3):
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    # Colores personalizados
    color_arch1 = '#D5E8D4'  # verde claro
    color_arch2 = '#F8CECC'  # rojo claro
    color_arch3 = '#DAE8FC'  # azul claro
    custom_palette = [color_arch1, color_arch2, color_arch3]

    # Filtrar columnas numéricas
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_filtered = X_train[numerical_columns]

    # Matriz de correlación y distancia
    correlation_matrix = X_filtered.corr()
    distance_matrix = 1 - correlation_matrix

    # Clustering jerárquico
    linkage_matrix = sch.linkage(distance_matrix, method='ward')

    # Establecer paleta de colores personalizada
    sch.set_link_color_palette(custom_palette)

    # Crear figura más grande y compacta
    fig, ax = plt.subplots(figsize=(12, 6))  # Tamaño adecuado

    dendro = sch.dendrogram(
        linkage_matrix,
        labels=correlation_matrix.columns,
        orientation='top',
        leaf_rotation=90,
        above_threshold_color='#CCCCCC',
        color_threshold=0.7,
        ax=ax
    )

    # Títulos y etiquetas
    plt.title("Dendrograma de agrupamiento jerárquico de variables predictoras correlacionadas")
    plt.ylabel("Distancia jerárquica entre variables predictoras")

    # Ajustes para que no se corte nada
    plt.tight_layout()  # Ajusta márgenes automáticamente
    plt.subplots_adjust(top=0.9, bottom=0.3)  # Espacio adicional arriba/abajo

    plt.show()


    # Restaurar paleta por defecto
    sch.set_link_color_palette(None)

    # Agrupar variables por cluster
    threshold = 0.3
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    clusters = {}
    for label, column in zip(cluster_labels, correlation_matrix.columns):
        clusters.setdefault(label, []).append(column)

    for cluster_id, variables in clusters.items():
        if len(variables) > 1:
            print(f"Cluster {cluster_id}: {', '.join(variables)}")
    
def graficar_rf_importancia(feature_importances):
    import matplotlib.pyplot as plt

    # Colores personalizados
    colores = ['#DAE8FC', '#F8CECC', '#D5E8D4']

    # Ordenar por importancia ascendente
    features = feature_importances.sort_values(by='Importance', ascending=True)
    print(features)
    num_barras = len(features)

    # Colores repetidos
    colores_repetidos = colores * (num_barras // len(colores)) + colores[:num_barras % len(colores)]

    # Figura con más altura
    plt.figure(figsize=(14, 8))

    # Gráfico de barras verticales
    plt.bar(
        features['Feature'],
        features['Importance'],
        color=colores_repetidos,
        edgecolor='grey',
    )

    # Rotar etiquetas con inclinación y alineación
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Importancia')
    plt.title('Evaluación de la importancia de las variables con Random Forest')
    plt.subplots_adjust(bottom=0.5, top=0.85)  # Más espacio en la parte inferior
    plt.tight_layout()
    plt.show()

def prueba4(X_train, y_class3):
    # Entrenar el modelo RandomForest con los pesos
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_class3)

    # Obtener importancia de características
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    graficar_rf_importancia(feature_importances)
    
