import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE

def matriz_correlacion(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    return correlation_matrix

def correlacion_pares(df, umbral):
    df = matriz_correlacion(df)
    # Toma solo la parte superior de la matriz para evitar duplicados
    upper_tri = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))

    # Identifica pares altamente correlacionados
    correlated_pairs = []
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, col] > umbral:
                correlated_pairs.append((row, col))

    # Selecciona las columnas a eliminar (de cada par, se elimina la que aparece como columna)
    alta_corr_pares = [col for col in upper_tri.columns if any(upper_tri[col] > umbral)]
    
    return alta_corr_pares

def correlacion_respecto_objetivo(df, target, umbral):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Calculamos la correlación con la variable objetivo
    target_correlation = df[numeric_cols].corrwith(target).abs().sort_values(ascending=True)

    # Nos quedamos solo con las características que tengan correlación >= 0.1
    baja_corr_respecto_obj = target_correlation[target_correlation < umbral].index.tolist()

    return baja_corr_respecto_obj

def seleccionar_variables_pca(X_train, X_val, n_components=0.95, num_top_features=10):
    """
    Aplica PCA para seleccionar las características más influyentes, pero mantiene los datos originales.
    
    Parámetros:
        - X_train: DataFrame de entrenamiento
        - X_val: DataFrame de validación
        - n_components: float/int, cantidad de componentes principales o porcentaje de varianza a retener
        - num_top_features: int, número de características más influyentes a seleccionar

    Retorna:
        - X_train_filtrado: DataFrame de entrenamiento con las características seleccionadas
        - X_val_filtrado: DataFrame de validación con las características seleccionadas
    """

    # Aplicar PCA (sin guardar la transformación)
    pca = PCA(n_components=n_components)
    pca.fit(X_train)  # Solo ajustamos el modelo, no transformamos los datos

    # Obtener nombres originales de las variables
    original_feature_names = np.array(X_train.columns)

    # Contador de importancia de características en PCA
    feature_counter = Counter()
    
    for comp in pca.components_:
        top_indices = np.argsort(np.abs(comp))[-num_top_features:]  # Índices de las más importantes
        top_features = original_feature_names[top_indices]  # Obtener nombres
        feature_counter.update(top_features)  # Contar ocurrencias

    # Seleccionar las variables más influyentes ordenadas por frecuencia de aparición
    variables_pca = [feature for feature, _ in feature_counter.most_common()]

    # Filtrar las variables seleccionadas en los conjuntos de datos
    X_train_filtrado = X_train[variables_pca]
    X_val_filtrado = X_val[variables_pca]
    
    return X_train_filtrado, X_val_filtrado

def seleccionar_variables_rfe(X_train, X_val, y_train, num_features=10):
    # Modelo base para RFE
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Aplicar RFE para seleccionar las 20 mejores características
    rfe = RFE(estimator=modelo_rf, n_features_to_select=num_features, step=10)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_val_rfe = rfe.transform(X_val)

    # Ver qué variables fueron seleccionadas
    selected_features = X_train.columns[rfe.support_]
    
    X_train_filtrado = X_train[selected_features]
    X_val_filtrado = X_val[selected_features]
    print("Características seleccionadas por RFE:", selected_features.tolist())
    
    return X_train_filtrado, X_val_filtrado

def seleccionar_variables_randomForest(X_train, X_val, y_train, sample_weight_train):
    # Entrenar el modelo RandomForest con los pesos
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train, sample_weight=sample_weight_train)

    # Obtener importancia de características
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    caracteristicas_imp_rf = feature_importances.head(150)
    caracteristicas_imp_rf = caracteristicas_imp_rf.Feature.to_list()
    # print(caracteristicas_imp_rf)

    X_train_processed = X_train[caracteristicas_imp_rf]
    X_val_processed = X_val[caracteristicas_imp_rf]
    
    X_train_processed.shape
    
    return X_train_processed, X_val_processed

def proyectar_tsne(X_train, X_val, n_components=2, perplexity=300, max_iter=5000, sample_size=300000, random_state=42):
    """
    Aplica T-SNE para proyectar los datos a un espacio de menor dimensión.
    
    Parámetros:
        - X_train: DataFrame de entrenamiento
        - X_val: DataFrame de validación
        - n_components: int, número de dimensiones del embedding
        - perplexity: float, parámetro de T-SNE
        - n_iter: int, número de iteraciones
        - sample_size: int, cantidad de instancias a muestrear para T-SNE
        - random_state: int, semilla para reproducibilidad

    Retorna:
        - X_train_tsne: DataFrame de entrenamiento con la proyección T-SNE
        - X_val_tsne: DataFrame de validación con la proyección T-SNE
    """
    # Concatenar datos y muestrear si es necesario
    X_all = pd.concat([X_train, X_val])
    if X_all.shape[0] > sample_size:
        X_all_sample = X_all.sample(n=sample_size, random_state=random_state)
    else:
        X_all_sample = X_all.copy()

    # Aplicar T-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    X_all_tsne = tsne.fit_transform(X_all_sample)
    df_tsne = pd.DataFrame(X_all_tsne, index=X_all_sample.index, columns=[f'tsne_{i+1}' for i in range(n_components)])
    
    # Separar proyección en conjuntos de entrenamiento y validación
    X_train_tsne = df_tsne.loc[df_tsne.index.intersection(X_train.index)]
    X_val_tsne = df_tsne.loc[df_tsne.index.intersection(X_val.index)]
    
    return X_train_tsne, X_val_tsne
