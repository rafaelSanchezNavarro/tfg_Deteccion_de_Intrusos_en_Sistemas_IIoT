param_grid = {
    'DecisionTreeClassifier': [
        {
            'criterion': ['gini', 'entropy'],  # Criterio para medir la calidad de la división
            'splitter': ['best', 'random'],  # Estrategia para dividir nodos
            'max_depth': [3, 5, 7, 10, None],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5, 10, 15, 20],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3, 5, 7, 10],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2', None, 0.5, 0.7],  # Número máximo de características consideradas en cada división
            'ccp_alpha': [0.0, 0.05, 0.1, 0.2, 0.3],  # Parámetro de complejidad para la poda
        }
    ],

    'RandomForestClassifier': [
        {
            'n_estimators': [25, 50, 75, 100, 125],  # Número de árboles en el bosque
            'criterion': ['gini', 'entropy'],  # Criterio para medir la calidad de la división
            'max_depth': [3, 5, 7, 10, None],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5, 10, 15, 20],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3, 5, 7, 10],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2', None, 0.5, 0.7],  # Número máximo de características consideradas en cada división
            'bootstrap': [True, False],  # Uso de muestreo con reemplazo
            'ccp_alpha': [0.0, 0.05, 0.1, 0.2, 0.3],  # Parámetro de complejidad para la poda
        }
    ],
    'GradientBoostingClassifier': [
        {
            'n_estimators': [100, 150],  # Número de árboles en el modelo
            'learning_rate': [0.01, 0.1, 0.2],  # Tasa de aprendizaje
            'max_depth': [3, 5],  # Profundidad máxima de los árboles
            'min_samples_split': [2, 5],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3],  # Mínimo número de muestras en una hoja
            'subsample': [0.8, 1.0],  # Porcentaje de muestras usadas en cada iteración
            'max_features': ['sqrt', 'log2'],  # Número máximo de características consideradas en cada división
            'ccp_alpha': [0.0, 0.05],  # Parámetro de complejidad para la poda
        }
    ],
    'AdaBoostClassifier': [
        {
            'n_estimators': [50, 100, 150],  # Número de clasificadores base
            'learning_rate': [0.01, 0.1, 1.0],  # Tasa de aprendizaje
            'algorithm': ['SAMME'],  # Algoritmo de boosting
            
        }
    ],
    'KNeighborsClassifier': [
        {
            'n_neighbors': [3, 5, 7, 10, 15, 20],                         # Número de vecinos
            'weights': ['uniform', 'distance'],                           # Cómo ponderar los vecinos
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],       # Algoritmo para buscar los vecinos más cercanos
            'leaf_size': [20, 30, 40],                                    # Tamaño de la hoja para el algoritmo 'ball_tree' o 'kd_tree'
            'p': [1, 2],                                                  # Parámetro para la distancia de Minkowski (1 = Manhattan, 2 = Euclidean)
            'metric': ['minkowski', 'euclidean', 'manhattan'],            # Métrica para la distancia
        }
    ],
    'SVC': [
        {
            'C': [0.01, 0.1, 1, 10, 100],                                 # Parámetro de regularización
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],               # Función kernel
            'degree': [2, 3, 4],                                          # Grado del kernel polinómico
            'gamma': ['scale', 'auto'],                                   # Coeficiente del kernel
            'probability': [True, False],                                 # Habilitar probabilidades
        }
    ],
    'GaussianNB': [
        {
            'priors': [None],
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
        }
    ],
    
}