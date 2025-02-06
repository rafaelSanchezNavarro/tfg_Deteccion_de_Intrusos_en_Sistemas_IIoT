param_grid = {
    'DecisionTreeClassifier': [
        {
            'criterion': ['gini', 'entropy'],  # Criterio para medir la calidad de la división
            'splitter': ['best', 'random'],  # Estrategia para dividir nodos
            'max_depth': [3, 5],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2'],  # Número máximo de características consideradas en cada división
            'ccp_alpha': [0.0, 0.05],  # Parámetro de complejidad para la poda
        }
    ],
    'RandomForestClassifier': [
        {
            'n_estimators': [100, 150],  # Número de árboles en el bosque
            'criterion': ['gini', 'entropy'],  # Criterio para medir la calidad de la división
            'max_depth': [3, 5],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2'],  # Número máximo de características consideradas en cada división
            'bootstrap': [True, False],  # Uso de muestreo con reemplazo
            'ccp_alpha': [0.0, 0.05],  # Parámetro de complejidad para la poda
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
            'algorithm': ['SAMME', 'SAMME.R'],  # Algoritmo de actualización de pesos
        }
    ]
}