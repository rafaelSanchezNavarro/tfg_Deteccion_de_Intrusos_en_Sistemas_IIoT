# Definir algoritmos basados en árboles
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
random_state = 42

algorithms = {
    "DecisionTreeClassifier": DecisionTreeClassifier,  # Árbol de decisión simple
    "RandomForestClassifier": RandomForestClassifier,  # Bosque aleatorio
    # "GradientBoostingClassifier": GradientBoostingClassifier,  # Gradient Boosting
    # "AdaBoostClassifier": AdaBoostClassifier,  # AdaBoost
    # "XGBClassifier": XGBClassifier,  # XGBoost
}


param_grid = {
    'DecisionTreeClassifier': [
        {
            'criterion': ['gini', 'entropy', 'log_loss'],  # Criterio para medir la calidad de la división
            'splitter': ['best', 'random'],  # Estrategia para dividir nodos
            'max_depth': [3, 5, 10, None],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5, 10],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3, 5],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2', None],  # Número máximo de características consideradas en cada división
            'ccp_alpha': [0.0, 0.05, 0.1],  # Parámetro de complejidad para la poda
            'random_state': [random_state]  # Semilla aleatoria para reproducibilidad
        }
    ],
    'RandomForestClassifier': [
        {
            'n_estimators': [100, 150, 200, 300],  # Número de árboles en el bosque
            'criterion': ['gini', 'entropy', 'log_loss'],  # Criterio para medir la calidad de la división
            'max_depth': [3, 5, 10, None],  # Profundidad máxima del árbol
            'min_samples_split': [2, 5, 10],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3, 5],  # Mínimo número de muestras en una hoja
            'max_features': ['sqrt', 'log2', None],  # Número máximo de características consideradas en cada división
            'bootstrap': [True, False],  # Uso de muestreo con reemplazo
            'ccp_alpha': [0.0, 0.05, 0.1],  # Parámetro de complejidad para la poda
            'random_state': [random_state]  # Semilla aleatoria para reproducibilidad
        }
    ],
    'GradientBoostingClassifier': [
        {
            'n_estimators': [100, 150, 200, 300],  # Número de árboles en el modelo
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
            'max_depth': [3, 5, 10],  # Profundidad máxima de los árboles
            'min_samples_split': [2, 5, 10],  # Mínimo número de muestras para dividir un nodo
            'min_samples_leaf': [1, 3, 5],  # Mínimo número de muestras en una hoja
            'subsample': [0.6, 0.8, 1.0],  # Porcentaje de muestras usadas en cada iteración
            'max_features': ['sqrt', 'log2', None],  # Número máximo de características consideradas en cada división
            'ccp_alpha': [0.0, 0.05, 0.1],  # Parámetro de complejidad para la poda
            'random_state': [random_state]  # Semilla aleatoria para reproducibilidad
        }
    ],
    'AdaBoostClassifier': [
        {
            'n_estimators': [50, 100, 150, 200],  # Número de clasificadores base
            'learning_rate': [0.001, 0.01, 0.1, 1.0],  # Tasa de aprendizaje
            'algorithm': ['SAMME', 'SAMME.R'],  # Algoritmo de actualización de pesos
            'random_state': [random_state]  # Semilla aleatoria para reproducibilidad
        }
    ]
}

n_iter = 3
extern_kfold = 5
intern_kfold = 1