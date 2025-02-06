# Definir algoritmos basados en árboles
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


algorithms = {
    "DecisionTreeClassifier": DecisionTreeClassifier,  # Árbol de decisión simple
    "RandomForestClassifier": RandomForestClassifier,  # Bosque aleatorio
    # "GradientBoostingClassifier": GradientBoostingClassifier,  # Gradient Boosting
    # "AdaBoostClassifier": AdaBoostClassifier,  # AdaBoost
    # "XGBClassifier": XGBClassifier,  # XGBoost
}