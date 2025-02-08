# Definir algoritmos basados en árboles
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


algorithms = {
    "DecisionTreeClassifier": DecisionTreeClassifier,  # Árbol de decisión simple
    "RandomForestClassifier": RandomForestClassifier,  # Bosque aleatorio
    # "GradientBoostingClassifier": GradientBoostingClassifier,  # Gradient Boosting
    "AdaBoostClassifier": AdaBoostClassifier,  # AdaBoost
    # "XGBClassifier": XGBClassifier,  # XGBoost
    "KNN": KNeighborsClassifier,  # K-vecinos más cercanos
    "SVC": SVC,  # Máquinas de vectores de soporte
    'GaussianNB': GaussianNB
}