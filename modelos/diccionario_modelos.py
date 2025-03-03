# Definir algoritmos basados en árboles
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB



algorithms = {
    "DecisionTreeClassifier": DecisionTreeClassifier,  # DT - Árbol de decisión
    "GaussianNB": GaussianNB,  # NB - Naïve Bayes
    "KNN": KNeighborsClassifier,  # KNN - K-vecinos más cercanos
    "SVC": SVC,  # SVM - Máquinas de vectores de soporte
    "LogisticRegression": LogisticRegression,  # LR - Regresión Logística
    "RandomForestClassifier": RandomForestClassifier,  # RF - Bosques Aleatorios
    "CalibratedClassifierCV": CalibratedClassifierCV,  # CalibratedClassifierCV
    "MultinomialNB": MultinomialNB,  # MultinomialNB
}