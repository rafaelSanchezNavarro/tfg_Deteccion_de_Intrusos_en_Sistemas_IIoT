import os
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize
from scripts.entrenamiento.entrenamiento_utils.diccionarios import algorithms, param_grid, n_iter, extern_kfold, intern_kfold, random_state


def cargar_datos():
    """Carga todos los archivos procesados y los devuelve como DataFrames."""
    carpeta = r"datos/procesados"
    datos = {}

    # Cargar X_train
    path_X_train = os.path.join(carpeta, "X_train_processed.csv")
    try:
        datos["X_train"] = pd.read_csv(path_X_train, low_memory=False)
        print(f"✅ X_train cargado: {datos['X_train'].shape[0]} filas, {datos['X_train'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_train}.")
        return None

    # Cargar X_val
    path_X_val = os.path.join(carpeta, "X_val_processed.csv")
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

def entrenar_modelo(X_train, X_val, y_train_class3, y_val_class3, y_train_class2, y_val_class2, y_train_class1, y_val_class1):
    # Identificar columnas categóricas, numéricas
    categorical_cols_processed = X_train.select_dtypes(include=['object']).columns
    numerical_cols_processed = X_train.select_dtypes(include=['float64', 'int64']).columns

    # Configurar Repeated Stratified K-Fold
    print("Configurando Repeated Stratified K-Fold...")
    cv = RepeatedStratifiedKFold(n_splits=extern_kfold, n_repeats=intern_kfold, random_state=random_state)
    print(f"Configuración de CV completa: {extern_kfold} pliegues, {intern_kfold} repeticiones, {n_iter} combinaciones de hiperparametros\n")


    X_train_sampled = X_train.sample(n=10000, random_state=random_state)
    y_train_class3_sampled  = y_train_class3.loc[X_train_sampled.index]

    # Eliminar las instancias muestreadas del conjunto original
    X_train = X_train.drop(X_train_sampled.index)
    y_train_class3 = y_train_class3.drop(X_train_sampled.index)

    best_overall_score = -np.inf

    # Optimización de hiperparámetros para cada algoritmo
    for name in algorithms:
        print(f"Iniciando optimización de hiperparámetros para: {name}")
        model = algorithms[name]()
        print(f"Modelo instanciado: {model}")

        grid_search = optimize(
            random_grid=True,
            estimator=model,
            X=X_train_sampled,
            y=y_train_class3_sampled,
            param_grid=param_grid[name],
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
        )
        print(f"Optimización completa para {name}.\n")

        best_score = grid_search[0].best_score_

        if best_score > best_overall_score:
            best_model_name = name
            model_with_best_params = grid_search[0].best_estimator_
            best_params = grid_search[0].best_params_
    print(f"Mejores parámetros para {best_model_name}: {best_params}\n")


    # Definir los modelos base para el ensemble (más rápidos)
    print("Definiendo clasificadores base para el ensemble...")
    clf1 = model_with_best_params
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    print(f"Clasificadores definidos: clf1={clf1}, clf2={clf2}, clf3={clf3}\n")

    # Definir el ensemble usando VotingClassifier
    print("Configurando VotingClassifier para el ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('mwbp', clf1), # Mejor modelo de arbol
            ('gnb', clf2),
            ('knn', clf3),
            # añadir mas diversidad de algoritmos que no esten basados en arboles
        ],
        voting='soft'  # Cambiar a 'hard' para votación mayoritaria
    )
    print(f"Ensemble configurado con los clasificadores: {[nombre for (nombre, _) in ensemble.estimators]}\n")


    # Crear el pipeline incluyendo preprocesamiento y el ensemble
    print("Creando el pipeline del ensemble...")
    ensemble_pipeline_class3 = create_pipeline(
        model=ensemble,  # Modelo del algoritmo final (ensemble)
        categorical_features=categorical_cols_processed,  # Columnas categóricas
        numerical_features=numerical_cols_processed,  # Columnas numéricas
        # feature_selection=rfe  # Añadir RFE al pipeline
    )
    print("Pipeline creado exitosamente.\n")


    # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
    print("Entrenando el pipeline del ensemble...")
    ensemble_pipeline_class3.fit(X_train, y_train_class3)
    print("Entrenamiento completo.\n")


    # Realizar predicciones
    print("Realizando predicciones en el conjunto de validación...")
    y_pred_class3 = ensemble_pipeline_class3.predict(X_val)
    print("Predicciones realizadas.\n")


    # Evaluar el rendimiento
    accuracy = accuracy_score(y_val_class3, y_pred_class3)
    print(f'Accuracy del Ensemble (validacion): {accuracy:.4f}\n')


def main():  
    print("\n🚀 Iniciando entrenamiento...\n")
    
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
    
    # Entrenar el modelo
    entrenar_modelo(X_train, X_val, y_train_class3, y_val_class3, y_train_class2, y_val_class2, y_train_class1, y_val_class1)

    
