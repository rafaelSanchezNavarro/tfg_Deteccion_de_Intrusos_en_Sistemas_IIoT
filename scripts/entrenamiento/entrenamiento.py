import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scripts.entrenamiento.entrenamiento_utils.grid import param_grid
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize


def cargar_datos():
    """Carga todos los archivos procesados y los devuelve como DataFrames."""
    carpeta = r"datos/preprocesados"
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

def entrenar_modelo(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3):
    
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train[boolean_cols] = X_train[boolean_cols].astype(int)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

        if grid:
            X_train_sampled = X_train.sample(n=10000, random_state=random_state)
            y_train_class3_sampled  = y_train_class3.loc[X_train_sampled.index]
            
            grid_search = optimize(
                random_grid=random_grid,
                random_state=random_state,
                estimator=model,
                X=X_train_sampled,
                y=y_train_class3_sampled,
                param_grid=param_grid[model.__class__.__name__],
                n_iter=grid_n_iter,
                cv=validacion_grid,
                scoring='accuracy',
                n_jobs=-1,
            )
            model = grid_search[0].best_estimator_
            print(f"Optimización completa para {model.__class__.__name__}.")

        print("Creando el pipeline del ensemble...")
        pipeline = create_pipeline(
            model=model,  # Modelo del algoritmo final (ensemble)
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("Pipeline creado exitosamente.")
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        print("Entrenando el pipeline del ensemble...")
        pipeline.fit(X_train, y_train_class3)
        print("Entrenamiento completo.")


        # Realizar predicciones
        print("Realizando predicciones en el conjunto de validación...")
        y_pred_class3 = pipeline.predict(X_val)
        print("Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class3, y_pred_class3)
        print(f'Accuracy (validacion): {accuracy:.4f}')
        
        # Guardar el modelo
        nombre_modelo = f"{model.__class__.__name__}_{accuracy:.4f}.pkl"
        path = os.path.join("modelos", nombre_modelo)
        joblib.dump(pipeline, path)
        print("Pipeline guardado exitosamente.\n")

def main(random_state, model, grid, validacion_grid, grid_n_iter, random_grid):  
    print("🚀 Iniciando entrenamiento...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_val = datos["X_val"]

    y_train_class3 = datos["y_train_class3"]
    y_val_class3 = datos["y_val_class3"]
    
    # Entrenar el modelo
    entrenar_modelo(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3)

    
