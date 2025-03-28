import os
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.utils import compute_class_weight
from scripts.entrenamiento.entrenamiento_utils.grid import param_grid
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize
from modelos.diccionario_modelos import algorithms

def cargar_datos():
    """Carga todos los archivos procesados y los devuelve como DataFrames."""
    carpeta = r"datos/preprocesados"
    datos = {}

    # Cargar X_train
    path_X_train = os.path.join(carpeta, "X_train.csv")
    try:
        datos["X_train"] = pd.read_csv(path_X_train, low_memory=False)
        print(f"✅ X_train cargado: {datos['X_train'].shape[0]} filas, {datos['X_train'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_train}.")
        return None

    # Cargar X_val
    path_X_val = os.path.join(carpeta, "X_val.csv")
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

def clasificacion_binaria(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3, ensemble):

        y_train_class3 = y_train_class3.values.ravel()
        y_val_class3 = y_val_class3.values.ravel()
        
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train[boolean_cols] = X_train[boolean_cols].astype(float)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        if ensemble and grid:
            tree_model = model.estimators[0][1]
        
        if grid:
            # X_train_sampled = X_train.sample(n=10000, random_state=random_state)
            # y_train_class3_sampled  = y_train_class3.loc[X_train_sampled.index]
            
            # X_train = X_train.drop(index=X_train_sampled.index)
            # y_train_class3 = y_train_class3.drop(index=X_train_sampled.index)
            
            if ensemble:
                grid_model = tree_model
            else:
                grid_model = model
                
            grid_search = optimize(
                random_grid=random_grid,
                random_state=random_state,
                estimator=grid_model,
                X=X_train,
                y=y_train_class3,
                param_grid=param_grid[grid_model.__class__.__name__],
                n_iter=grid_n_iter,
                cv=validacion_grid,
                scoring='accuracy',
                n_jobs=-1,
            )
            
            grid_model = grid_search[0].best_estimator_
            print(f"➡️  Optimización completa para {grid_model.__class__.__name__}.")

            if ensemble:
                model.estimators[0] = ('mwbp', grid_model)
            else:
                model = grid_model
            
        print("➡️  Creando el pipeline...")
        pipeline = create_pipeline(
            model=model,  
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("➡️  Pipeline creado exitosamente.")
        
        # # Validación cruzada de 5 pliegues
        # print("➡️  Realizando validación cruzada de 5 pliegues...")
        # cv_scores = cross_val_score(pipeline, X_train, y_train_class3, cv=5, scoring='accuracy')
        # print("📈 CV scores:", cv_scores)
        # print("📈 Accuracy media (CV): {:.4f}".format(cv_scores.mean()))
        
        print("➡️  Entrenando el pipeline...")
        pipeline.fit(X_train, y_train_class3)
        print("➡️  Entrenamiento completo.")

        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class3 = pipeline.predict(X_val)
        print("➡️  Predicciones realizadas.")

        accuracy = accuracy_score(y_val_class3, y_pred_class3)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
        
        precision = precision_score(y_val_class3, y_pred_class3)
        print(f'📈 Precision (validacion): {precision:.4f}')
        
        recall = recall_score(y_val_class3, y_pred_class3)
        print(f'📈 Recall (validacion): {recall:.4f}')
        
        f1 = f1_score(y_val_class3, y_pred_class3)
        print(f'📈 F1 (validacion): {f1:.4f}')
        
        roc = roc_auc_score(y_val_class3, y_pred_class3)
        print(f'📈 ROC (validacion): {roc:.4f}')
        
        return pipeline, accuracy, precision, recall, f1, roc, y_pred_class3

def clasificacion_multiclase_categoria(random_state, model, X_train, X_val, y_train_class3, y_train_class2 , y_val_class2, y_pred_class3):
        

        y_train_class2 = y_train_class2.values.ravel()
        
        y_val_class2 = y_val_class2.values.ravel()
        
        
        class_names = np.unique(y_train_class2)
        
        weights = compute_class_weight('balanced', classes=class_names, y=y_train_class2)
        class_weights = dict(zip(class_names, weights))

        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train[boolean_cols] = X_train[boolean_cols].astype(int)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        print("➡️  Creando el pipeline multiclase tipo...")
        # model.set_params(class_weight=class_weights)
        pipeline = create_pipeline(
            model=model,  # Modelo del algoritmo final (ensemble)
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("➡️  Pipeline multiclase tipo creado exitosamente.")
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        print("➡️  Entrenando el pipeline multiclase tipo...")
        pipeline.fit(X_train, y_train_class2)
        print("➡️  Entrenamiento completo.")


        # Realizar predicciones
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class2 = pipeline.predict(X_val)
        print("➡️  Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class2, y_pred_class2)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
        
        return pipeline, y_pred_class2

def clasificacion_multiclase_tipo(random_state, model, X_train, X_val, y_train_class3, y_train_class1 , y_val_class1, y_pred_class3, y_pred_class2, y_train_class2 , y_val_class2):
        
        pipeline_tipos = {}
        
        y_train_class1 = y_train_class1.values.ravel()
        
        y_val_class1 = y_val_class1.values.ravel()
        

        class_names_tipo = np.unique(y_train_class1)
        weights = compute_class_weight('balanced', classes=class_names_tipo, y=y_train_class1)
        class_weights = dict(zip(class_names_tipo, weights))
        
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train[boolean_cols] = X_train[boolean_cols].astype(int)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

        model_clonado = clone(model)
        # model_clonado.set_params(class_weight=class_weights)
        pipeline = create_pipeline(
            model=model_clonado,
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        pipeline.fit(X_train, y_train_class1)


        # Realizar predicciones
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class1 = pipeline.predict(X_val)
        print("➡️  Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class1, y_pred_class1)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
    
        pipeline_tipos[model_clonado.__class__.__name__] = pipeline
        
            
        return pipeline_tipos, y_pred_class1
        
def main(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, ensemble, model_class2, model_class1):  
    print("🚀 Iniciando entrenamiento...")
    
    # Cargar todos los archivos de datos preprocesados
    datos = cargar_datos()

    X_train = datos["X_train"]
    X_val = datos["X_val"]

    y_train_class3 = datos["y_train_class3"]
    y_val_class3 = datos["y_val_class3"]
    
    y_train_class2 = datos["y_train_class2"]
    y_val_class2 = datos["y_val_class2"]
    
    y_train_class1 = datos["y_train_class1"]
    y_val_class1 = datos["y_val_class1"]
    
    # Entrenar el modelo
    pipeline_class3, accuracy, precision, recall, f1, roc, y_pred_class3 = clasificacion_binaria(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3, ensemble)
    pipeline_class2, y_pred_class2 = clasificacion_multiclase_categoria(random_state, model_class2, X_train, X_val, y_train_class3, y_train_class2 , y_val_class2, y_pred_class3)
    pipelines_class1, y_pred_class1_cat = clasificacion_multiclase_tipo(random_state, model_class1, X_train, X_val, y_train_class3, y_train_class1 , y_val_class1, y_pred_class3, y_pred_class2, y_train_class2 , y_val_class2)
    print("🎯 Entrenamiento finalizado")
    
    
    pipeline_class3 = pipeline_class3.named_steps['model']
    pipeline_class2 = pipeline_class2.named_steps['model']
    
    return pipeline_class3, accuracy, precision, recall, f1, roc, pipeline_class2, pipelines_class1
